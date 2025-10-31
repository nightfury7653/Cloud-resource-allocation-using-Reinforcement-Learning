"""
DDQN Agent Implementation

Double Deep Q-Network with Dueling Architecture for cloud resource allocation.

Key Features:
- Double Q-learning to reduce overestimation bias
- Dueling architecture for better value estimation
- Experience replay for sample efficiency
- Epsilon-greedy exploration
- Target network with periodic updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import copy

# Handle both relative and absolute imports
try:
    from ..networks.dueling_network import DuelingNetwork
    from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
except ImportError:
    from networks.dueling_network import DuelingNetwork
    from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DDQNAgent:
    """
    Double Deep Q-Network Agent with Dueling Architecture.
    
    Learns to allocate tasks to VMs by maximizing cumulative reward.
    Uses Double Q-learning to address overestimation and Dueling
    architecture to separate value and advantage learning.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize DDQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of VMs)
            config_path: Path to configuration file
            device: Device to use ('cpu' or 'cuda')
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        if device is None:
            use_gpu = self.config['device']['use_gpu']
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DDQN Agent initialized on device: {self.device}")
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build networks
        self._build_networks()
        
        # Initialize replay buffer
        self._init_replay_buffer()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training parameters
        self.gamma = self.config['training']['discount_factor']
        self.batch_size = self.config['training']['batch_size']
        self.target_update_freq = self.config['training']['target_update_frequency']
        self.gradient_clip = self.config['training']['gradient_clip']
        
        # Exploration parameters
        self.epsilon = self.config['exploration']['epsilon_start']
        self.epsilon_end = self.config['exploration']['epsilon_end']
        self.epsilon_decay = self.config['exploration']['epsilon_decay']
        
        # Double Q-learning
        self.use_double_q = self.config['double_q']['enabled']
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        self.losses = []
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            # Use default config path
            config_path = Path(__file__).parent.parent.parent / 'config' / 'ddqn_config.yaml'
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _build_networks(self):
        """Build main and target networks"""
        network_config = self.config['network']
        
        # Main network (online network)
        self.policy_net = DuelingNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            shared_layers=network_config['shared_layers'],
            value_layers=network_config['value_stream'][:-1],  # Exclude output
            advantage_layers=network_config['advantage_stream'][:-1],
            activation=network_config['activation'],
            dueling=self.config['dueling']['enabled'],
            aggregation=self.config['dueling']['aggregation']
        ).to(self.device)
        
        # Target network
        self.target_net = DuelingNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            shared_layers=network_config['shared_layers'],
            value_layers=network_config['value_stream'][:-1],
            advantage_layers=network_config['advantage_stream'][:-1],
            activation=network_config['activation'],
            dueling=self.config['dueling']['enabled'],
            aggregation=self.config['dueling']['aggregation']
        ).to(self.device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained
        
        print(f"Networks built: {self.policy_net.get_num_parameters():,} parameters")
    
    def _init_replay_buffer(self):
        """Initialize experience replay buffer"""
        buffer_config = self.config['replay_buffer']
        seed = self.config['random_seed']['seed'] if self.config['random_seed']['enabled'] else None
        
        if buffer_config['prioritized']:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_config['capacity'],
                alpha=buffer_config['priority_alpha'],
                beta=buffer_config['priority_beta'],
                beta_increment=buffer_config['priority_beta_increment'],
                seed=seed
            )
            print(f"Prioritized Replay Buffer initialized (capacity: {buffer_config['capacity']:,})")
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_config['capacity'],
                seed=seed
            )
            print(f"Replay Buffer initialized (capacity: {buffer_config['capacity']:,})")
        
        self.min_samples = buffer_config['min_samples']
    
    def _init_optimizer(self):
        """Initialize optimizer"""
        training_config = self.config['training']
        
        self.learning_rate = training_config['learning_rate']
        self.lr_decay = training_config['learning_rate_decay']
        self.min_lr = training_config['min_learning_rate']
        
        if training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.learning_rate,
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.policy_net.parameters(),
                lr=self.learning_rate,
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
        
        print(f"Optimizer initialized: {training_config['optimizer']}")
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Select action using policy network
        action = self.policy_net.select_action(state_tensor, epsilon)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one gradient descent step.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        # Check if enough samples
        if not self.replay_buffer.is_ready(self.min_samples):
            return None
        
        # Sample batch from replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size, self.device)
            use_importance_weights = True
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size, self.device)
            weights = torch.ones(self.batch_size).to(self.device)
            use_importance_weights = False
        
        # Compute current Q values
        current_q_values = self.policy_net.get_q_value(states, actions)
        
        # Compute next Q values
        with torch.no_grad():
            if self.use_double_q:
                # Double Q-learning: use policy net to select actions,
                # target net to evaluate them
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net.get_q_value(next_states, next_actions.squeeze(1))
            else:
                # Standard DQN: use target net for both selection and evaluation
                next_q_values = self.target_net(next_states).max(dim=1)[0]
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if use_importance_weights:
            # Weighted loss for prioritized replay
            loss = (weights * td_errors.pow(2)).mean()
            
            # Update priorities
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        else:
            # Standard MSE loss
            loss = td_errors.pow(2).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update steps and losses
        self.steps += 1
        self.losses.append(loss.item())
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def decay_learning_rate(self):
        """Decay learning rate"""
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def episode_end(self, episode_reward: float, episode_length: int):
        """
        Called at the end of each episode.
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in the episode
        """
        self.episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Decay learning rate
        if self.episodes % 10 == 0:
            self.decay_learning_rate()
    
    def save_checkpoint(self, path: str, **kwargs):
        """
        Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'losses': self.losses,
            'config': self.config,
            **kwargs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load agent checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.learning_rate = checkpoint['learning_rate']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.epsilon_history = checkpoint['epsilon_history']
        self.losses = checkpoint['losses']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Episodes: {self.episodes}, Steps: {self.steps}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 0 else [0]
        recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) > 0 else [0]
        recent_losses = self.losses[-1000:] if len(self.losses) > 0 else [0]
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'avg_reward_100': np.mean(recent_rewards),
            'avg_length_100': np.mean(recent_lengths),
            'avg_loss_1000': np.mean(recent_losses),
            'buffer_size': len(self.replay_buffer),
            'buffer_stats': self.replay_buffer.get_stats(),
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"DDQNAgent(state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"episodes={self.episodes}, "
                f"steps={self.steps})")

