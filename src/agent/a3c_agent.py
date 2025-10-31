"""
A3C (Asynchronous Advantage Actor-Critic) Agent

Simplified synchronous version (A2C) for easier implementation.
Can be extended to full async with threading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List

try:
    from ..networks.actor_critic_network import ActorCriticNetwork
except ImportError:
    from networks.actor_critic_network import ActorCriticNetwork


class A3CAgent:
    """
    A3C/A2C Agent (Synchronous version).
    
    Uses n-step returns and direct policy gradients.
    Simpler than PPO, but less stable.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize A3C Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config_path: Path to configuration file
            device: Device to use
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        if device is None:
            use_gpu = self.config['device']['use_gpu']
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"A3C Agent initialized on device: {self.device}")
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        self._build_network()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training parameters
        self.gamma = self.config['training']['discount_factor']
        self.n_steps = self.config['training']['n_steps']
        self.entropy_coef = self.config['training']['entropy_coef']
        self.value_coef = self.config['training']['value_coef']
        self.max_grad_norm = self.config['training']['max_grad_norm']
        
        # Storage for n-step returns
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []
        self.dones_buffer = []
        
        # Training state
        self.episodes = 0
        self.steps = 0
        self.updates = 0
        
        # Statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'a3c_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_network(self):
        """Build actor-critic network"""
        network_config = self.config['network']
        
        self.policy = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            shared_layers=network_config['shared_layers'],
            policy_layers=network_config['policy_layers'],
            value_layers=network_config['value_layers'],
            activation=network_config['activation']
        ).to(self.device)
        
        print(f"Network built: {self.policy.get_num_parameters():,} parameters")
    
    def _init_optimizer(self):
        """Initialize optimizer"""
        self.learning_rate = self.config['training']['learning_rate']
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        print(f"Optimizer initialized: Adam (LR={self.learning_rate})")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
        
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition in buffer"""
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.values_buffer.append(value)
        self.log_probs_buffer.append(log_prob)
        self.dones_buffer.append(done)
        self.steps += 1
    
    def should_update(self) -> bool:
        """Check if should perform update"""
        return len(self.states_buffer) >= self.n_steps or (len(self.dones_buffer) > 0 and self.dones_buffer[-1])
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Update policy using n-step returns.
        
        Args:
            last_value: Bootstrap value for last state
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.states_buffer) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Compute n-step returns
        returns = []
        R = last_value
        
        for i in reversed(range(len(self.rewards_buffer))):
            if self.dones_buffer[i]:
                R = 0
            R = self.rewards_buffer[i] + self.gamma * R
            returns.insert(0, R)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states_buffer)).to(self.device)
        actions = torch.LongTensor(self.actions_buffer).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs_buffer).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.values_buffer).to(self.device)
        
        # Evaluate actions
        log_probs, new_values, entropy = self.policy.evaluate_actions(states, actions)
        
        # Compute advantages
        advantages = returns_tensor - values.detach()
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = (new_values - returns_tensor).pow(2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear buffers
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []
        self.dones_buffer = []
        
        self.updates += 1
        
        # Store statistics
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        
        self.policy_losses.append(policy_loss_val)
        self.value_losses.append(value_loss_val)
        
        return {
            'policy_loss': policy_loss_val,
            'value_loss': value_loss_val,
            'entropy': entropy.mean().item()
        }
    
    def episode_end(self, episode_reward: float):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(episode_reward)
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'episodes': self.episodes,
            'steps': self.steps,
            'updates': self.updates,
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'config': self.config,
            **kwargs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate = checkpoint['learning_rate']
        self.episodes = checkpoint['episodes']
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.episode_rewards = checkpoint['episode_rewards']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Episodes: {self.episodes}, Steps: {self.steps}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 0 else [0]
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'updates': self.updates,
            'learning_rate': self.learning_rate,
            'avg_reward_100': np.mean(recent_rewards),
        }
    
    def __repr__(self) -> str:
        return (f"A3CAgent(state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"episodes={self.episodes})")

