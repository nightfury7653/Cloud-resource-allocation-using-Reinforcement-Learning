"""
PPO (Proximal Policy Optimization) Agent

State-of-the-art on-policy RL algorithm with clipped surrogate objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    from ..networks.actor_critic_network import ActorCriticNetwork
    from .rollout_buffer import RolloutBuffer
except ImportError:
    from networks.actor_critic_network import ActorCriticNetwork
    from agent.rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    
    Uses clipped surrogate objective for stable policy updates.
    On-policy algorithm with actor-critic architecture.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize PPO Agent.
        
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
        
        print(f"PPO Agent initialized on device: {self.device}")
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        self._build_network()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize rollout buffer
        self._init_rollout_buffer()
        
        # Training parameters
        self.gamma = self.config['training']['discount_factor']
        self.gae_lambda = self.config['training']['gae_lambda']
        self.clip_epsilon = self.config['training']['clip_epsilon']
        self.n_epochs = self.config['training']['n_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.entropy_coef = self.config['training']['entropy_coef']
        self.value_coef = self.config['training']['value_coef']
        self.max_grad_norm = self.config['training']['max_grad_norm']
        self.clip_value = self.config['training']['clip_value']
        self.value_clip = self.config['training']['value_clip']
        
        # Training state
        self.episodes = 0
        self.steps = 0
        self.updates = 0
        
        # Statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'ppo_config.yaml'
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
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
        training_config = self.config['training']
        
        self.learning_rate = training_config['learning_rate']
        self.lr_decay = training_config['lr_decay']
        self.min_lr = training_config['min_learning_rate']
        
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )
        
        print(f"Optimizer initialized: Adam (LR={self.learning_rate})")
    
    def _init_rollout_buffer(self):
        """Initialize rollout buffer"""
        rollout_config = self.config['rollout']
        
        self.rollout_buffer = RolloutBuffer(
            buffer_size=rollout_config['buffer_size'],
            state_dim=self.state_dim,
            action_dim=1,  # Discrete action
            gae_lambda=rollout_config['gae_lambda'],
            gamma=rollout_config['gamma'],
            device=self.device
        )
        
        print(f"Rollout buffer initialized (size: {rollout_config['buffer_size']})")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, select best action
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
        
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition in rollout buffer"""
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
        self.steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO clipped objective.
        
        Returns:
            Dictionary of training statistics
        """
        # Compute returns and advantages
        last_value = 0.0  # Bootstrap value (0 if episode ended)
        self.rollout_buffer.compute_returns_and_advantages(last_value)
        
        # Normalize advantages
        advantages = torch.FloatTensor(self.rollout_buffer.advantages[:len(self.rollout_buffer)]).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        kl_divs = []
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Sample mini-batches
            for batch in self.rollout_buffer.get(self.batch_size):
                states, actions, old_values, old_log_probs, batch_advantages, returns = batch
                
                # Normalize advantages (again, per batch)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.clip_value:
                    # Clip value updates
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values, -self.value_clip, self.value_clip
                    )
                    value_loss1 = (values - returns).pow(2)
                    value_loss2 = (value_pred_clipped - returns).pow(2)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = (values - returns).pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # Clip fraction (how often we clipped)
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())
                    
                    # Approximate KL divergence
                    kl_div = (old_log_probs - log_probs).mean()
                    kl_divs.append(kl_div.item())
        
        # Reset buffer
        self.rollout_buffer.reset()
        self.updates += 1
        
        # Store statistics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'clip_fraction': np.mean(clip_fractions),
            'kl_divergence': np.mean(kl_divs)
        }
    
    def decay_learning_rate(self):
        """Decay learning rate"""
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def episode_end(self, episode_reward: float):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(episode_reward)
        
        # Decay learning rate
        if self.episodes % 10 == 0:
            self.decay_learning_rate()
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save agent checkpoint"""
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
            'entropies': self.entropies,
            'config': self.config,
            **kwargs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint"""
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
        self.entropies = checkpoint['entropies']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Episodes: {self.episodes}, Steps: {self.steps}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 0 else [0]
        recent_policy_loss = self.policy_losses[-100:] if len(self.policy_losses) > 0 else [0]
        recent_value_loss = self.value_losses[-100:] if len(self.value_losses) > 0 else [0]
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'updates': self.updates,
            'learning_rate': self.learning_rate,
            'avg_reward_100': np.mean(recent_rewards),
            'avg_policy_loss_100': np.mean(recent_policy_loss),
            'avg_value_loss_100': np.mean(recent_value_loss),
            'buffer_size': len(self.rollout_buffer),
        }
    
    def __repr__(self) -> str:
        return (f"PPOAgent(state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"episodes={self.episodes})")

