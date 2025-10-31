"""
DDPG Agent adapted for discrete actions

Uses Gumbel-Softmax for discrete action spaces.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import copy

try:
    from ..networks.ddpg_networks import DDPGActor, DDPGCritic
    from .replay_buffer import ReplayBuffer
except ImportError:
    from networks.ddpg_networks import DDPGActor, DDPGCritic
    from agent.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    DDPG Agent adapted for discrete actions.
    
    Uses soft actions (probabilities) instead of hard discrete actions.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """Initialize DDPG Agent"""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        if device is None:
            use_gpu = self.config['device']['use_gpu']
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DDPG Agent initialized on device: {self.device}")
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build networks
        self._build_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize replay buffer
        self._init_replay_buffer()
        
        # Training parameters
        self.gamma = self.config['training']['discount_factor']
        self.tau = self.config['training']['tau']
        self.batch_size = self.config['training']['batch_size']
        self.max_grad_norm = self.config['training']['max_grad_norm']
        
        # Exploration noise
        self.noise_sigma = self.config['exploration']['noise_sigma']
        self.noise_sigma_decay = self.config['exploration']['noise_sigma_decay']
        self.noise_sigma_min = self.config['exploration']['noise_sigma_min']
        
        # Training state
        self.episodes = 0
        self.steps = 0
        self.updates = 0
        
        # Statistics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'ddpg_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_networks(self):
        """Build actor and critic networks"""
        actor_config = self.config['actor']
        critic_config = self.config['critic']
        
        # Actor networks (policy)
        self.actor = DDPGActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_layers=actor_config['layers'],
            activation=actor_config['activation']
        ).to(self.device)
        
        self.actor_target = copy.deepcopy(self.actor)
        
        # Critic networks (Q-function)
        self.critic = DDPGCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_layers=critic_config['layers'],
            activation=critic_config['activation']
        ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        
        print(f"Actor: {self.actor.get_num_parameters():,} parameters")
        print(f"Critic: {self.critic.get_num_parameters():,} parameters")
    
    def _init_optimizers(self):
        """Initialize optimizers"""
        self.actor_lr = self.config['training']['learning_rate_actor']
        self.critic_lr = self.config['training']['learning_rate_critic']
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        print(f"Optimizers initialized: Actor LR={self.actor_lr}, Critic LR={self.critic_lr}")
    
    def _init_replay_buffer(self):
        """Initialize replay buffer"""
        buffer_size = self.config['training']['buffer_size']
        self.min_samples = self.config['training']['min_samples']
        
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        print(f"Replay buffer initialized (capacity: {buffer_size:,})")
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
        
        Returns:
            Action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor.get_action(state_tensor)
        
        # Add noise for exploration
        if add_noise and self.noise_sigma > 0:
            noise = torch.randn_like(action_probs) * self.noise_sigma
            action_probs = action_probs + noise
            action_probs = F.softmax(action_probs, dim=-1)
        
        # Sample action from probabilities
        action = torch.multinomial(action_probs, 1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1
        
        # Decay noise
        self.noise_sigma = max(self.noise_sigma_min, 
                              self.noise_sigma * self.noise_sigma_decay)
    
    def update(self) -> Optional[Dict[str, float]]:
        """Update actor and critic"""
        if not self.replay_buffer.is_ready(self.min_samples):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)
        
        # Convert actions to one-hot
        actions_onehot = F.one_hot(actions, self.action_dim).float()
        
        # Update Critic
        with torch.no_grad():
            next_action_probs = self.actor_target.get_action(next_states)
            next_q_values = self.critic_target(next_states, next_action_probs).squeeze(-1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.critic(states, actions_onehot).squeeze(-1)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update Actor
        action_probs = self.actor.get_action(states)
        actor_loss = -self.critic(states, action_probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        self.updates += 1
        
        # Store statistics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'noise_sigma': self.noise_sigma
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def episode_end(self, episode_reward: float):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(episode_reward)
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_sigma': self.noise_sigma,
            'episodes': self.episodes,
            'steps': self.steps,
            'updates': self.updates,
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'config': self.config,
            **kwargs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.noise_sigma = checkpoint['noise_sigma']
        self.episodes = checkpoint['episodes']
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.episode_rewards = checkpoint['episode_rewards']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Episodes: {self.episodes}, Steps: {self.steps}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 0 else [0]
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'updates': self.updates,
            'noise_sigma': self.noise_sigma,
            'avg_reward_100': np.mean(recent_rewards),
            'buffer_size': len(self.replay_buffer),
        }
    
    def __repr__(self) -> str:
        return (f"DDPGAgent(state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"episodes={self.episodes})")

