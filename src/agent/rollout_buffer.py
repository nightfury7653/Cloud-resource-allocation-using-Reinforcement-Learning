"""
Rollout Buffer for PPO

Stores trajectories and computes advantages using GAE (Generalized Advantage Estimation).
"""

import numpy as np
import torch
from typing import Generator, Optional, Tuple


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, A3C).
    
    Stores complete trajectories and computes returns and advantages.
    """
    
    def __init__(self,
                 buffer_size: int,
                 state_dim: int,
                 action_dim: int = 1,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum buffer capacity
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device
        
        # Storage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        # Computed at end of rollout
        self.returns = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(self,
            state: np.ndarray,
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            done: bool):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for last state (bootstrap)
        """
        last_gae_lambda = 0
        
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            # TD error
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE
            last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            self.advantages[step] = last_gae_lambda
        
        # Returns = advantages + values (only up to pos)
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]
    
    def get(self, batch_size: Optional[int] = None) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Get batches of data.
        
        Args:
            batch_size: Size of mini-batches (None = full batch)
        
        Yields:
            Batches of (states, actions, old_values, old_log_probs, advantages, returns)
        """
        if not self.full and self.pos == 0:
            return
        
        # Get all stored data
        indices = np.arange(self.pos)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        if batch_size is None:
            batch_size = self.pos
        
        start_idx = 0
        while start_idx < self.pos:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield (
                torch.FloatTensor(self.states[batch_indices]).to(self.device),
                torch.LongTensor(self.actions[batch_indices]).to(self.device),
                torch.FloatTensor(self.values[batch_indices]).to(self.device),
                torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                torch.FloatTensor(self.returns[batch_indices]).to(self.device)
            )
            
            start_idx += batch_size
    
    def reset(self):
        """Reset buffer"""
        self.pos = 0
        self.full = False
    
    def __len__(self) -> int:
        """Get current buffer size"""
        return self.pos if not self.full else self.buffer_size

