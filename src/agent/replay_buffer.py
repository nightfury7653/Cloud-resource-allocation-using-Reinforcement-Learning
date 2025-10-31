"""
Experience Replay Buffer for DDQN

Stores transitions and samples batches for training.
Implements uniform random sampling with optional prioritized replay.
"""

import numpy as np
import random
from collections import namedtuple, deque
from typing import List, Tuple, Optional
import torch


# Named tuple for storing transitions
Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN/DDQN.
    
    Stores transitions and provides random batch sampling for training.
    Uses a circular buffer (deque) for efficient memory management.
    """
    
    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to put tensors on ('cpu' or 'cuda')
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # Sample random batch
        transitions = random.sample(self.buffer, batch_size)
        
        # Transpose the batch
        # (see https://stackoverflow.com/a/19343/3343043 for details)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(np.array(batch.action)).to(device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_ready(self, min_samples: int) -> bool:
        """
        Check if buffer has enough samples to start training.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            True if buffer size >= min_samples
        """
        return len(self.buffer) >= min_samples
    
    def clear(self):
        """Clear all transitions from buffer"""
        self.buffer.clear()
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
            }
        
        # Extract rewards for statistics
        rewards = [t.reward for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD error priority.
    Implements importance sampling to correct for bias.
    
    Reference: Schaul et al. (2015) "Prioritized Experience Replay"
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full)
            beta_increment: Amount to increase beta per sample
            seed: Random seed
        """
        super().__init__(capacity, seed)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority storage (parallel to buffer)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add transition with maximum priority.
        
        New transitions get highest priority to ensure they're sampled.
        """
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """
        Sample batch based on priorities with importance sampling.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device for tensors
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max for stability
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(np.array(batch.action)).to(device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).to(device)
        weights_tensor = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights_tensor
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def get_stats(self) -> dict:
        """Get buffer statistics including priority info"""
        stats = super().get_stats()
        
        if len(self.priorities) > 0:
            priorities = np.array(self.priorities)
            stats.update({
                'avg_priority': np.mean(priorities),
                'max_priority': np.max(priorities),
                'min_priority': np.min(priorities),
                'beta': self.beta,
            })
        
        return stats

