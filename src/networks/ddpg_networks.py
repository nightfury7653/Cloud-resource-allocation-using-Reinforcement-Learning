"""
DDPG Networks for Actor and Critic

Adapted for discrete actions using Gumbel-Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class DDPGActor(nn.Module):
    """
    DDPG Actor Network (Policy).
    
    For discrete actions, outputs logits that can be used with Gumbel-Softmax.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: List[int] = [256, 256],
                 activation: str = 'relu'):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: Hidden layer sizes
            activation: Activation function
        """
        super(DDPGActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = state
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Output logits for discrete actions
        action_logits = self.output(x)
        return action_logits
    
    def get_action(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get action using Gumbel-Softmax"""
        logits = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DDPGCritic(nn.Module):
    """
    DDPG Critic Network (Q-function).
    
    Takes state and action as input, outputs Q-value.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: List[int] = [256, 256],
                 activation: str = 'relu'):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for discrete: one-hot size)
            hidden_layers: Hidden layer sizes
            activation: Activation function
        """
        super(DDPGCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Build network (state + action as input)
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
            action: Action tensor (can be one-hot or soft for discrete)
        
        Returns:
            Q-value
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        for layer in self.layers:
            x = self.activation(layer(x))
        
        q_value = self.output(x)
        return q_value
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

