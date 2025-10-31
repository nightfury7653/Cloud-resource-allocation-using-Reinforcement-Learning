"""
Dueling Network Architecture for DDQN

Separates Q-value estimation into:
1. Value function V(s): How good is the state
2. Advantage function A(s,a): How much better is each action

Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

Reference: Wang et al. (2016) "Dueling Network Architectures for Deep RL"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class DuelingNetwork(nn.Module):
    """
    Dueling DQN Network Architecture.
    
    Architecture:
        Input (state) → Shared Layers → Split into two streams:
            1. Value Stream → V(s)
            2. Advantage Stream → A(s,a)
        Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 shared_layers: List[int] = [256, 256],
                 value_layers: List[int] = [128],
                 advantage_layers: List[int] = [128],
                 activation: str = 'relu',
                 dueling: bool = True,
                 aggregation: str = 'mean'):
        """
        Initialize Dueling Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of actions)
            shared_layers: List of hidden units in shared layers
            value_layers: List of hidden units in value stream
            advantage_layers: List of hidden units in advantage stream
            activation: Activation function ('relu', 'tanh', 'elu')
            dueling: Whether to use dueling architecture
            aggregation: How to combine V and A ('mean' or 'max')
        """
        super(DuelingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.aggregation = aggregation
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for hidden_dim in shared_layers:
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        shared_output_dim = input_dim
        
        if self.dueling:
            # Build value stream
            self.value_stream = nn.ModuleList()
            input_dim = shared_output_dim
            
            for hidden_dim in value_layers:
                self.value_stream.append(nn.Linear(input_dim, hidden_dim))
                input_dim = hidden_dim
            
            # Value output (single value)
            self.value_output = nn.Linear(input_dim, 1)
            
            # Build advantage stream
            self.advantage_stream = nn.ModuleList()
            input_dim = shared_output_dim
            
            for hidden_dim in advantage_layers:
                self.advantage_stream.append(nn.Linear(input_dim, hidden_dim))
                input_dim = hidden_dim
            
            # Advantage output (one per action)
            self.advantage_output = nn.Linear(input_dim, action_dim)
        else:
            # Standard DQN architecture (no dueling)
            self.q_output = nn.Linear(shared_output_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'elu':
            return F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            Q-values of shape (batch_size, action_dim)
        """
        # Shared feature extraction
        x = state
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        if self.dueling:
            # Value stream
            v = x
            for layer in self.value_stream:
                v = self.activation(layer(v))
            value = self.value_output(v)  # Shape: (batch_size, 1)
            
            # Advantage stream
            a = x
            for layer in self.advantage_stream:
                a = self.activation(layer(a))
            advantage = self.advantage_output(a)  # Shape: (batch_size, action_dim)
            
            # Combine value and advantage
            if self.aggregation == 'mean':
                # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            elif self.aggregation == 'max':
                # Q(s,a) = V(s) + (A(s,a) - max(A(s,:)))
                q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        else:
            # Standard DQN (no dueling)
            q_values = self.q_output(x)
        
        return q_values
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-value for specific state-action pairs.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Q-values of shape (batch_size,)
        """
        q_values = self.forward(state)
        
        # Ensure action is 2D
        if action.dim() == 1:
            action = action.unsqueeze(1)
        
        # Gather Q-values for selected actions
        q_value = q_values.gather(1, action).squeeze(1)
        
        return q_value
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor of shape (state_dim,) or (1, state_dim)
            epsilon: Exploration probability
        
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def get_value_and_advantage(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get separate value and advantage outputs (for debugging).
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            Tuple of (value, advantage)
        """
        if not self.dueling:
            raise ValueError("Network must use dueling architecture")
        
        # Shared features
        x = state
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Value stream
        v = x
        for layer in self.value_stream:
            v = self.activation(layer(v))
        value = self.value_output(v)
        
        # Advantage stream
        a = x
        for layer in self.advantage_stream:
            a = self.activation(layer(a))
        advantage = self.advantage_output(a)
        
        return value, advantage
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> dict:
        """Get information about network layers"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'shared_layers': [layer.out_features for layer in self.shared_layers],
            'value_layers': [layer.out_features for layer in self.value_stream] if self.dueling else None,
            'advantage_layers': [layer.out_features for layer in self.advantage_stream] if self.dueling else None,
            'dueling': self.dueling,
            'aggregation': self.aggregation if self.dueling else None,
            'total_parameters': self.get_num_parameters(),
        }
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_layer_info()
        return (f"DuelingNetwork(\n"
                f"  state_dim={info['state_dim']},\n"
                f"  action_dim={info['action_dim']},\n"
                f"  shared={info['shared_layers']},\n"
                f"  value={info['value_layers']},\n"
                f"  advantage={info['advantage_layers']},\n"
                f"  dueling={info['dueling']},\n"
                f"  params={info['total_parameters']:,}\n"
                f")")

