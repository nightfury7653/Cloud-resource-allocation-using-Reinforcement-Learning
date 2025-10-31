"""
Actor-Critic Network for PPO and A3C

Shared feature extractor with separate policy (actor) and value (critic) heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network Architecture.
    
    Architecture:
        Input → Shared Layers → Split:
            ├─ Policy Head (Actor) → Action probabilities
            └─ Value Head (Critic) → State value
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 shared_layers: List[int] = [256, 256],
                 policy_layers: List[int] = [128],
                 value_layers: List[int] = [128],
                 activation: str = 'tanh'):
        """
        Initialize Actor-Critic Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            shared_layers: Hidden units in shared layers
            policy_layers: Hidden units in policy head
            value_layers: Hidden units in value head
            activation: Activation function
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Shared feature extraction
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for hidden_dim in shared_layers:
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        shared_output_dim = input_dim
        
        # Policy head (actor)
        self.policy_layers = nn.ModuleList()
        input_dim = shared_output_dim
        
        for hidden_dim in policy_layers:
            self.policy_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.policy_output = nn.Linear(input_dim, action_dim)
        
        # Value head (critic)
        self.value_layers = nn.ModuleList()
        input_dim = shared_output_dim
        
        for hidden_dim in value_layers:
            self.value_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.value_output = nn.Linear(input_dim, 1)
        
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
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            state: State tensor
        
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Shared features
        x = state
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Policy head
        policy = x
        for layer in self.policy_layers:
            policy = self.activation(layer(policy))
        action_logits = self.policy_output(policy)
        
        # Value head
        value = x
        for layer in self.value_layers:
            value = self.activation(layer(value))
        state_value = self.value_output(value)
        
        return action_logits, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, select argmax action
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            states: Batch of states
            actions: Batch of actions
        
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        action_logits, values = self.forward(states)
        
        # Create distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Evaluate
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value only"""
        _, value = self.forward(state)
        return value
    
    def get_num_parameters(self) -> int:
        """Get total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (f"ActorCriticNetwork(\n"
                f"  state_dim={self.state_dim},\n"
                f"  action_dim={self.action_dim},\n"
                f"  params={self.get_num_parameters():,}\n"
                f")")

