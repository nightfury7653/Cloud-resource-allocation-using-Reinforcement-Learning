"""
Neural Network Architectures for RL Agents

This module contains network architectures:
- DuelingNetwork: Dueling DQN architecture for DDQN
- ActorCriticNetwork: For PPO and A3C (TODO)
- DDPGNetworks: Actor and Critic for DDPG (TODO)
"""

from .dueling_network import DuelingNetwork

__all__ = [
    'DuelingNetwork',
]

