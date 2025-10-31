#!/usr/bin/env python3
"""
Quick test script to verify DDQN implementation
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.realistic_cloud_env import RealisticCloudEnvironment
from agent.ddqn_agent import DDQNAgent
from networks.dueling_network import DuelingNetwork

def test_network():
    """Test Dueling Network"""
    print("\n" + "="*70)
    print("TEST 1: Dueling Network")
    print("="*70)
    
    state_dim = 50
    action_dim = 10
    
    network = DuelingNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_layers=[256, 256],
        value_layers=[128],
        advantage_layers=[128]
    )
    
    print(f"✓ Network created")
    print(network)
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    q_values = network(states)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {states.shape}")
    print(f"  Output shape: {q_values.shape}")
    
    # Test action selection
    state = torch.randn(state_dim)
    action = network.select_action(state, epsilon=0.1)
    
    print(f"✓ Action selection successful")
    print(f"  Selected action: {action}")
    
    print("✓ Network test PASSED\n")


def test_replay_buffer():
    """Test Replay Buffer"""
    print("="*70)
    print("TEST 2: Replay Buffer")
    print("="*70)
    
    from agent.replay_buffer import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Add transitions
    for i in range(100):
        state = np.random.randn(50)
        action = np.random.randint(0, 10)
        reward = np.random.randn()
        next_state = np.random.randn(50)
        done = np.random.rand() < 0.1
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"✓ Added 100 transitions")
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    
    print(f"✓ Sampled batch of 32")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    
    print("✓ Replay buffer test PASSED\n")


def test_agent():
    """Test DDQN Agent"""
    print("="*70)
    print("TEST 3: DDQN Agent")
    print("="*70)
    
    state_dim = 50
    action_dim = 10
    
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    print(f"✓ Agent created")
    print(f"  {agent}")
    
    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    
    print(f"✓ Action selection successful")
    print(f"  Action: {action}")
    
    # Store transitions
    for i in range(50):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"✓ Stored 50 transitions")
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    # Get stats
    stats = agent.get_stats()
    print(f"✓ Agent statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"    {key}: {value}")
    
    print("✓ Agent test PASSED\n")


def test_integration():
    """Test integration with environment"""
    print("="*70)
    print("TEST 4: Integration Test")
    print("="*70)
    
    # Create environment
    env = RealisticCloudEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"✓ Environment created")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create agent
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    print(f"✓ Agent created for environment")
    
    # Run one episode
    state, info = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 50
    
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done or truncated)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done or truncated:
            break
    
    print(f"✓ Completed {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    print("✓ Integration test PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DDQN IMPLEMENTATION TESTS")
    print("="*70)
    
    try:
        test_network()
        test_replay_buffer()
        test_agent()
        test_integration()
        
        print("="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nDDQN implementation is working correctly!")
        print("Ready to begin training with: python scripts/train_ddqn.py\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

