#!/usr/bin/env python3
"""
Test all RL algorithms before full training
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.realistic_cloud_env import RealisticCloudEnvironment

print("\n" + "="*70)
print("TESTING ALL RL ALGORITHMS")
print("="*70)

# Create environment once
print("\nCreating environment...")
env = RealisticCloudEnvironment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"✓ Environment created (state_dim={state_dim}, action_dim={action_dim})")

results = {}

# ==================== TEST DDQN ====================
print("\n" + "="*70)
print("TEST 1: DDQN (Double Deep Q-Network)")
print("="*70)

try:
    from agent.ddqn_agent import DDQNAgent
    
    print("Creating DDQN agent...")
    ddqn_agent = DDQNAgent(state_dim, action_dim)
    print(f"✓ DDQN agent created: {ddqn_agent}")
    
    # Test episode
    print("Running test episode (20 steps)...")
    state, info = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = ddqn_agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        ddqn_agent.store_transition(state, action, reward, next_state, done or truncated)
        total_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    print(f"✓ Completed {step+1} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Buffer size: {len(ddqn_agent.replay_buffer)}")
    print("✓ DDQN TEST PASSED")
    results['DDQN'] = 'PASS'
    
except Exception as e:
    print(f"❌ DDQN TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    results['DDQN'] = 'FAIL'

# ==================== TEST PPO ====================
print("\n" + "="*70)
print("TEST 2: PPO (Proximal Policy Optimization)")
print("="*70)

try:
    from agent.ppo_agent import PPOAgent
    
    print("Creating PPO agent...")
    ppo_agent = PPOAgent(state_dim, action_dim)
    print(f"✓ PPO agent created: {ppo_agent}")
    
    # Test episode
    print("Running test episode (20 steps)...")
    state, info = env.reset()
    total_reward = 0
    
    for step in range(20):
        action, log_prob, value = ppo_agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        ppo_agent.store_transition(state, action, reward, value, log_prob, done or truncated)
        total_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    # Test update
    print("Testing PPO update...")
    if ppo_agent.rollout_buffer.pos > 0:
        stats = ppo_agent.update()
        print(f"✓ Update successful")
        print(f"  Policy loss: {stats['policy_loss']:.4f}")
        print(f"  Value loss: {stats['value_loss']:.4f}")
    
    print(f"✓ Completed {step+1} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print("✓ PPO TEST PASSED")
    results['PPO'] = 'PASS'
    
except Exception as e:
    print(f"❌ PPO TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    results['PPO'] = 'FAIL'

# ==================== TEST A3C ====================
print("\n" + "="*70)
print("TEST 3: A3C (Asynchronous Advantage Actor-Critic)")
print("="*70)

try:
    from agent.a3c_agent import A3CAgent
    
    print("Creating A3C agent...")
    a3c_agent = A3CAgent(state_dim, action_dim)
    print(f"✓ A3C agent created: {a3c_agent}")
    
    # Test episode
    print("Running test episode (20 steps)...")
    state, info = env.reset()
    total_reward = 0
    
    for step in range(20):
        action, log_prob, value = a3c_agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        a3c_agent.store_transition(state, action, reward, value, log_prob, done or truncated)
        total_reward += reward
        state = next_state
        
        # Update every n_steps or at episode end
        if a3c_agent.should_update():
            print(f"Testing A3C update at step {step+1}...")
            stats = a3c_agent.update(last_value=0.0 if done else value)
            print(f"✓ Update successful")
            print(f"  Policy loss: {stats['policy_loss']:.4f}")
            print(f"  Value loss: {stats['value_loss']:.4f}")
        
        if done or truncated:
            break
    
    print(f"✓ Completed {step+1} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print("✓ A3C TEST PASSED")
    results['A3C'] = 'PASS'
    
except Exception as e:
    print(f"❌ A3C TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    results['A3C'] = 'FAIL'

# ==================== TEST DDPG ====================
print("\n" + "="*70)
print("TEST 4: DDPG (Deep Deterministic Policy Gradient)")
print("="*70)

try:
    from agent.ddpg_agent import DDPGAgent
    
    print("Creating DDPG agent...")
    ddpg_agent = DDPGAgent(state_dim, action_dim)
    print(f"✓ DDPG agent created: {ddpg_agent}")
    
    # Test episode
    print("Running test episode (20 steps)...")
    state, info = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = ddpg_agent.select_action(state, add_noise=True)
        next_state, reward, done, truncated, info = env.step(action)
        ddpg_agent.store_transition(state, action, reward, next_state, done or truncated)
        total_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    print(f"✓ Completed {step+1} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Buffer size: {len(ddpg_agent.replay_buffer)}")
    print(f"  Noise sigma: {ddpg_agent.noise_sigma:.4f}")
    print("✓ DDPG TEST PASSED")
    results['DDPG'] = 'PASS'
    
except Exception as e:
    print(f"❌ DDPG TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    results['DDPG'] = 'FAIL'

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

all_passed = True
for algo, status in results.items():
    icon = "✓" if status == "PASS" else "❌"
    print(f"{icon} {algo:<10} {status}")
    if status == "FAIL":
        all_passed = False

print("="*70)

if all_passed:
    print("\n🎉 ALL TESTS PASSED!")
    print("\nAll algorithms are ready for training!")
    print("\nStart training with:")
    print("  ./scripts/train_all_parallel.sh")
    print("\n" + "="*70 + "\n")
    sys.exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED!")
    print("\nPlease fix the failing algorithms before training.")
    print("="*70 + "\n")
    sys.exit(1)

