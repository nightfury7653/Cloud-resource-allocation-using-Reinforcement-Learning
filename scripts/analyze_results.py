#!/usr/bin/env python3
"""
Analyze DDQN training results
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINT ANALYSIS: {Path(checkpoint_path).name}")
    print(f"{'='*70}")
    
    # Basic info
    print(f"\nTraining Progress:")
    print(f"  Episodes completed:     {checkpoint['episodes']}")
    print(f"  Total steps:            {checkpoint['global_step']}")
    print(f"  Best reward achieved:   {checkpoint['best_reward']:.2f}")
    print(f"  Best acceptance rate:   {checkpoint['best_acceptance']:.2%}")
    
    # Current state
    print(f"\nAgent State:")
    print(f"  Current epsilon:        {checkpoint['epsilon']:.4f}")
    print(f"  Current learning rate:  {checkpoint['learning_rate']:.6f}")
    
    # Episode statistics
    if 'episode_rewards' in checkpoint and len(checkpoint['episode_rewards']) > 0:
        rewards = checkpoint['episode_rewards']
        print(f"\nReward Statistics:")
        print(f"  Total episodes:         {len(rewards)}")
        print(f"  Average reward:         {np.mean(rewards):.2f}")
        print(f"  Std deviation:          {np.std(rewards):.2f}")
        print(f"  Min reward:             {np.min(rewards):.2f}")
        print(f"  Max reward:             {np.max(rewards):.2f}")
        
        # Last 10 episodes
        if len(rewards) >= 10:
            last_10 = rewards[-10:]
            print(f"\nLast 10 Episodes:")
            print(f"  Average reward:         {np.mean(last_10):.2f}")
            print(f"  Best reward:            {np.max(last_10):.2f}")
            print(f"  Worst reward:           {np.min(last_10):.2f}")
    
    # Loss statistics
    if 'losses' in checkpoint and len(checkpoint['losses']) > 0:
        losses = checkpoint['losses']
        print(f"\nLoss Statistics:")
        print(f"  Total updates:          {len(losses)}")
        print(f"  Average loss:           {np.mean(losses):.4f}")
        print(f"  Final loss (last 100):  {np.mean(losses[-100:]):.4f}")
    
    # Epsilon history
    if 'epsilon_history' in checkpoint and len(checkpoint['epsilon_history']) > 0:
        eps_hist = checkpoint['epsilon_history']
        print(f"\nExploration Progress:")
        print(f"  Initial epsilon:        {eps_hist[0]:.4f}")
        print(f"  Final epsilon:          {eps_hist[-1]:.4f}")
        print(f"  Decay rate achieved:    {(eps_hist[0] - eps_hist[-1]) / eps_hist[0]:.1%}")
    
    return checkpoint

def compare_with_baseline():
    """Compare with baseline results"""
    baseline_file = Path("results/baseline_evaluation.json")
    
    if not baseline_file.exists():
        print("\nBaseline results not found for comparison")
        return
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH BASELINES")
    print(f"{'='*70}")
    
    # Get best baseline (Least-Loaded)
    best_baseline = baseline.get('Least-Loaded', {})
    
    print(f"\nBest Baseline (Least-Loaded):")
    print(f"  Acceptance Rate:        {best_baseline.get('acceptance_rate', 0):.2%}")
    print(f"  Avg Utilization:        {best_baseline.get('avg_utilization', 0):.2%}")
    print(f"  Avg Reward:             {best_baseline.get('avg_reward', 0):.2f}")
    
    print(f"\nNote: DDQN evaluation needed for direct comparison")
    print(f"Run: python scripts/train_ddqn.py --eval-only --resume results/checkpoints/ddqn/best_model_ep30.pt")

def main():
    """Main analysis"""
    print(f"\n{'#'*70}")
    print(f"# DDQN TRAINING RESULTS ANALYSIS")
    print(f"{'#'*70}")
    
    # Analyze final model
    final_model = "results/checkpoints/ddqn/final_model.pt"
    if Path(final_model).exists():
        final_checkpoint = analyze_checkpoint(final_model)
    else:
        print("\nFinal model not found!")
        return
    
    # Analyze best models
    best_models = list(Path("results/checkpoints/ddqn").glob("best_model_*.pt"))
    if best_models:
        print(f"\n{'='*70}")
        print(f"BEST MODELS FOUND: {len(best_models)}")
        print(f"{'='*70}")
        
        best_models_sorted = sorted(best_models, key=lambda x: x.stat().st_mtime)
        for model in best_models_sorted:
            episode = model.stem.split('_')[-1]
            checkpoint = torch.load(model, map_location='cpu', weights_only=False)
            print(f"\n  Episode {episode}:")
            print(f"    Reward:     {checkpoint['best_reward']:.2f}")
            print(f"    Acceptance: {checkpoint['best_acceptance']:.2%}")
    
    # Compare with baselines
    compare_with_baseline()
    
    # Training quality assessment
    print(f"\n{'='*70}")
    print(f"TRAINING QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    rewards = final_checkpoint['episode_rewards']
    losses = final_checkpoint['losses']
    
    # Check for improvement
    first_10_avg = np.mean(rewards[:10]) if len(rewards) >= 10 else rewards[0]
    last_10_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else rewards[-1]
    improvement = last_10_avg - first_10_avg
    
    print(f"\nLearning Progress:")
    print(f"  First 10 episodes avg:  {first_10_avg:.2f}")
    print(f"  Last 10 episodes avg:   {last_10_avg:.2f}")
    print(f"  Improvement:            {improvement:.2f} ({improvement/abs(first_10_avg)*100:.1f}%)")
    
    if improvement > 0:
        print(f"  ✓ Agent is learning! (rewards improving)")
    else:
        print(f"  ⚠ Limited learning detected")
    
    # Check loss convergence
    if len(losses) > 100:
        first_100_loss = np.mean(losses[:100])
        last_100_loss = np.mean(losses[-100:])
        
        print(f"\nLoss Convergence:")
        print(f"  First 100 updates:      {first_100_loss:.4f}")
        print(f"  Last 100 updates:       {last_100_loss:.4f}")
        
        if last_100_loss < first_100_loss:
            print(f"  ✓ Loss is decreasing (good convergence)")
        else:
            print(f"  ⚠ Loss not decreasing significantly")
    
    # Epsilon decay check
    eps_hist = final_checkpoint['epsilon_history']
    print(f"\nExploration Strategy:")
    print(f"  Epsilon decay:          {eps_hist[0]:.3f} → {eps_hist[-1]:.3f}")
    
    if eps_hist[-1] < eps_hist[0] * 0.7:
        print(f"  ✓ Good exploration decay")
    else:
        print(f"  ⚠ Limited exploration decay")
    
    print(f"\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. Run full evaluation:")
    print(f"   python scripts/train_ddqn.py --eval-only --resume results/checkpoints/ddqn/best_model_ep30.pt")
    print(f"\n2. View training curves in TensorBoard:")
    print(f"   tensorboard --logdir results/logs/ddqn")
    print(f"\n3. Train for more episodes if needed:")
    print(f"   python scripts/train_ddqn.py --episodes 1000 --resume results/checkpoints/ddqn/checkpoint_ep50.pt")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()

