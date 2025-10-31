#!/usr/bin/env python3
"""
A3C Training Script

Train an A3C agent on the realistic cloud resource allocation environment.

Usage:
    python scripts/train_a3c.py                          # Train with default settings
    python scripts/train_a3c.py --episodes 500           # Train for 500 episodes
    python scripts/train_a3c.py --resume checkpoint.pt   # Resume from checkpoint
    python scripts/train_a3c.py --eval-only              # Evaluation only
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.realistic_cloud_env import RealisticCloudEnvironment
from agent.a3c_agent import A3CAgent

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False


class A3CTrainer:
    """Trainer for A3C Agent"""
    
    def __init__(self, 
                 env: RealisticCloudEnvironment,
                 agent: A3CAgent,
                 config: dict,
                 log_dir: str = None,
                 checkpoint_dir: str = None):
        """
        Initialize trainer.
        
        Args:
            env: Training environment
            agent: A3C agent
            config: Training configuration
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.env = env
        self.agent = agent
        self.config = config
        
        # Setup logging
        self.log_dir = log_dir or f"results/logs/a3c/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = None
        if TENSORBOARD_AVAILABLE and config['logging']['use_tensorboard']:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logging to: {self.log_dir}")
        
        # Setup checkpointing
        self.checkpoint_dir = checkpoint_dir or config['checkpointing']['save_dir']
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_reward = float('-inf')
        self.best_acceptance = 0.0
        self.episode = 0
        self.global_step = 0
    
    def train(self, num_episodes: int, eval_frequency: int = 10, eval_episodes: int = 10):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of training episodes
            eval_frequency: Evaluate every N episodes
            eval_episodes: Number of episodes for evaluation
        """
        print(f"\n{'='*70}")
        print(f"STARTING DDQN TRAINING")
        print(f"{'='*70}")
        print(f"Episodes: {num_episodes}")
        print(f"Environment: RealisticCloudEnvironment")
        print(f"Agent: {self.agent}")
        print(f"Device: {self.agent.device}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            self.episode = episode
            
            # Training episode
            episode_stats = self._train_episode()
            
            # Log training stats
            self._log_training(episode_stats)
            
            # Evaluation
            if episode % eval_frequency == 0:
                eval_stats = self.evaluate(eval_episodes)
                self._log_evaluation(eval_stats)
                
                # Save best model
                if eval_stats['avg_reward'] > self.best_reward:
                    self.best_reward = eval_stats['avg_reward']
                    self.save_checkpoint(f"best_model_ep{episode}.pt", is_best=True)
                
                if eval_stats['acceptance_rate'] > self.best_acceptance:
                    self.best_acceptance = eval_stats['acceptance_rate']
            
            # Save periodic checkpoint
            if episode % self.config['checkpointing']['save_frequency'] == 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
            
            # Print progress
            if episode % self.config['logging']['log_frequency'] == 0:
                self._print_progress(episode, num_episodes, episode_stats, start_time)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Episodes: {num_episodes}")
        print(f"Steps: {self.agent.steps}")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Best acceptance: {self.best_acceptance:.2%}")
        print(f"{'='*70}\n")
        
        # Final evaluation
        print("Running final evaluation...")
        final_stats = self.evaluate(100)
        self._print_eval_results(final_stats)
        
        # Save final model
        self.save_checkpoint("final_model.pt", is_best=False)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def _train_episode(self) -> dict:
        """Train for one episode"""
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_allocations = 0
        episode_rejections = 0
        
        done = False
        while not done:
            # Select action
            action = self.agent.select_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Update agent
            loss = self.agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            
            if info['allocation_info']['allocated']:
                episode_allocations += 1
            else:
                episode_rejections += 1
            
            state = next_state
            
            if done or truncated:
                break
        
        # Episode end
        self.agent.episode_end(episode_reward, episode_length)
        
        # Calculate utilization
        avg_utilization = np.mean([vm.cpu_utilization for vm in self.env.vms])
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'loss': np.mean(episode_losses) if episode_losses else 0.0,
            'epsilon': self.agent.epsilon,
            'allocations': episode_allocations,
            'rejections': episode_rejections,
            'acceptance_rate': episode_allocations / (episode_allocations + episode_rejections) if (episode_allocations + episode_rejections) > 0 else 0.0,
            'utilization': avg_utilization,
            'learning_rate': self.agent.learning_rate,
        }
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        Evaluate the agent.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary of evaluation statistics
        """
        total_rewards = []
        total_lengths = []
        total_allocations = []
        total_rejections = []
        total_utilizations = []
        
        # Save current epsilon
        train_epsilon = self.agent.epsilon
        self.agent.epsilon = self.config['exploration']['epsilon_eval']
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_allocations = 0
            episode_rejections = 0
            
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if info['allocation_info']['allocated']:
                    episode_allocations += 1
                else:
                    episode_rejections += 1
                
                state = next_state
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            total_allocations.append(episode_allocations)
            total_rejections.append(episode_rejections)
            
            avg_util = np.mean([vm.cpu_utilization for vm in self.env.vms])
            total_utilizations.append(avg_util)
        
        # Restore training epsilon
        self.agent.epsilon = train_epsilon
        
        # Calculate statistics
        total_tasks = np.sum(total_allocations) + np.sum(total_rejections)
        acceptance_rate = np.sum(total_allocations) / total_tasks if total_tasks > 0 else 0.0
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_length': np.mean(total_lengths),
            'acceptance_rate': acceptance_rate,
            'avg_utilization': np.mean(total_utilizations),
            'num_episodes': num_episodes,
        }
    
    def _log_training(self, stats: dict):
        """Log training statistics to TensorBoard"""
        if self.writer is None:
            return
        
        self.writer.add_scalar('Train/Reward', stats['reward'], self.episode)
        self.writer.add_scalar('Train/Length', stats['length'], self.episode)
        self.writer.add_scalar('Train/Loss', stats['loss'], self.episode)
        self.writer.add_scalar('Train/Epsilon', stats['epsilon'], self.episode)
        self.writer.add_scalar('Train/AcceptanceRate', stats['acceptance_rate'], self.episode)
        self.writer.add_scalar('Train/Utilization', stats['utilization'], self.episode)
        self.writer.add_scalar('Train/LearningRate', stats['learning_rate'], self.episode)
        self.writer.add_scalar('Train/BufferSize', len(self.agent.replay_buffer), self.episode)
    
    def _log_evaluation(self, stats: dict):
        """Log evaluation statistics to TensorBoard"""
        if self.writer is None:
            return
        
        self.writer.add_scalar('Eval/Reward', stats['avg_reward'], self.episode)
        self.writer.add_scalar('Eval/AcceptanceRate', stats['acceptance_rate'], self.episode)
        self.writer.add_scalar('Eval/Utilization', stats['avg_utilization'], self.episode)
    
    def _print_progress(self, episode: int, total_episodes: int, stats: dict, start_time: float):
        """Print training progress"""
        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        eta = (total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
        
        print(f"Episode {episode}/{total_episodes} | "
              f"Reward: {stats['reward']:>7.2f} | "
              f"Accept: {stats['acceptance_rate']:>5.1%} | "
              f"Util: {stats['utilization']:>5.1%} | "
              f"ε: {stats['epsilon']:.3f} | "
              f"Loss: {stats['loss']:.4f} | "
              f"Buffer: {len(self.agent.replay_buffer):>6} | "
              f"ETA: {eta/60:.1f}m")
    
    def _print_eval_results(self, stats: dict):
        """Print evaluation results"""
        print(f"\n{'-'*70}")
        print(f"EVALUATION RESULTS ({stats['num_episodes']} episodes)")
        print(f"{'-'*70}")
        print(f"  Average Reward:     {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Acceptance Rate:    {stats['acceptance_rate']:.2%}")
        print(f"  Avg Utilization:    {stats['avg_utilization']:.2%}")
        print(f"  Avg Episode Length: {stats['avg_length']:.1f}")
        print(f"{'-'*70}\n")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        filepath = Path(self.checkpoint_dir) / filename
        
        self.agent.save_checkpoint(
            str(filepath),
            episode=self.episode,
            global_step=self.global_step,
            best_reward=self.best_reward,
            best_acceptance=self.best_acceptance
        )
        
        if is_best:
            print(f"  ✨ New best model! Reward: {self.best_reward:.2f}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train A3C agent")
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--eval-frequency', type=int, default=10,
                       help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to DDQN config file')
    parser.add_argument('--env-config', type=str, default=None,
                       help='Path to environment config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                       help='Evaluation only (no training)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"A3C TRAINING SETUP")
    print(f"{'='*70}")
    
    # Create environment
    print("Creating environment...")
    env = RealisticCloudEnvironment(config_path=args.env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create agent
    print("Creating A3C agent...")
    agent = A3CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config_path=args.config,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)
    
    # Load config
    import yaml
    config_path = args.config or Path(__file__).parent.parent / 'config' / 'a3c_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = A3CTrainer(env, agent, config)
    
    # Run training or evaluation
    if args.eval_only:
        print("\nRunning evaluation only...")
        eval_stats = trainer.evaluate(100)
        trainer._print_eval_results(eval_stats)
    else:
        trainer.train(
            num_episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes
        )


if __name__ == "__main__":
    main()

