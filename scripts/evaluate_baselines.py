#!/usr/bin/env python3
"""
Baseline Scheduler Evaluation Script

Evaluates all baseline scheduling algorithms on the realistic cloud environment
and generates comparative performance metrics.

Usage:
    python scripts/evaluate_baselines.py                    # Evaluate all algorithms
    python scripts/evaluate_baselines.py --algo random      # Evaluate specific algorithm
    python scripts/evaluate_baselines.py --episodes 50      # Fewer episodes (faster)
    python scripts/evaluate_baselines.py --compare          # Generate comparison table
"""

import sys
import os
import argparse
import time
from pathlib import Path
import json
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.realistic_cloud_env import RealisticCloudEnvironment
from baselines import (
    RandomScheduler,
    RoundRobinScheduler,
    FCFSScheduler,
    SJFScheduler,
    BestFitScheduler,
    LeastLoadedScheduler
)


class BaselineEvaluator:
    """Evaluator for baseline scheduling algorithms"""
    
    def __init__(self, env_config_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            env_config_path: Path to environment configuration file
        """
        self.env_config_path = env_config_path
        self.results = {}
    
    def evaluate_algorithm(self, scheduler, num_episodes: int = 100, verbose: bool = True) -> Dict:
        """
        Evaluate a single scheduling algorithm.
        
        Args:
            scheduler: Scheduler to evaluate
            num_episodes: Number of episodes to run
            verbose: Whether to print progress
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {scheduler.name}")
            print(f"{'='*60}")
        
        # Create environment
        env = RealisticCloudEnvironment(config_path=self.env_config_path)
        
        # Run evaluation
        start_time = time.time()
        metrics = scheduler.evaluate(env, num_episodes=num_episodes, verbose=verbose)
        elapsed_time = time.time() - start_time
        
        # Add evaluation time
        metrics['evaluation_time'] = elapsed_time
        
        if verbose:
            self._print_metrics(scheduler.name, metrics)
        
        return metrics
    
    def evaluate_all(self, num_episodes: int = 100, verbose: bool = True) -> Dict[str, Dict]:
        """
        Evaluate all baseline algorithms.
        
        Args:
            num_episodes: Number of episodes per algorithm
            verbose: Whether to print progress
        
        Returns:
            Dictionary mapping algorithm names to their metrics
        """
        schedulers = [
            RandomScheduler(),
            RoundRobinScheduler(),
            FCFSScheduler(),
            SJFScheduler(),
            BestFitScheduler(),
            LeastLoadedScheduler(),
        ]
        
        print(f"\n{'#'*60}")
        print(f"# BASELINE ALGORITHM EVALUATION")
        print(f"# Episodes per algorithm: {num_episodes}")
        print(f"{'#'*60}")
        
        results = {}
        
        for scheduler in schedulers:
            metrics = self.evaluate_algorithm(scheduler, num_episodes, verbose)
            results[scheduler.name] = metrics
        
        self.results = results
        return results
    
    def _print_metrics(self, name: str, metrics: Dict):
        """Print metrics in a formatted way"""
        print(f"\n{name} Results:")
        print(f"{'-'*60}")
        print(f"  Total Tasks:           {metrics['total_tasks']}")
        print(f"  Allocated Tasks:       {metrics['allocated_tasks']}")
        print(f"  Rejected Tasks:        {metrics['rejected_tasks']}")
        print(f"  Acceptance Rate:       {metrics['acceptance_rate']:.2%}")
        print(f"  Avg Completion Time:   {metrics['avg_completion_time']:.2f} time units")
        print(f"  Avg Wait Time:         {metrics['avg_wait_time']:.2f} time units")
        print(f"  Avg Utilization:       {metrics['avg_utilization']:.2%}")
        print(f"  SLA Violations:        {metrics['sla_violations']}")
        print(f"  SLA Violation Rate:    {metrics['sla_violation_rate']:.2%}")
        print(f"  Avg Episode Reward:    {metrics['avg_reward']:.2f}")
        print(f"  Evaluation Time:       {metrics['evaluation_time']:.2f} seconds")
        print(f"{'-'*60}")
    
    def print_comparison_table(self):
        """Print a comparison table of all algorithms"""
        if not self.results:
            print("No results to compare. Run evaluation first.")
            return
        
        print(f"\n{'='*100}")
        print(f"PERFORMANCE COMPARISON TABLE")
        print(f"{'='*100}")
        
        # Header
        print(f"{'Algorithm':<15} | {'Accept%':<8} | {'AvgComp':<8} | {'AvgWait':<8} | "
              f"{'Util%':<8} | {'SLA%':<8} | {'Reward':<8}")
        print(f"{'-'*100}")
        
        # Sort by average reward (descending)
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['avg_reward'], 
                               reverse=True)
        
        # Print each algorithm
        for name, metrics in sorted_results:
            print(f"{name:<15} | "
                  f"{metrics['acceptance_rate']*100:>7.1f}% | "
                  f"{metrics['avg_completion_time']:>8.2f} | "
                  f"{metrics['avg_wait_time']:>8.2f} | "
                  f"{metrics['avg_utilization']*100:>7.1f}% | "
                  f"{metrics['sla_violation_rate']*100:>7.1f}% | "
                  f"{metrics['avg_reward']:>8.2f}")
        
        print(f"{'='*100}")
        
        # Find best performers
        best_acceptance = max(self.results.items(), key=lambda x: x[1]['acceptance_rate'])
        best_completion = min(self.results.items(), key=lambda x: x[1]['avg_completion_time'])
        best_utilization = max(self.results.items(), key=lambda x: x[1]['avg_utilization'])
        best_reward = max(self.results.items(), key=lambda x: x[1]['avg_reward'])
        
        print(f"\nBest Performers:")
        print(f"  Highest Acceptance Rate:    {best_acceptance[0]} ({best_acceptance[1]['acceptance_rate']:.2%})")
        print(f"  Lowest Completion Time:     {best_completion[0]} ({best_completion[1]['avg_completion_time']:.2f})")
        print(f"  Highest Utilization:        {best_utilization[0]} ({best_utilization[1]['avg_utilization']:.2%})")
        print(f"  Highest Reward:             {best_reward[0]} ({best_reward[1]['avg_reward']:.2f})")
        print(f"{'='*100}\n")
    
    def save_results(self, output_path: str = "results/baseline_evaluation.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        # Create results directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def generate_summary_stats(self):
        """Generate summary statistics across all algorithms"""
        if not self.results:
            print("No results to summarize. Run evaluation first.")
            return
        
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        # Collect all values for each metric
        metrics_to_analyze = [
            'acceptance_rate',
            'avg_completion_time',
            'avg_utilization',
            'sla_violation_rate',
            'avg_reward'
        ]
        
        for metric in metrics_to_analyze:
            values = [results[metric] for results in self.results.values()]
            
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:   {np.mean(values):.4f}")
            print(f"  Std:    {np.std(values):.4f}")
            print(f"  Min:    {np.min(values):.4f}")
            print(f"  Max:    {np.max(values):.4f}")
            print(f"  Range:  {np.max(values) - np.min(values):.4f}")
        
        print(f"{'='*60}\n")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline scheduling algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--algo', '--algorithm',
        type=str,
        choices=['random', 'round-robin', 'fcfs', 'sjf', 'best-fit', 'least-loaded', 'all'],
        default='all',
        help='Algorithm to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to run (default: 100)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to environment configuration file'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparison table'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default='results/baseline_evaluation.json',
        help='Path to save results (default: results/baseline_evaluation.json)'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BaselineEvaluator(env_config_path=args.config)
    
    verbose = not args.no_verbose
    
    # Evaluate specific algorithm or all
    if args.algo == 'all':
        results = evaluator.evaluate_all(num_episodes=args.episodes, verbose=verbose)
    else:
        # Map algorithm names to classes
        algo_map = {
            'random': RandomScheduler,
            'round-robin': RoundRobinScheduler,
            'fcfs': FCFSScheduler,
            'sjf': SJFScheduler,
            'best-fit': BestFitScheduler,
            'least-loaded': LeastLoadedScheduler,
        }
        
        scheduler_class = algo_map[args.algo]
        scheduler = scheduler_class()
        metrics = evaluator.evaluate_algorithm(scheduler, num_episodes=args.episodes, verbose=verbose)
        results = {scheduler.name: metrics}
        evaluator.results = results
    
    # Generate comparison table if requested or if evaluating all
    if args.compare or args.algo == 'all':
        evaluator.print_comparison_table()
        evaluator.generate_summary_stats()
    
    # Save results
    evaluator.save_results(args.save)
    
    print("\n✓ Evaluation complete!\n")


if __name__ == "__main__":
    main()

