import numpy as np
from typing import List, Dict, Any
from collections import deque

class MetricsTracker:
    """
    Tracks and computes various performance metrics for the cloud resource allocation system.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Episode metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        
        # Task metrics
        self.completion_times = deque(maxlen=window_size)
        self.acceptance_rates = deque(maxlen=window_size)
        self.timeout_rates = deque(maxlen=window_size)
        
        # Resource metrics
        self.cpu_utilization = deque(maxlen=window_size)
        self.memory_utilization = deque(maxlen=window_size)
        
        # Training metrics
        self.losses = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        
    def update_episode_metrics(
        self,
        episode_reward: float,
        episode_length: int,
        completed_tasks: List[Any],
        total_tasks: int,
        avg_cpu_util: float,
        avg_memory_util: float
    ):
        """Update metrics after each episode"""
        # Episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Task metrics
        if completed_tasks:
            completion_times = [
                task.completion_time - task.arrival_time 
                for task in completed_tasks
            ]
            self.completion_times.append(np.mean(completion_times))
            
        acceptance_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
        self.acceptance_rates.append(acceptance_rate)
        
        # Resource utilization
        self.cpu_utilization.append(avg_cpu_util)
        self.memory_utilization.append(avg_memory_util)
        
    def update_training_metrics(self, loss: float, q_value: float):
        """Update training-related metrics"""
        self.losses.append(loss)
        self.q_values.append(q_value)
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics averaged over the window"""
        metrics = {
            "avg_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "avg_completion_time": np.mean(self.completion_times) if self.completion_times else 0.0,
            "avg_acceptance_rate": np.mean(self.acceptance_rates) if self.acceptance_rates else 0.0,
            "avg_cpu_utilization": np.mean(self.cpu_utilization) if self.cpu_utilization else 0.0,
            "avg_memory_utilization": np.mean(self.memory_utilization) if self.memory_utilization else 0.0,
            "avg_loss": np.mean(self.losses) if self.losses else 0.0,
            "avg_q_value": np.mean(self.q_values) if self.q_values else 0.0
        }
        
        return metrics
        
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary metrics for the most recent episode"""
        return {
            "episode_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "episode_length": self.episode_lengths[-1] if self.episode_lengths else 0,
            "completion_time": self.completion_times[-1] if self.completion_times else 0.0,
            "acceptance_rate": self.acceptance_rates[-1] if self.acceptance_rates else 0.0,
            "cpu_utilization": self.cpu_utilization[-1] if self.cpu_utilization else 0.0,
            "memory_utilization": self.memory_utilization[-1] if self.memory_utilization else 0.0
        }
        
    def get_training_curves(self) -> Dict[str, List[float]]:
        """Get metrics as lists for plotting learning curves"""
        return {
            "rewards": list(self.episode_rewards),
            "lengths": list(self.episode_lengths),
            "completion_times": list(self.completion_times),
            "acceptance_rates": list(self.acceptance_rates),
            "cpu_utilization": list(self.cpu_utilization),
            "memory_utilization": list(self.memory_utilization),
            "losses": list(self.losses),
            "q_values": list(self.q_values)
        }
        
    def has_improved(self, metric: str, min_improvement: float = 0.01) -> bool:
        """
        Check if a specific metric has improved over its window
        
        Args:
            metric: Name of the metric to check
            min_improvement: Minimum improvement threshold
            
        Returns:
            bool: True if metric has improved by at least min_improvement
        """
        if metric == "episode_reward":
            values = self.episode_rewards
        elif metric == "completion_time":
            values = self.completion_times
        elif metric == "acceptance_rate":
            values = self.acceptance_rates
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if len(values) < self.window_size:
            return True
            
        # Compare first and second half of window
        half_size = self.window_size // 2
        first_half = list(values)[:half_size]
        second_half = list(values)[half_size:]
        
        improvement = np.mean(second_half) - np.mean(first_half)
        
        if metric == "completion_time":  # Lower is better
            improvement = -improvement
            
        return improvement >= min_improvement
