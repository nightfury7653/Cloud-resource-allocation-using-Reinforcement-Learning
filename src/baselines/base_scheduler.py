"""
Base Scheduler Abstract Class

Defines the interface that all baseline scheduling algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SchedulerMetrics:
    """Metrics collected during scheduler evaluation"""
    total_tasks: int = 0
    allocated_tasks: int = 0
    rejected_tasks: int = 0
    total_completion_time: float = 0.0
    total_wait_time: float = 0.0
    total_utilization: float = 0.0
    sla_violations: int = 0
    episode_rewards: List[float] = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate task acceptance rate"""
        if self.total_tasks == 0:
            return 0.0
        return self.allocated_tasks / self.total_tasks
    
    @property
    def average_completion_time(self) -> float:
        """Calculate average task completion time"""
        if self.allocated_tasks == 0:
            return 0.0
        return self.total_completion_time / self.allocated_tasks
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average task wait time"""
        if self.allocated_tasks == 0:
            return 0.0
        return self.total_wait_time / self.allocated_tasks
    
    @property
    def average_utilization(self) -> float:
        """Calculate average resource utilization"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_utilization / self.total_tasks
    
    @property
    def sla_violation_rate(self) -> float:
        """Calculate SLA violation rate"""
        if self.allocated_tasks == 0:
            return 0.0
        return self.sla_violations / self.allocated_tasks
    
    @property
    def average_reward(self) -> float:
        """Calculate average episode reward"""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_tasks': self.total_tasks,
            'allocated_tasks': self.allocated_tasks,
            'rejected_tasks': self.rejected_tasks,
            'acceptance_rate': self.acceptance_rate,
            'avg_completion_time': self.average_completion_time,
            'avg_wait_time': self.average_wait_time,
            'avg_utilization': self.average_utilization,
            'sla_violations': self.sla_violations,
            'sla_violation_rate': self.sla_violation_rate,
            'avg_reward': self.average_reward,
        }


class BaseScheduler(ABC):
    """
    Abstract base class for all scheduling algorithms.
    
    All schedulers must implement the select_vm method which determines
    which VM should be allocated for a given task.
    """
    
    def __init__(self, name: str = "BaseScheduler"):
        """
        Initialize the scheduler.
        
        Args:
            name: Name of the scheduler algorithm
        """
        self.name = name
        self.metrics = SchedulerMetrics()
        self.reset()
    
    def reset(self):
        """
        Reset scheduler state.
        
        Called at the beginning of each evaluation episode.
        Can be overridden by subclasses that maintain internal state.
        """
        pass
    
    @abstractmethod
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select a VM for task allocation.
        
        Args:
            task: Task object to be allocated
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of selected VM (0 to len(vms)-1)
            Returns -1 if no suitable VM found
        """
        raise NotImplementedError("Subclasses must implement select_vm method")
    
    def can_allocate(self, task: Any, vm: Any) -> bool:
        """
        Check if a task can be allocated to a VM.
        
        Args:
            task: Task to allocate
            vm: VM to check
        
        Returns:
            True if task can be allocated, False otherwise
        """
        # Check CPU capacity
        if vm.cpu_used + task.cpu_requirement > vm.cpu_total:
            return False
        
        # Check memory capacity
        if vm.memory_used + task.memory_requirement > vm.memory_total:
            return False
        
        return True
    
    def evaluate(self, env: Any, num_episodes: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate scheduler on an environment.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to run
            verbose: Whether to print progress
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Reset metrics
        self.metrics = SchedulerMetrics()
        
        for episode in range(num_episodes):
            # Reset environment and scheduler
            state, info = env.reset()
            self.reset()
            
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done:
                # Get current task and VMs from environment
                if len(env.task_queue) == 0:
                    # No tasks, step with action 0 (will generate new tasks)
                    state, reward, done, truncated, info = env.step(0)
                    episode_reward += reward
                    continue
                
                current_task = env.task_queue[0]
                
                # Select VM using scheduling algorithm
                vm_index = self.select_vm(current_task, env.vms, env.current_time)
                
                # Take action in environment
                state, reward, done, truncated, info = env.step(vm_index)
                episode_reward += reward
                
                # Update metrics
                self.metrics.total_tasks += 1
                
                if info['allocation_info']['allocated']:
                    self.metrics.allocated_tasks += 1
                    
                    # Get the allocated task from allocation_info
                    allocated_task = info['allocation_info']['task']
                    
                    # Calculate wait time (time from arrival to start)
                    if hasattr(allocated_task, 'start_time') and allocated_task.start_time is not None:
                        wait_time = allocated_task.start_time - allocated_task.arrival_time
                        self.metrics.total_wait_time += wait_time
                    
                    # Estimate completion time based on actual_execution_time
                    if hasattr(allocated_task, 'actual_execution_time'):
                        self.metrics.total_completion_time += allocated_task.actual_execution_time
                    
                    # Check SLA violation (if task has completed)
                    if hasattr(allocated_task, 'deadline') and hasattr(allocated_task, 'completion_time'):
                        if allocated_task.completion_time and allocated_task.completion_time > allocated_task.deadline:
                            self.metrics.sla_violations += 1
                else:
                    self.metrics.rejected_tasks += 1
                
                # Calculate utilization
                total_util = sum(vm.cpu_used / vm.cpu_total for vm in env.vms) / len(env.vms)
                self.metrics.total_utilization += total_util
                
                step += 1
                
                if done or truncated:
                    break
            
            # Store episode reward
            self.metrics.episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Acceptance: {self.metrics.acceptance_rate:.2%}")
        
        # Return final metrics
        return self.metrics.to_dict()
    
    def __str__(self) -> str:
        """String representation of scheduler"""
        return f"{self.name} Scheduler"
    
    def __repr__(self) -> str:
        """Detailed representation of scheduler"""
        return f"{self.__class__.__name__}(name='{self.name}')"

