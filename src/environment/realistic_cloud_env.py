"""
Realistic Cloud Environment with Enhanced Models
Integrates all performance, workload, and resource models
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from typing import Dict, Tuple, Any, List, Optional
from pathlib import Path

from .task_models import Task, VirtualMachine, TaskType
from .workload_generator import WorkloadGenerator, WorkloadPattern, RealTraceWorkloadGenerator
from .performance_models import PerformanceModel, NetworkModel, ResourceContentionModel

class RealisticCloudEnvironment(gym.Env):
    """
    Realistic Cloud Resource Allocation Environment
    
    Enhancements over basic environment:
    - Realistic workload patterns
    - Performance degradation modeling
    - Resource contention effects
    - Task interference
    - Dynamic execution times
    - Network simulation
    """
    
    def __init__(
        self,
        config_path: str = "config/env_config.yaml",
        workload_pattern: WorkloadPattern = WorkloadPattern.PERIODIC,
        use_real_trace: bool = False,
        trace_file: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.seed_value = seed
        
        # Initialize environment parameters
        self.num_vms = self.config["vm_pool"]["num_vms"]
        self.max_queue_size = self.config["environment"]["max_queue_size"]
        self.episode_length = self.config["environment"]["episode_length"]
        
        # Initialize VMs with enhanced model
        self.vms = self._initialize_vms()
        
        # Initialize workload generator
        if use_real_trace and trace_file:
            self.workload_generator = RealTraceWorkloadGenerator(
                trace_file=trace_file,
                base_arrival_rate=self.config["task"]["arrival_rate"],
                seed=seed
            )
        else:
            self.workload_generator = WorkloadGenerator(
                pattern=workload_pattern,
                base_arrival_rate=self.config["task"]["arrival_rate"],
                seed=seed
            )
        
        # Initialize performance models
        self.performance_model = PerformanceModel()
        self.network_model = NetworkModel()
        self.contention_model = ResourceContentionModel()
        
        # Task management
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.running_tasks = []
        self.current_time = 0
        self.time_step = self.config["environment"]["time_step"]
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_vms)
        
        # Enhanced state space
        state_dim = self._calculate_state_dim()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Statistics tracking
        self.episode_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_wait_time': 0.0,
            'total_execution_time': 0.0,
            'resource_utilization': []
        }
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset time and state
        self.current_time = 0
        
        # Clear all queues
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.running_tasks = []
        
        # Reset VMs
        self.vms = self._initialize_vms()
        
        # Reset workload generator
        self.workload_generator.current_time = 0
        
        # Reset statistics
        self.episode_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_wait_time': 0.0,
            'total_execution_time': 0.0,
            'resource_utilization': []
        }
        
        # Generate initial tasks
        self._generate_new_tasks()
        
        return self._get_state(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Index of VM to assign the current task
            
        Returns:
            state: New environment state
            reward: Reward for the action
            terminated: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Process current task allocation
        allocation_info = self._process_allocation(action)
        
        # Update environment (advance time, update tasks, etc.)
        self._update_environment()
        
        # Generate new tasks
        self._generate_new_tasks()
        
        # Calculate reward
        reward = self._calculate_reward(allocation_info)
        
        # Check if episode is done
        terminated = self.current_time >= self.episode_length
        truncated = False
        
        # Get new state and info
        new_state = self._get_state()
        info = self._get_info(allocation_info)
        
        return new_state, reward, terminated, truncated, info
    
    def _process_allocation(self, action: int) -> Dict:
        """
        Process task allocation decision
        
        Args:
            action: VM index to allocate to
            
        Returns:
            Dictionary with allocation information
        """
        info = {
            'allocated': False,
            'task': None,
            'vm': None,
            'reason': None
        }
        
        if len(self.task_queue) == 0:
            info['reason'] = 'empty_queue'
            return info
        
        # Get current task and selected VM
        current_task = self.task_queue[0]
        selected_vm = self.vms[action]
        
        # Try to allocate task to VM
        if selected_vm.allocate_task(current_task):
            # Remove from queue
            self.task_queue.pop(0)
            
            # Start task
            current_task.start_time = self.current_time
            
            # Calculate actual execution time using performance model
            current_task.actual_execution_time = self.performance_model.calculate_execution_time(
                current_task,
                selected_vm
            )
            
            # Add to running tasks
            self.running_tasks.append(current_task)
            
            info['allocated'] = True
            info['task'] = current_task
            info['vm'] = selected_vm
            info['reason'] = 'success'
            
            self.episode_stats['total_tasks'] += 1
            
        else:
            info['reason'] = 'insufficient_resources'
        
        return info
    
    def _update_environment(self):
        """Update environment state - advance time, complete tasks, etc."""
        # Advance time
        self.current_time += self.time_step
        
        # Update all VMs
        for vm in self.vms:
            vm.update_state(self.time_step)
        
        # Update running tasks
        completed_this_step = []
        
        for task in self.running_tasks:
            # Get VM executing this task
            vm = self.vms[task.assigned_vm]
            
            # Update task progress
            new_progress = self.performance_model.update_task_progress(
                task,
                vm,
                self.time_step
            )
            task.progress = new_progress
            
            # Update dynamic resource usage
            task.update_resource_usage(
                self.current_time - task.start_time,
                vm.cpu_utilization
            )
            
            # Check if task is complete
            if task.progress >= 1.0:
                task.completed = True
                task.completion_time = self.current_time
                completed_this_step.append(task)
                
                # Check if deadline was met
                if not task.is_deadline_met:
                    self.failed_tasks.append(task)
                    self.episode_stats['failed_tasks'] += 1
                else:
                    self.completed_tasks.append(task)
                    self.episode_stats['completed_tasks'] += 1
                
                # Update statistics
                self.episode_stats['total_wait_time'] += task.wait_time
                self.episode_stats['total_execution_time'] += task.execution_duration
                
                # Release resources
                vm.release_task(task)
        
        # Remove completed tasks from running list
        for task in completed_this_step:
            self.running_tasks.remove(task)
        
        # Track resource utilization
        avg_util = np.mean([vm.cpu_utilization for vm in self.vms])
        self.episode_stats['resource_utilization'].append(avg_util)
    
    def _generate_new_tasks(self):
        """Generate new tasks using workload generator"""
        if len(self.task_queue) >= self.max_queue_size:
            return
        
        # Generate tasks
        new_tasks = self.workload_generator.generate_tasks(
            self.current_time,
            self.time_step
        )
        
        # Add to queue
        for task in new_tasks:
            if len(self.task_queue) < self.max_queue_size:
                self.task_queue.append(task)
    
    def _calculate_state_dim(self) -> int:
        """Calculate state space dimension"""
        return (
            1 +                      # queue length
            self.num_vms * 4 +       # per-VM: CPU util, memory util, num tasks, load factor
            6                        # current task features (if any)
        )
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Queue length (normalized)
        state.append(len(self.task_queue) / self.max_queue_size)
        
        # VM states
        for vm in self.vms:
            state.append(vm.cpu_utilization)
            state.append(vm.memory_utilization)
            state.append(len(vm.tasks) / 10.0)  # Normalize by max expected tasks
            state.append(1.0 / vm.get_performance_factor())  # Load factor
        
        # Current task features (if any)
        if len(self.task_queue) > 0:
            task = self.task_queue[0]
            max_cpu = max(vm.cpu_total for vm in self.vms)
            max_mem = max(vm.memory_total for vm in self.vms)
            
            state.extend([
                task.cpu_requirement / max_cpu,
                task.memory_requirement / max_mem,
                task.priority / 5.0,
                min(1.0, (task.deadline - self.current_time) / 1000.0),
                task.profile.cpu_intensity,
                task.profile.memory_intensity
            ])
        else:
            state.extend([0, 0, 0, 0, 0, 0])
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, allocation_info: Dict) -> float:
        """
        Calculate reward based on allocation decision
        
        Args:
            allocation_info: Information about the allocation
            
        Returns:
            Reward value
        """
        if not allocation_info['allocated']:
            return -1.0  # Penalty for failed allocation
        
        task = allocation_info['task']
        vm = allocation_info['vm']
        
        # Get reward weights from config
        weights = self.config.get("reward", {})
        alpha = weights.get("completion_time_weight", 1.0)
        beta = weights.get("utilization_weight", 0.5)
        gamma = weights.get("training_cost_weight", 0.3)
        delta = weights.get("acceptance_rate_weight", 0.7)
        
        # Component rewards
        
        # 1. Execution time reward (negative, prefer shorter times)
        estimated_time = task.actual_execution_time
        time_reward = -alpha * (estimated_time / 600.0)  # Normalize by max time
        
        # 2. Resource utilization reward
        target_util = 0.7  # Target 70% utilization
        util_deviation = abs(vm.cpu_utilization - target_util)
        util_reward = beta * (1.0 - util_deviation)
        
        # 3. Load balancing reward
        vm_utils = [v.cpu_utilization for v in self.vms]
        util_std = np.std(vm_utils)
        balance_reward = 0.3 * (1.0 - min(1.0, util_std))
        
        # 4. Task acceptance reward
        acceptance_reward = delta
        
        # 5. Performance factor reward (prefer VMs with good performance)
        perf_factor = vm.get_performance_factor()
        perf_reward = 0.2 * perf_factor
        
        # Total reward
        total_reward = (
            time_reward +
            util_reward +
            balance_reward +
            acceptance_reward +
            perf_reward
        )
        
        return total_reward
    
    def _get_info(self, allocation_info: Dict) -> Dict[str, Any]:
        """Get additional information about the environment state"""
        # Build task_metrics dict for better tracking
        task_metrics = {
            'completion_time': None,
            'wait_time': None,
            'sla_violated': False,
        }
        
        # If a task was allocated, add its metrics
        if allocation_info['allocated'] and allocation_info['task']:
            task = allocation_info['task']
            if hasattr(task, 'start_time') and task.start_time is not None:
                task_metrics['wait_time'] = task.start_time - task.arrival_time
            if hasattr(task, 'actual_execution_time'):
                task_metrics['completion_time'] = task.actual_execution_time
            if hasattr(task, 'deadline'):
                # Estimated completion time
                estimated_completion = self.current_time + task.actual_execution_time
                task_metrics['sla_violated'] = estimated_completion > task.deadline
        
        info = {
            'current_time': self.current_time,
            'queue_length': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'allocation_info': allocation_info,
            'task_metrics': task_metrics,
            'metrics': {
                'new_tasks_generated': getattr(self, '_tasks_generated_this_step', 0)
            }
        }
        
        # Average metrics
        if self.episode_stats['completed_tasks'] > 0:
            info['avg_wait_time'] = (
                self.episode_stats['total_wait_time'] / 
                self.episode_stats['completed_tasks']
            )
            info['avg_execution_time'] = (
                self.episode_stats['total_execution_time'] / 
                self.episode_stats['completed_tasks']
            )
        else:
            info['avg_wait_time'] = 0.0
            info['avg_execution_time'] = 0.0
        
        # Resource utilization
        if self.episode_stats['resource_utilization']:
            info['avg_utilization'] = np.mean(
                self.episode_stats['resource_utilization']
            )
        else:
            info['avg_utilization'] = 0.0
        
        # VM states
        info['vm_utilizations'] = [vm.cpu_utilization for vm in self.vms]
        info['vm_loads'] = [len(vm.tasks) for vm in self.vms]
        
        return info
    
    def _load_config(self, config_path: str) -> dict:
        """Load environment configuration from YAML file"""
        if config_path is None:
            # Use default configuration
            return self._get_default_config()
        
        config_file = Path(config_path)
        if not config_file.exists():
            # Use default configuration
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "vm_pool": {
                "num_vms": 10,
                "vm_types": [
                    {"type": "small", "cpu_cores": 2, "memory_gb": 4},
                    {"type": "medium", "cpu_cores": 4, "memory_gb": 8},
                    {"type": "large", "cpu_cores": 8, "memory_gb": 16}
                ]
            },
            "task": {
                "arrival_rate": 5
            },
            "environment": {
                "max_queue_size": 100,
                "episode_length": 500,
                "time_step": 1.0
            },
            "reward": {
                "completion_time_weight": 1.0,
                "utilization_weight": 0.5,
                "training_cost_weight": 0.3,
                "acceptance_rate_weight": 0.7
            }
        }
    
    def _initialize_vms(self) -> List[VirtualMachine]:
        """Initialize virtual machines based on configuration"""
        vms = []
        vm_types = self.config["vm_pool"]["vm_types"]
        
        for i in range(self.num_vms):
            # Cycle through VM types
            vm_type_config = vm_types[i % len(vm_types)]
            
            vm = VirtualMachine(
                id=i,
                vm_type=vm_type_config["type"],
                cpu_total=vm_type_config["cpu_cores"],
                memory_total=vm_type_config["memory_gb"]
            )
            vms.append(vm)
        
        return vms
