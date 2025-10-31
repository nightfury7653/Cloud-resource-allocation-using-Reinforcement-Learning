import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from typing import Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Task:
    """Represents a task in the cloud environment"""
    id: int
    cpu_requirement: float
    memory_requirement: float
    arrival_time: float
    deadline: float
    priority: int
    execution_time: float
    completed: bool = False
    assigned_vm: int = -1
    completion_time: float = 0.0

@dataclass
class VirtualMachine:
    """Represents a VM in the cloud environment"""
    id: int
    cpu_total: float
    memory_total: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    tasks: list = None

    def __post_init__(self):
        self.tasks = []

    @property
    def cpu_utilization(self) -> float:
        return self.cpu_used / self.cpu_total

    @property
    def memory_utilization(self) -> float:
        return self.memory_used / self.memory_total

class CloudEnvironment(gym.Env):
    """
    Cloud Resource Allocation Environment
    
    This environment simulates a cloud computing system where tasks need to be
    allocated to virtual machines (VMs) efficiently.
    
    State Space:
        - Task queue length
        - CPU utilization per VM
        - Memory availability per VM
        - Current task requirements (CPU, memory)
        - Task priority and deadline
    
    Action Space:
        - VM selection for task assignment (discrete)
        
    Reward:
        Multi-objective reward considering:
        - Task completion time
        - Resource utilization
        - Task acceptance rate
    """
    
    def __init__(self, config_path: str = "config/env_config.yaml"):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize environment parameters
        self.num_vms = self.config["vm_pool"]["num_vms"]
        self.max_queue_size = self.config["environment"]["max_queue_size"]
        self.episode_length = self.config["environment"]["episode_length"]
        
        # Initialize VMs
        self.vms = self._initialize_vms()
        
        # Task management
        self.task_queue = []
        self.completed_tasks = []
        self.current_task_id = 0
        self.current_time = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_vms)
        
        # State space: [queue_length, *cpu_utils, *memory_avail, task_features]
        state_dim = (1 +                     # queue length
                    self.num_vms * 2 +       # CPU and memory utilization per VM
                    4)                       # Current task features (CPU, memory, priority, deadline)
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(state_dim,),
            dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Reset time and metrics
        self.current_time = 0
        self.current_task_id = 0
        
        # Clear queues and reset VMs
        self.task_queue = []
        self.completed_tasks = []
        self.vms = self._initialize_vms()
        
        # Generate initial tasks
        self._generate_new_tasks()
        
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Index of VM to assign the current task
            
        Returns:
            state: New environment state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        if len(self.task_queue) == 0:
            return self._get_state(), 0, True, {"error": "No tasks in queue"}
        
        # Get current task and selected VM
        current_task = self.task_queue[0]
        selected_vm = self.vms[action]
        
        # Try to allocate task to VM
        allocation_successful = self._allocate_task(current_task, selected_vm)
        
        # Remove task from queue if allocated
        if allocation_successful:
            self.task_queue.pop(0)
        
        # Update environment
        self.current_time += 1
        self._update_environment()
        self._generate_new_tasks()
        
        # Calculate reward
        reward = self._calculate_reward(allocation_successful, current_task, selected_vm)
        
        # Check if episode is done
        done = self.current_time >= self.episode_length
        
        # Get new state and info
        new_state = self._get_state()
        info = {
            "task_allocated": allocation_successful,
            "num_completed_tasks": len(self.completed_tasks),
            "average_completion_time": self._get_average_completion_time(),
            "average_utilization": self._get_average_utilization()
        }
        
        return new_state, reward, done, info

    def _load_config(self, config_path: str) -> dict:
        """Load environment configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_vms(self) -> list:
        """Initialize virtual machines based on configuration"""
        vms = []
        vm_types = self.config["vm_pool"]["vm_types"]
        
        for i in range(self.num_vms):
            # Cycle through VM types
            vm_type = vm_types[i % len(vm_types)]
            vm = VirtualMachine(
                id=i,
                cpu_total=vm_type["cpu_cores"],
                memory_total=vm_type["memory_gb"]
            )
            vms.append(vm)
        
        return vms

    def _generate_new_tasks(self):
        """Generate new tasks based on arrival rate"""
        if len(self.task_queue) >= self.max_queue_size:
            return
            
        arrival_rate = self.config["task"]["arrival_rate"]
        num_new_tasks = np.random.poisson(arrival_rate)
        
        for _ in range(num_new_tasks):
            if len(self.task_queue) >= self.max_queue_size:
                break
                
            task = Task(
                id=self.current_task_id,
                cpu_requirement=np.random.uniform(*self.config["task"]["cpu_requirement"]),
                memory_requirement=np.random.uniform(*self.config["task"]["memory_requirement"]),
                arrival_time=self.current_time,
                deadline=self.current_time + np.random.uniform(*self.config["task"]["deadline_range"]),
                priority=np.random.randint(*self.config["task"]["priority_levels"]),
                execution_time=np.random.uniform(*self.config["task"]["execution_time"])
            )
            
            self.task_queue.append(task)
            self.current_task_id += 1

    def _allocate_task(self, task: Task, vm: VirtualMachine) -> bool:
        """Try to allocate a task to a VM"""
        # Check if VM has enough resources
        if (vm.cpu_used + task.cpu_requirement > vm.cpu_total or
            vm.memory_used + task.memory_requirement > vm.memory_total):
            return False
        
        # Allocate resources
        vm.cpu_used += task.cpu_requirement
        vm.memory_used += task.memory_requirement
        
        # Update task
        task.assigned_vm = vm.id
        vm.tasks.append(task)
        
        return True

    def _update_environment(self):
        """Update environment state (complete tasks, free resources)"""
        for vm in self.vms:
            completed = []
            for task in vm.tasks:
                if not task.completed:
                    # Check if task is completed
                    time_in_system = self.current_time - task.arrival_time
                    if time_in_system >= task.execution_time:
                        task.completed = True
                        task.completion_time = self.current_time
                        completed.append(task)
                        
                        # Free resources
                        vm.cpu_used -= task.cpu_requirement
                        vm.memory_used -= task.memory_requirement
            
            # Remove completed tasks from VM
            for task in completed:
                vm.tasks.remove(task)
                self.completed_tasks.append(task)

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Queue length (normalized by max_queue_size)
        state.append(len(self.task_queue) / self.max_queue_size)
        
        # VM CPU utilizations
        for vm in self.vms:
            state.append(vm.cpu_utilization)
            
        # VM memory utilizations
        for vm in self.vms:
            state.append(vm.memory_utilization)
        
        # Current task features (if any)
        if len(self.task_queue) > 0:
            task = self.task_queue[0]
            # Normalize task features
            state.extend([
                task.cpu_requirement / max(vm.cpu_total for vm in self.vms),
                task.memory_requirement / max(vm.memory_total for vm in self.vms),
                task.priority / self.config["task"]["priority_levels"][1],
                (task.deadline - self.current_time) / self.config["task"]["deadline_range"][1]
            ])
        else:
            # No task in queue, pad with zeros
            state.extend([0, 0, 0, 0])
        
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, allocation_successful: bool, task: Task, vm: VirtualMachine) -> float:
        """Calculate reward based on multiple objectives"""
        if not allocation_successful:
            return -1.0  # Penalty for failed allocation
        
        # Get reward weights from config
        weights = self.config["reward"]
        
        # Completion time reward (negative, as we want to minimize it)
        completion_time_reward = -weights["completion_time_weight"] * (
            task.execution_time / self.config["task"]["execution_time"][1]
        )
        
        # Resource utilization reward
        utilization_reward = weights["utilization_weight"] * (
            (vm.cpu_utilization + vm.memory_utilization) / 2
        )
        
        # Task acceptance reward
        acceptance_reward = weights["acceptance_rate_weight"]
        
        # Combine rewards
        total_reward = completion_time_reward + utilization_reward + acceptance_reward
        
        return total_reward

    def _get_average_completion_time(self) -> float:
        """Calculate average task completion time"""
        if not self.completed_tasks:
            return 0.0
        completion_times = [
            task.completion_time - task.arrival_time 
            for task in self.completed_tasks
        ]
        return np.mean(completion_times)

    def _get_average_utilization(self) -> float:
        """Calculate average resource utilization across all VMs"""
        cpu_utils = [vm.cpu_utilization for vm in self.vms]
        memory_utils = [vm.memory_utilization for vm in self.vms]
        return np.mean(cpu_utils + memory_utils)
