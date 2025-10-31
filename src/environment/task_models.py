"""
Enhanced Task Models for Realistic Cloud Simulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class TaskType(Enum):
    """Types of tasks with different resource usage patterns"""
    CPU_INTENSIVE = "cpu_intensive"      # High CPU, low I/O
    MEMORY_INTENSIVE = "memory_intensive"  # High memory, moderate CPU
    IO_INTENSIVE = "io_intensive"        # High I/O, lower CPU
    MIXED = "mixed"                      # Balanced resource usage
    BATCH = "batch"                      # Long-running, steady usage
    WEB_SERVICE = "web_service"          # Bursty, variable usage

@dataclass
class TaskProfile:
    """Performance profile for different task types"""
    cpu_intensity: float  # 0-1, how CPU-bound the task is
    memory_intensity: float  # 0-1, memory usage pattern
    io_intensity: float  # 0-1, I/O requirements
    burstiness: float  # 0-1, how bursty the resource usage is
    cache_sensitivity: float  # 0-1, sensitivity to cache effects

    @staticmethod
    def get_profile(task_type: TaskType) -> 'TaskProfile':
        """Get predefined profile for task type"""
        profiles = {
            TaskType.CPU_INTENSIVE: TaskProfile(0.9, 0.3, 0.1, 0.2, 0.7),
            TaskType.MEMORY_INTENSIVE: TaskProfile(0.3, 0.9, 0.2, 0.3, 0.4),
            TaskType.IO_INTENSIVE: TaskProfile(0.2, 0.3, 0.9, 0.6, 0.2),
            TaskType.MIXED: TaskProfile(0.5, 0.5, 0.5, 0.4, 0.5),
            TaskType.BATCH: TaskProfile(0.7, 0.4, 0.3, 0.1, 0.3),
            TaskType.WEB_SERVICE: TaskProfile(0.4, 0.5, 0.6, 0.8, 0.6)
        }
        return profiles[task_type]

@dataclass
class Task:
    """Enhanced task model with realistic properties"""
    # Basic properties
    id: int
    task_type: TaskType
    
    # Resource requirements
    cpu_requirement: float  # CPU cores needed
    memory_requirement: float  # Memory in GB
    
    # Timing properties
    arrival_time: float
    deadline: float
    base_execution_time: float  # Expected time under ideal conditions
    priority: int  # Moved before fields with defaults
    
    # Fields with defaults
    actual_execution_time: float = 0.0  # Will be calculated based on conditions
    user_id: Optional[int] = None
    
    # Runtime properties
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_vm: int = -1
    completed: bool = False
    failed: bool = False
    
    # Performance tracking
    progress: float = 0.0  # 0-1, task completion progress
    current_cpu_usage: float = 0.0
    current_memory_usage: float = 0.0
    
    # Task profile
    profile: TaskProfile = field(default_factory=lambda: TaskProfile(0.5, 0.5, 0.5, 0.3, 0.5))
    
    # Dependencies
    dependencies: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize task profile based on type"""
        self.profile = TaskProfile.get_profile(self.task_type)
        self.actual_execution_time = self.base_execution_time
    
    @property
    def is_deadline_met(self) -> bool:
        """Check if task met its deadline"""
        if not self.completed:
            return False
        return self.completion_time <= self.deadline
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue"""
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time
    
    @property
    def execution_duration(self) -> float:
        """Actual execution duration"""
        if self.start_time is None or self.completion_time is None:
            return 0.0
        return self.completion_time - self.start_time
    
    @property
    def total_time(self) -> float:
        """Total time from arrival to completion"""
        if not self.completed:
            return 0.0
        return self.completion_time - self.arrival_time
    
    def update_resource_usage(self, elapsed_time: float, vm_load: float):
        """
        Update dynamic resource usage based on task progress and VM load
        
        Args:
            elapsed_time: Time since task started
            vm_load: Current VM load (0-1)
        """
        # Calculate base usage pattern
        if self.profile.burstiness > 0.5:
            # Bursty pattern
            burst_factor = np.sin(elapsed_time * 2 * np.pi / (self.base_execution_time / 3))
            usage_factor = 0.7 + 0.3 * burst_factor
        else:
            # Steady pattern
            usage_factor = 1.0
        
        # Adjust for VM load
        if vm_load > 0.7:
            usage_factor *= (1 - 0.2 * (vm_load - 0.7))
        
        # Update current usage
        self.current_cpu_usage = self.cpu_requirement * usage_factor * self.profile.cpu_intensity
        self.current_memory_usage = self.memory_requirement * usage_factor * self.profile.memory_intensity

@dataclass
class VirtualMachine:
    """Enhanced VM model with realistic resource management"""
    # Basic properties
    id: int
    vm_type: str  # "small", "medium", "large"
    
    # Total resources
    cpu_total: float
    memory_total: float
    storage_total: float = 100.0  # GB
    
    # Current usage
    cpu_used: float = 0.0
    memory_used: float = 0.0
    storage_used: float = 0.0
    
    # Performance characteristics
    cpu_clock_speed: float = 2.5  # GHz
    memory_bandwidth: float = 25.0  # GB/s
    network_bandwidth: float = 1.0  # Gbps
    
    # State tracking
    tasks: List[Task] = field(default_factory=list)
    is_healthy: bool = True
    temperature: float = 50.0  # Celsius
    
    # Performance degradation factors
    cache_state: float = 1.0  # 0-1, cache efficiency
    fragmentation: float = 0.0  # 0-1, memory fragmentation
    
    # History for tracking
    cpu_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)
    
    @property
    def cpu_utilization(self) -> float:
        """Current CPU utilization ratio"""
        return min(1.0, self.cpu_used / self.cpu_total)
    
    @property
    def memory_utilization(self) -> float:
        """Current memory utilization ratio"""
        return min(1.0, self.memory_used / self.memory_total)
    
    @property
    def cpu_available(self) -> float:
        """Available CPU capacity"""
        return max(0.0, self.cpu_total - self.cpu_used)
    
    @property
    def memory_available(self) -> float:
        """Available memory"""
        return max(0.0, self.memory_total - self.memory_used)
    
    @property
    def is_overloaded(self) -> bool:
        """Check if VM is overloaded"""
        return self.cpu_utilization > 0.9 or self.memory_utilization > 0.85
    
    def can_accommodate(self, task: Task) -> bool:
        """
        Check if VM can accommodate a new task
        
        Args:
            task: Task to check
            
        Returns:
            bool: True if task can be accommodated
        """
        return (self.cpu_available >= task.cpu_requirement and
                self.memory_available >= task.memory_requirement)
    
    def allocate_task(self, task: Task) -> bool:
        """
        Allocate a task to this VM
        
        Args:
            task: Task to allocate
            
        Returns:
            bool: True if allocation successful
        """
        if not self.can_accommodate(task):
            return False
        
        # Allocate resources
        self.cpu_used += task.cpu_requirement
        self.memory_used += task.memory_requirement
        
        # Add task
        self.tasks.append(task)
        task.assigned_vm = self.id
        
        # Update cache state (more tasks = more cache pollution)
        self.cache_state *= 0.95
        
        # Update fragmentation
        self.fragmentation = min(1.0, self.fragmentation + 0.05 * len(self.tasks))
        
        return True
    
    def release_task(self, task: Task):
        """
        Release a completed task and free resources
        
        Args:
            task: Task to release
        """
        if task in self.tasks:
            # Free resources
            self.cpu_used = max(0.0, self.cpu_used - task.cpu_requirement)
            self.memory_used = max(0.0, self.memory_used - task.memory_requirement)
            
            # Remove task
            self.tasks.remove(task)
            
            # Improve cache state when tasks complete
            if len(self.tasks) == 0:
                self.cache_state = 1.0
                self.fragmentation = 0.0
            else:
                self.cache_state = min(1.0, self.cache_state * 1.05)
                self.fragmentation = max(0.0, self.fragmentation - 0.03)
    
    def get_performance_factor(self) -> float:
        """
        Calculate current performance factor based on load and state
        
        Returns:
            float: Performance multiplier (< 1.0 means degraded performance)
        """
        base_performance = 1.0
        
        # CPU throttling at high utilization
        if self.cpu_utilization > 0.8:
            cpu_penalty = (self.cpu_utilization - 0.8) * 2.0
            base_performance *= (1.0 - cpu_penalty)
        
        # Memory swapping penalty
        if self.memory_utilization > 0.85:
            swap_penalty = (self.memory_utilization - 0.85) * 3.0
            base_performance *= (1.0 - swap_penalty)
        
        # Cache efficiency
        base_performance *= (0.5 + 0.5 * self.cache_state)
        
        # Fragmentation penalty
        base_performance *= (1.0 - 0.2 * self.fragmentation)
        
        # Temperature throttling (simplified)
        if self.temperature > 80:
            thermal_penalty = (self.temperature - 80) / 100
            base_performance *= (1.0 - thermal_penalty)
        
        return max(0.1, base_performance)  # Minimum 10% performance
    
    def update_state(self, time_delta: float):
        """
        Update VM state over time
        
        Args:
            time_delta: Time elapsed since last update
        """
        # Update temperature based on load
        target_temp = 50 + 40 * self.cpu_utilization
        self.temperature += (target_temp - self.temperature) * 0.1
        
        # Track history
        self.cpu_history.append(self.cpu_utilization)
        self.memory_history.append(self.memory_utilization)
        
        # Keep only recent history (last 100 entries)
        if len(self.cpu_history) > 100:
            self.cpu_history.pop(0)
            self.memory_history.pop(0)
    
    def get_contention_factor(self) -> float:
        """
        Calculate resource contention factor
        
        Returns:
            float: Contention multiplier (> 1.0 means more contention)
        """
        num_tasks = len(self.tasks)
        if num_tasks <= 1:
            return 1.0
        
        # Base contention from number of tasks
        base_contention = 1.0 + 0.1 * (num_tasks - 1)
        
        # Additional contention from resource pressure
        resource_pressure = (self.cpu_utilization + self.memory_utilization) / 2
        pressure_factor = 1.0 + 0.5 * resource_pressure
        
        return base_contention * pressure_factor
