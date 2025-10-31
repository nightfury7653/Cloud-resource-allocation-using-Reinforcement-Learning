"""
Performance Models for Realistic Cloud Simulation
Handles execution time calculation, performance degradation, and resource contention
"""

import numpy as np
from typing import List, Dict, Tuple
from .task_models import Task, VirtualMachine, TaskType

class PerformanceModel:
    """
    Models realistic task execution and performance characteristics
    """
    
    def __init__(self):
        """Initialize performance model"""
        # Contention factors
        self.cpu_contention_factor = 0.15  # 15% slowdown per competing task
        self.memory_contention_factor = 0.10  # 10% slowdown
        self.cache_miss_penalty = 0.20  # 20% penalty for cache misses
        
        # Performance degradation thresholds
        self.cpu_throttle_threshold = 0.80  # Start throttling at 80%
        self.memory_swap_threshold = 0.85  # Start swapping at 85%
        
    def calculate_execution_time(
        self,
        task: Task,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate realistic execution time for a task on a VM
        
        Args:
            task: Task to execute
            vm: VM executing the task
            
        Returns:
            Actual execution time considering all factors
        """
        base_time = task.base_execution_time
        
        # Calculate all performance factors
        resource_factor = self._calculate_resource_factor(task, vm)
        contention_factor = self._calculate_contention_factor(task, vm)
        degradation_factor = self._calculate_degradation_factor(vm)
        interference_factor = self._calculate_interference_factor(task, vm)
        
        # Combined performance impact
        total_factor = (resource_factor * 
                       contention_factor * 
                       degradation_factor *
                       interference_factor)
        
        # Calculate actual execution time
        actual_time = base_time * total_factor
        
        return max(base_time * 0.5, actual_time)  # Minimum 50% of base time
    
    def _calculate_resource_factor(
        self,
        task: Task,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate performance factor based on resource availability
        
        Args:
            task: Task to execute
            vm: VM executing the task
            
        Returns:
            Performance multiplier (>1 means slower)
        """
        factor = 1.0
        
        # CPU resource factor
        cpu_ratio = task.cpu_requirement / vm.cpu_total
        if cpu_ratio > 0.5:
            # Task uses more than half of CPU
            factor *= (1 + 0.2 * (cpu_ratio - 0.5))
        
        # Memory resource factor
        memory_ratio = task.memory_requirement / vm.memory_total
        if memory_ratio > 0.6:
            # High memory usage can cause slowdown
            factor *= (1 + 0.3 * (memory_ratio - 0.6))
        
        return factor
    
    def _calculate_contention_factor(
        self,
        task: Task,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate slowdown due to resource contention
        
        Args:
            task: Task to execute
            vm: VM executing the task
            
        Returns:
            Contention multiplier (>1 means slower)
        """
        factor = 1.0
        
        num_tasks = len(vm.tasks)
        if num_tasks <= 1:
            return factor
        
        # Base contention from number of tasks
        factor += self.cpu_contention_factor * (num_tasks - 1)
        
        # Additional contention based on task types
        cpu_intensive_tasks = sum(
            1 for t in vm.tasks 
            if t.task_type == TaskType.CPU_INTENSIVE
        )
        
        if task.task_type == TaskType.CPU_INTENSIVE and cpu_intensive_tasks > 1:
            # CPU-intensive tasks interfere more with each other
            factor += 0.1 * (cpu_intensive_tasks - 1)
        
        # Memory contention
        total_memory_req = sum(t.memory_requirement for t in vm.tasks)
        if total_memory_req / vm.memory_total > 0.7:
            memory_pressure = (total_memory_req / vm.memory_total - 0.7) / 0.3
            factor += self.memory_contention_factor * memory_pressure
        
        return factor
    
    def _calculate_degradation_factor(
        self,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate performance degradation due to system state
        
        Args:
            vm: Virtual machine
            
        Returns:
            Degradation multiplier (>1 means slower)
        """
        factor = 1.0
        
        # CPU throttling
        if vm.cpu_utilization > self.cpu_throttle_threshold:
            throttle_amount = (vm.cpu_utilization - self.cpu_throttle_threshold) / (1 - self.cpu_throttle_threshold)
            factor += 0.5 * throttle_amount
        
        # Memory swapping
        if vm.memory_utilization > self.memory_swap_threshold:
            swap_amount = (vm.memory_utilization - self.memory_swap_threshold) / (1 - self.memory_swap_threshold)
            factor += 1.0 * swap_amount  # Swapping is expensive
        
        # Cache efficiency
        cache_penalty = (1 - vm.cache_state) * self.cache_miss_penalty
        factor += cache_penalty
        
        # Memory fragmentation
        fragmentation_penalty = vm.fragmentation * 0.15
        factor += fragmentation_penalty
        
        return factor
    
    def _calculate_interference_factor(
        self,
        task: Task,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate task interference based on co-located tasks
        
        Args:
            task: Task to execute
            vm: VM executing the task
            
        Returns:
            Interference multiplier (>1 means slower)
        """
        if len(vm.tasks) <= 1:
            return 1.0
        
        factor = 1.0
        
        # Calculate interference based on task profiles
        for other_task in vm.tasks:
            if other_task.id == task.id:
                continue
            
            # Cache interference
            if (task.profile.cache_sensitivity > 0.5 and 
                other_task.profile.cache_sensitivity > 0.5):
                factor += 0.05
            
            # I/O interference
            if (task.profile.io_intensity > 0.6 and 
                other_task.profile.io_intensity > 0.6):
                factor += 0.08
            
            # Memory bandwidth interference
            if (task.profile.memory_intensity > 0.6 and 
                other_task.profile.memory_intensity > 0.6):
                factor += 0.10
        
        return factor
    
    def update_task_progress(
        self,
        task: Task,
        vm: VirtualMachine,
        time_delta: float
    ) -> float:
        """
        Update task progress based on time elapsed
        
        Args:
            task: Task being executed
            vm: VM executing the task
            time_delta: Time elapsed
            
        Returns:
            New progress value (0-1)
        """
        # Get current performance factor
        perf_factor = vm.get_performance_factor()
        
        # Calculate effective progress
        effective_time = time_delta * perf_factor
        progress_increment = effective_time / task.actual_execution_time
        
        # Update task progress
        new_progress = min(1.0, task.progress + progress_increment)
        
        return new_progress


class NetworkModel:
    """
    Models network latency and bandwidth constraints
    """
    
    def __init__(self):
        """Initialize network model"""
        # Network latency parameters (milliseconds)
        self.base_latency = 1.0  # Base latency
        self.latency_variance = 0.5  # Variance in latency
        
        # Bandwidth parameters (Mbps)
        self.base_bandwidth = 1000.0  # 1 Gbps
        self.bandwidth_contention_factor = 0.1
    
    def calculate_data_transfer_time(
        self,
        data_size_mb: float,
        num_concurrent_transfers: int = 1
    ) -> float:
        """
        Calculate time to transfer data
        
        Args:
            data_size_mb: Data size in MB
            num_concurrent_transfers: Number of concurrent transfers
            
        Returns:
            Transfer time in seconds
        """
        # Calculate effective bandwidth
        contention = 1 + self.bandwidth_contention_factor * (num_concurrent_transfers - 1)
        effective_bandwidth = self.base_bandwidth / contention
        
        # Transfer time = data_size / bandwidth
        transfer_time = (data_size_mb * 8) / effective_bandwidth  # Convert to seconds
        
        # Add latency
        latency = self.base_latency + np.random.normal(0, self.latency_variance)
        latency = max(0, latency) / 1000  # Convert ms to seconds
        
        return transfer_time + latency
    
    def get_network_delay(
        self,
        source_vm: int,
        dest_vm: int
    ) -> float:
        """
        Get network delay between two VMs
        
        Args:
            source_vm: Source VM ID
            dest_vm: Destination VM ID
            
        Returns:
            Network delay in seconds
        """
        if source_vm == dest_vm:
            return 0.0
        
        # Simplified model: random delay based on network topology
        base_delay = self.base_latency / 1000  # Convert to seconds
        variance = np.random.normal(0, self.latency_variance / 1000)
        
        return max(0, base_delay + variance)


class ResourceContentionModel:
    """
    Models resource contention and interference effects
    """
    
    def __init__(self):
        """Initialize contention model"""
        self.contention_matrix = {}  # Task type interference matrix
        self._initialize_contention_matrix()
    
    def _initialize_contention_matrix(self):
        """Initialize task interference matrix"""
        # Define how much different task types interfere with each other
        # Values represent slowdown factor (1.0 = no interference)
        task_types = list(TaskType)
        
        for t1 in task_types:
            for t2 in task_types:
                if t1 == t2:
                    # Same type tasks interfere more
                    self.contention_matrix[(t1, t2)] = 1.15
                elif (t1 == TaskType.CPU_INTENSIVE and t2 == TaskType.CPU_INTENSIVE):
                    self.contention_matrix[(t1, t2)] = 1.25
                elif (t1 == TaskType.MEMORY_INTENSIVE and t2 == TaskType.MEMORY_INTENSIVE):
                    self.contention_matrix[(t1, t2)] = 1.20
                else:
                    # Different types interfere less
                    self.contention_matrix[(t1, t2)] = 1.05
    
    def calculate_interference(
        self,
        task: Task,
        colocated_tasks: List[Task]
    ) -> float:
        """
        Calculate interference from co-located tasks
        
        Args:
            task: Task to calculate interference for
            colocated_tasks: Other tasks on the same VM
            
        Returns:
            Interference factor (>1 means slowdown)
        """
        if not colocated_tasks:
            return 1.0
        
        interference = 1.0
        
        for other_task in colocated_tasks:
            if other_task.id == task.id:
                continue
            
            # Get interference factor from matrix
            key = (task.task_type, other_task.task_type)
            task_interference = self.contention_matrix.get(key, 1.05)
            
            # Accumulate interference (multiplicative)
            interference *= task_interference
        
        return interference
    
    def get_cache_contention(
        self,
        task: Task,
        vm: VirtualMachine
    ) -> float:
        """
        Calculate cache contention factor
        
        Args:
            task: Task to check
            vm: VM executing the task
            
        Returns:
            Cache contention factor
        """
        if len(vm.tasks) <= 1:
            return 1.0
        
        # Cache-sensitive tasks suffer more from contention
        base_contention = 1 + 0.05 * (len(vm.tasks) - 1)
        
        if task.profile.cache_sensitivity > 0.7:
            cache_penalty = 1 + 0.1 * (len(vm.tasks) - 1)
            return base_contention * cache_penalty
        
        return base_contention
