"""
Best Fit Scheduler

Allocates tasks to the VM with minimum remaining capacity after allocation.
Based on bin packing - minimizes wasted resources.
"""

from typing import List, Any
from .base_scheduler import BaseScheduler


class BestFitScheduler(BaseScheduler):
    """
    Best Fit Scheduler.
    
    Selects the VM that, after allocation, would have the minimum
    remaining capacity. This is similar to the Best Fit algorithm
    in bin packing problems.
    
    Aims to minimize resource fragmentation and maximize utilization.
    """
    
    def __init__(self):
        super().__init__(name="Best-Fit")
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select the VM with minimum remaining capacity after allocation.
        
        Calculates remaining capacity as the minimum of:
        - Remaining CPU ratio
        - Remaining memory ratio
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of best-fit VM, or -1 if none available
        """
        best_vm = -1
        min_remaining = float('inf')
        
        for i, vm in enumerate(vms):
            if self.can_allocate(task, vm):
                # Calculate remaining resources after allocation
                remaining_cpu = vm.cpu_total - (vm.cpu_used + task.cpu_requirement)
                remaining_memory = vm.memory_total - (vm.memory_used + task.memory_requirement)
                
                # Calculate remaining capacity as percentage
                cpu_remaining_pct = remaining_cpu / vm.cpu_total if vm.cpu_total > 0 else 0
                memory_remaining_pct = remaining_memory / vm.memory_total if vm.memory_total > 0 else 0
                
                # Use minimum of both as the remaining capacity metric
                # (bottleneck resource determines fit)
                remaining_capacity = min(cpu_remaining_pct, memory_remaining_pct)
                
                # Select VM with minimum remaining capacity (tightest fit)
                if remaining_capacity < min_remaining:
                    min_remaining = remaining_capacity
                    best_vm = i
        
        return best_vm

