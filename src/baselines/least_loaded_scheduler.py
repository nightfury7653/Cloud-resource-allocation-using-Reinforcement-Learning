"""
Least Loaded Scheduler

Allocates tasks to the VM with the lowest current utilization.
Load balancing strategy to distribute work evenly.
"""

from typing import List, Any
from .base_scheduler import BaseScheduler


class LeastLoadedScheduler(BaseScheduler):
    """
    Least Loaded Scheduler.
    
    Selects the VM with the lowest current resource utilization.
    This is a common load balancing strategy that aims to distribute
    work evenly across all VMs.
    
    Load is calculated as the average of CPU and memory utilization.
    """
    
    def __init__(self):
        super().__init__(name="Least-Loaded")
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select the VM with lowest current utilization.
        
        Utilization is calculated as the average of:
        - CPU utilization (cpu_used / cpu_total)
        - Memory utilization (memory_used / memory_total)
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of least loaded VM, or -1 if none available
        """
        best_vm = -1
        lowest_load = float('inf')
        
        for i, vm in enumerate(vms):
            if self.can_allocate(task, vm):
                # Calculate CPU and memory utilization
                cpu_util = vm.cpu_used / vm.cpu_total if vm.cpu_total > 0 else 0
                memory_util = vm.memory_used / vm.memory_total if vm.memory_total > 0 else 0
                
                # Average utilization as load metric
                vm_load = (cpu_util + memory_util) / 2.0
                
                # Select VM with lowest load
                if vm_load < lowest_load:
                    lowest_load = vm_load
                    best_vm = i
        
        return best_vm

