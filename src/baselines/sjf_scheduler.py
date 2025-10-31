"""
Shortest Job First (SJF) Scheduler

Prioritizes shorter tasks to minimize average completion time.
Greedy algorithm that can cause starvation of longer tasks.
"""

from typing import List, Any
from .base_scheduler import BaseScheduler


class SJFScheduler(BaseScheduler):
    """
    Shortest Job First Scheduler.
    
    Among available VMs, selects the one that minimizes estimated task
    completion time. Uses task's base_execution_time as the job length metric.
    
    Note: This scheduler makes decisions based on task properties,
    but the actual allocation is still done in task arrival order.
    The SJF logic applies to VM selection for each task.
    """
    
    def __init__(self):
        super().__init__(name="SJF")
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select the VM that can complete the task fastest.
        
        Strategy: Choose VM with lowest current load (CPU utilization)
        to minimize queue time and overall completion time.
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of VM with lowest load that can accommodate task,
            or -1 if none available
        """
        best_vm = -1
        lowest_load = float('inf')
        
        for i, vm in enumerate(vms):
            if self.can_allocate(task, vm):
                # Calculate VM load (CPU utilization ratio)
                vm_load = vm.cpu_used / vm.cpu_total if vm.cpu_total > 0 else 0
                
                # Select VM with lowest current load
                if vm_load < lowest_load:
                    lowest_load = vm_load
                    best_vm = i
        
        return best_vm

