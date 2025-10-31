"""
First-Come-First-Serve (FCFS) Scheduler

Allocates tasks in order of arrival to the first available VM.
Simple queue-based scheduling.
"""

from typing import List, Any
from .base_scheduler import BaseScheduler


class FCFSScheduler(BaseScheduler):
    """
    First-Come-First-Serve Scheduler.
    
    Processes tasks in arrival order and allocates to the first VM
    that has sufficient resources. No reordering or prioritization.
    """
    
    def __init__(self):
        super().__init__(name="FCFS")
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select the first VM that can accommodate the task.
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of first available VM, or -1 if none available
        """
        # Iterate through VMs in order and return first suitable one
        for i, vm in enumerate(vms):
            if self.can_allocate(task, vm):
                return i
        
        # No suitable VM found
        return -1

