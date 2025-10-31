"""
Random Scheduler

Randomly selects a VM for task allocation.
This is the naive baseline - worst case performance.
"""

import random
from typing import List, Any
from .base_scheduler import BaseScheduler


class RandomScheduler(BaseScheduler):
    """
    Random VM Selection Scheduler.
    
    Selects a random VM from those that can accommodate the task.
    This provides a lower bound on expected performance.
    """
    
    def __init__(self):
        super().__init__(name="Random")
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select a random VM that can accommodate the task.
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of randomly selected VM, or -1 if none available
        """
        # Find all VMs that can accommodate the task
        available_vms = []
        
        for i, vm in enumerate(vms):
            if self.can_allocate(task, vm):
                available_vms.append(i)
        
        # If no VMs available, return -1
        if not available_vms:
            return -1
        
        # Randomly select from available VMs
        return random.choice(available_vms)

