"""
Round-Robin Scheduler

Distributes tasks across VMs in a circular fashion.
Simple load distribution strategy.
"""

from typing import List, Any
from .base_scheduler import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """
    Round-Robin VM Selection Scheduler.
    
    Cycles through VMs in order, attempting to distribute load evenly.
    Maintains a pointer to the last selected VM.
    """
    
    def __init__(self):
        super().__init__(name="Round-Robin")
        self.last_vm_index = -1
    
    def reset(self):
        """Reset the VM pointer at the start of each episode"""
        self.last_vm_index = -1
    
    def select_vm(self, task: Any, vms: List[Any], current_time: float) -> int:
        """
        Select the next VM in round-robin order that can accommodate the task.
        
        Args:
            task: Task to allocate
            vms: List of available VMs
            current_time: Current simulation time
        
        Returns:
            Index of selected VM, or -1 if none available
        """
        num_vms = len(vms)
        
        # Try each VM starting from the next one in round-robin order
        for offset in range(num_vms):
            # Calculate VM index in round-robin fashion
            vm_index = (self.last_vm_index + 1 + offset) % num_vms
            
            # Check if this VM can accommodate the task
            if self.can_allocate(task, vms[vm_index]):
                # Update last selected VM
                self.last_vm_index = vm_index
                return vm_index
        
        # No suitable VM found
        return -1

