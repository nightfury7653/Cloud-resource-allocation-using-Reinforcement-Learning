"""
Realistic Workload Generator for Cloud Simulation
Supports real trace data and synthetic pattern generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import pandas as pd
from pathlib import Path

from .task_models import Task, TaskType, TaskProfile

class WorkloadPattern(Enum):
    """Different workload patterns"""
    CONSTANT = "constant"  # Steady arrival rate
    PERIODIC = "periodic"  # Daily/weekly patterns
    BURSTY = "bursty"  # Sudden spikes
    TRENDING = "trending"  # Increasing/decreasing trend
    REAL_TRACE = "real_trace"  # Based on real data

class WorkloadGenerator:
    """Generate realistic workload patterns for cloud simulation"""
    
    def __init__(
        self,
        pattern: WorkloadPattern = WorkloadPattern.PERIODIC,
        base_arrival_rate: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Initialize workload generator
        
        Args:
            pattern: Type of workload pattern
            base_arrival_rate: Average tasks per time unit
            seed: Random seed for reproducibility
        """
        self.pattern = pattern
        self.base_arrival_rate = base_arrival_rate
        self.current_time = 0
        self.task_id_counter = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        # Task type distribution (based on real cloud data)
        self.task_type_distribution = {
            TaskType.CPU_INTENSIVE: 0.25,
            TaskType.MEMORY_INTENSIVE: 0.15,
            TaskType.IO_INTENSIVE: 0.20,
            TaskType.MIXED: 0.25,
            TaskType.BATCH: 0.10,
            TaskType.WEB_SERVICE: 0.05
        }
        
        # Resource requirement distributions (based on Google cluster data)
        self.cpu_dist_params = {
            'mean': 1.5,
            'std': 1.0,
            'min': 0.5,
            'max': 8.0
        }
        
        self.memory_dist_params = {
            'mean': 4.0,
            'std': 3.0,
            'min': 1.0,
            'max': 16.0
        }
        
        # Execution time distribution (log-normal, as observed in real data)
        self.exec_time_params = {
            'mu': 4.5,  # log(mean)
            'sigma': 1.5,  # log(std)
            'min': 10.0,
            'max': 600.0
        }
    
    def generate_tasks(
        self,
        current_time: float,
        time_delta: float = 1.0
    ) -> List[Task]:
        """
        Generate tasks for the current time step
        
        Args:
            current_time: Current simulation time
            time_delta: Time since last generation
            
        Returns:
            List of newly generated tasks
        """
        self.current_time = current_time
        
        # Get arrival rate for current time
        arrival_rate = self._get_arrival_rate(current_time)
        
        # Generate number of arrivals (Poisson process)
        num_arrivals = np.random.poisson(arrival_rate * time_delta)
        
        # Generate tasks
        tasks = []
        for _ in range(num_arrivals):
            task = self._generate_single_task(current_time)
            tasks.append(task)
        
        return tasks
    
    def _get_arrival_rate(self, time: float) -> float:
        """
        Get arrival rate based on pattern and time
        
        Args:
            time: Current time
            
        Returns:
            Arrival rate for this time
        """
        if self.pattern == WorkloadPattern.CONSTANT:
            return self.base_arrival_rate
        
        elif self.pattern == WorkloadPattern.PERIODIC:
            # Daily pattern (24-hour cycle)
            hour = (time % 24)
            
            # Peak hours: 9 AM - 5 PM
            if 9 <= hour <= 17:
                rate_multiplier = 1.5
            # Night hours: 11 PM - 5 AM
            elif hour <= 5 or hour >= 23:
                rate_multiplier = 0.5
            else:
                rate_multiplier = 1.0
            
            # Add some random variation
            rate_multiplier *= (0.9 + 0.2 * np.random.random())
            
            return self.base_arrival_rate * rate_multiplier
        
        elif self.pattern == WorkloadPattern.BURSTY:
            # Random bursts with exponential inter-arrival
            if np.random.random() < 0.1:  # 10% chance of burst
                return self.base_arrival_rate * (3 + 2 * np.random.random())
            else:
                return self.base_arrival_rate * 0.7
        
        elif self.pattern == WorkloadPattern.TRENDING:
            # Linear trend
            trend_factor = 1.0 + 0.0001 * time
            return self.base_arrival_rate * trend_factor
        
        else:
            return self.base_arrival_rate
    
    def _generate_single_task(self, arrival_time: float) -> Task:
        """
        Generate a single task with realistic properties
        
        Args:
            arrival_time: Task arrival time
            
        Returns:
            Generated task
        """
        # Select task type based on distribution
        task_type = self._sample_task_type()
        
        # Generate resource requirements
        cpu_req = self._sample_cpu_requirement(task_type)
        memory_req = self._sample_memory_requirement(task_type)
        
        # Generate execution time
        exec_time = self._sample_execution_time(task_type)
        
        # Generate deadline (based on execution time)
        slack_factor = 2.0 + np.random.exponential(1.0)
        deadline = arrival_time + exec_time * slack_factor
        
        # Generate priority
        priority = self._sample_priority(task_type)
        
        # Create task
        task = Task(
            id=self.task_id_counter,
            task_type=task_type,
            cpu_requirement=cpu_req,
            memory_requirement=memory_req,
            arrival_time=arrival_time,
            deadline=deadline,
            base_execution_time=exec_time,
            priority=priority,
            user_id=np.random.randint(0, 100)
        )
        
        self.task_id_counter += 1
        return task
    
    def _sample_task_type(self) -> TaskType:
        """Sample task type from distribution"""
        types = list(self.task_type_distribution.keys())
        probs = list(self.task_type_distribution.values())
        return np.random.choice(types, p=probs)
    
    def _sample_cpu_requirement(self, task_type: TaskType) -> float:
        """
        Sample CPU requirement based on task type
        
        Args:
            task_type: Type of task
            
        Returns:
            CPU requirement in cores
        """
        params = self.cpu_dist_params
        
        # Adjust mean based on task type
        if task_type == TaskType.CPU_INTENSIVE:
            mean = params['mean'] * 1.5
        elif task_type == TaskType.MEMORY_INTENSIVE:
            mean = params['mean'] * 0.7
        else:
            mean = params['mean']
        
        # Sample from truncated normal
        cpu = np.random.normal(mean, params['std'])
        return np.clip(cpu, params['min'], params['max'])
    
    def _sample_memory_requirement(self, task_type: TaskType) -> float:
        """
        Sample memory requirement based on task type
        
        Args:
            task_type: Type of task
            
        Returns:
            Memory requirement in GB
        """
        params = self.memory_dist_params
        
        # Adjust mean based on task type
        if task_type == TaskType.MEMORY_INTENSIVE:
            mean = params['mean'] * 1.8
        elif task_type == TaskType.CPU_INTENSIVE:
            mean = params['mean'] * 0.6
        else:
            mean = params['mean']
        
        # Sample from truncated normal
        memory = np.random.normal(mean, params['std'])
        return np.clip(memory, params['min'], params['max'])
    
    def _sample_execution_time(self, task_type: TaskType) -> float:
        """
        Sample execution time (log-normal distribution)
        
        Args:
            task_type: Type of task
            
        Returns:
            Execution time in seconds
        """
        params = self.exec_time_params
        
        # Adjust parameters based on task type
        if task_type == TaskType.BATCH:
            mu = params['mu'] + 0.5  # Longer tasks
        elif task_type == TaskType.WEB_SERVICE:
            mu = params['mu'] - 0.5  # Shorter tasks
        else:
            mu = params['mu']
        
        # Sample from log-normal
        exec_time = np.random.lognormal(mu, params['sigma'])
        return np.clip(exec_time, params['min'], params['max'])
    
    def _sample_priority(self, task_type: TaskType) -> int:
        """
        Sample task priority
        
        Args:
            task_type: Type of task
            
        Returns:
            Priority level (1-5)
        """
        if task_type == TaskType.WEB_SERVICE:
            # Higher priority for interactive tasks
            return np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        elif task_type == TaskType.BATCH:
            # Lower priority for batch tasks
            return np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:
            # Normal distribution
            return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])


class RealTraceWorkloadGenerator(WorkloadGenerator):
    """
    Workload generator based on real cloud traces
    Supports Google Cluster Data and Alibaba traces
    """
    
    def __init__(
        self,
        trace_file: Optional[str] = None,
        base_arrival_rate: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Initialize trace-based workload generator
        
        Args:
            trace_file: Path to trace data file
            base_arrival_rate: Fallback arrival rate
            seed: Random seed
        """
        super().__init__(
            pattern=WorkloadPattern.REAL_TRACE,
            base_arrival_rate=base_arrival_rate,
            seed=seed
        )
        
        self.trace_data = None
        self.trace_index = 0
        
        if trace_file and Path(trace_file).exists():
            self._load_trace_data(trace_file)
    
    def _load_trace_data(self, trace_file: str):
        """
        Load and preprocess trace data
        
        Args:
            trace_file: Path to trace file
        """
        try:
            # Load trace data (assuming CSV format)
            self.trace_data = pd.read_csv(trace_file)
            
            # Extract relevant columns and normalize
            # Expected columns: timestamp, cpu, memory, duration
            if 'timestamp' in self.trace_data.columns:
                self.trace_data = self.trace_data.sort_values('timestamp')
            
            print(f"Loaded {len(self.trace_data)} task entries from trace")
            
        except Exception as e:
            print(f"Warning: Could not load trace data: {e}")
            print("Falling back to synthetic generation")
            self.trace_data = None
    
    def generate_tasks(
        self,
        current_time: float,
        time_delta: float = 1.0
    ) -> List[Task]:
        """
        Generate tasks from real trace data
        
        Args:
            current_time: Current simulation time
            time_delta: Time delta
            
        Returns:
            List of tasks
        """
        if self.trace_data is None:
            # Fall back to synthetic generation
            return super().generate_tasks(current_time, time_delta)
        
        # Generate tasks from trace data
        tasks = []
        
        # Sample from trace based on current time
        # This is a simplified approach - real implementation would be more sophisticated
        num_tasks = np.random.poisson(self.base_arrival_rate * time_delta)
        
        for _ in range(num_tasks):
            if self.trace_index < len(self.trace_data):
                task = self._task_from_trace(
                    self.trace_data.iloc[self.trace_index],
                    current_time
                )
                tasks.append(task)
                self.trace_index += 1
            else:
                # Wrap around or stop
                self.trace_index = 0
        
        return tasks
    
    def _task_from_trace(self, trace_entry: pd.Series, arrival_time: float) -> Task:
        """
        Create task from trace entry
        
        Args:
            trace_entry: Row from trace data
            arrival_time: Task arrival time
            
        Returns:
            Task object
        """
        # Extract data from trace (adapt to your trace format)
        cpu_req = trace_entry.get('cpu', np.random.uniform(0.5, 4.0))
        memory_req = trace_entry.get('memory', np.random.uniform(1.0, 8.0))
        exec_time = trace_entry.get('duration', np.random.uniform(10, 300))
        
        # Infer task type from resource requirements
        if cpu_req > 2.0 and memory_req < 4.0:
            task_type = TaskType.CPU_INTENSIVE
        elif memory_req > 6.0:
            task_type = TaskType.MEMORY_INTENSIVE
        else:
            task_type = TaskType.MIXED
        
        deadline = arrival_time + exec_time * (2 + np.random.random())
        priority = np.random.randint(1, 6)
        
        task = Task(
            id=self.task_id_counter,
            task_type=task_type,
            cpu_requirement=cpu_req,
            memory_requirement=memory_req,
            arrival_time=arrival_time,
            deadline=deadline,
            base_execution_time=exec_time,
            priority=priority
        )
        
        self.task_id_counter += 1
        return task
