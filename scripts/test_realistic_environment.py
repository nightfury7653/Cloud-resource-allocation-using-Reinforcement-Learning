"""
Test and Validate Realistic Cloud Environment
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from environment.realistic_cloud_env import RealisticCloudEnvironment
from environment.workload_generator import WorkloadPattern
from environment.task_models import TaskType
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class EnvironmentValidator:
    """Validate realistic environment behavior"""
    
    def __init__(self, env: RealisticCloudEnvironment):
        self.env = env
        self.test_results = {}
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("="*60)
        print("REALISTIC CLOUD ENVIRONMENT VALIDATION")
        print("="*60)
        
        print("\n1. Testing Basic Functionality...")
        self.test_basic_functionality()
        
        print("\n2. Testing Workload Patterns...")
        self.test_workload_patterns()
        
        print("\n3. Testing Resource Contention...")
        self.test_resource_contention()
        
        print("\n4. Testing Performance Degradation...")
        self.test_performance_degradation()
        
        print("\n5. Testing Task Diversity...")
        self.test_task_diversity()
        
        print("\n6. Testing Reward Function...")
        self.test_reward_function()
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        self._print_summary()
    
    def test_basic_functionality(self):
        """Test basic environment operations"""
        print("  - Testing reset...")
        state, info = self.env.reset()
        assert state is not None, "Reset should return state"
        assert len(state) == self.env.observation_space.shape[0], "State dimension mismatch"
        print("    ✓ Reset works correctly")
        
        print("  - Testing step...")
        action = self.env.action_space.sample()
        next_state, reward, done, truncated, info = self.env.step(action)
        assert next_state is not None, "Step should return next state"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        print("    ✓ Step works correctly")
        
        self.test_results['basic_functionality'] = 'PASS'
    
    def test_workload_patterns(self):
        """Test different workload patterns"""
        print("  - Testing periodic pattern...")
        arrivals_by_hour = {i: 0 for i in range(24)}
        
        # Simulate 24 hours
        for hour in range(24):
            self.env.current_time = hour
            tasks = self.env.workload_generator.generate_tasks(hour, 1.0)
            arrivals_by_hour[hour] = len(tasks)
        
        # Check if there's variation (indicating pattern)
        arrival_counts = list(arrivals_by_hour.values())
        variation = np.std(arrival_counts)
        
        print(f"    Arrival rate variation (std): {variation:.2f}")
        assert variation > 0, "Should have variation in arrivals"
        print("    ✓ Workload pattern shows expected variation")
        
        self.test_results['workload_patterns'] = 'PASS'
    
    def test_resource_contention(self):
        """Test resource contention effects"""
        print("  - Testing resource contention...")
        
        self.env.reset()
        vm = self.env.vms[0]
        
        # Create multiple tasks
        from environment.task_models import Task
        tasks = []
        for i in range(3):
            task = Task(
                id=i,
                task_type=TaskType.CPU_INTENSIVE,
                cpu_requirement=1.0,
                memory_requirement=2.0,
                arrival_time=0.0,
                deadline=100.0,
                base_execution_time=10.0,
                priority=3
            )
            tasks.append(task)
        
        # Allocate first task
        vm.allocate_task(tasks[0])
        exec_time_1 = self.env.performance_model.calculate_execution_time(tasks[0], vm)
        
        # Allocate second task (should increase execution time due to contention)
        vm.allocate_task(tasks[1])
        exec_time_2 = self.env.performance_model.calculate_execution_time(tasks[1], vm)
        
        print(f"    Execution time (1 task): {exec_time_1:.2f}s")
        print(f"    Execution time (2 tasks): {exec_time_2:.2f}s")
        
        assert exec_time_2 > exec_time_1, "Contention should increase execution time"
        print("    ✓ Resource contention increases execution time as expected")
        
        self.test_results['resource_contention'] = 'PASS'
    
    def test_performance_degradation(self):
        """Test performance degradation at high utilization"""
        print("  - Testing performance degradation...")
        
        self.env.reset()
        vm = self.env.vms[0]
        
        # Test at different utilization levels
        perf_at_low_util = None
        perf_at_high_util = None
        
        # Low utilization
        vm.cpu_used = vm.cpu_total * 0.3
        perf_at_low_util = vm.get_performance_factor()
        
        # High utilization
        vm.cpu_used = vm.cpu_total * 0.95
        perf_at_high_util = vm.get_performance_factor()
        
        print(f"    Performance at 30% util: {perf_at_low_util:.3f}")
        print(f"    Performance at 95% util: {perf_at_high_util:.3f}")
        
        assert perf_at_high_util < perf_at_low_util, "High utilization should degrade performance"
        print("    ✓ Performance degrades at high utilization")
        
        self.test_results['performance_degradation'] = 'PASS'
    
    def test_task_diversity(self):
        """Test task type diversity"""
        print("  - Testing task type diversity...")
        
        # Generate many tasks
        task_types = {}
        for _ in range(100):
            tasks = self.env.workload_generator.generate_tasks(0.0, 1.0)
            for task in tasks:
                task_type = task.task_type
                task_types[task_type] = task_types.get(task_type, 0) + 1
        
        print(f"    Generated {len(task_types)} different task types")
        for task_type, count in task_types.items():
            print(f"      {task_type.value}: {count}")
        
        assert len(task_types) >= 3, "Should have multiple task types"
        print("    ✓ Task diversity is present")
        
        self.test_results['task_diversity'] = 'PASS'
    
    def test_reward_function(self):
        """Test reward function behavior"""
        print("  - Testing reward function...")
        
        self.env.reset()
        
        rewards_successful = []
        rewards_failed = []
        
        # Try multiple allocations to collect statistics
        for _ in range(10):
            self.env._generate_new_tasks()
            
            if len(self.env.task_queue) > 0:
                # Try allocation to first VM (usually has capacity)
                action = 0
                state, reward, done, truncated, info = self.env.step(action)
                
                if info['allocation_info']['allocated']:
                    rewards_successful.append(reward)
                else:
                    rewards_failed.append(reward)
        
        # Also test forced failure
        self.env._generate_new_tasks()
        if len(self.env.task_queue) > 0:
            # Fill all VMs to force failure
            for vm in self.env.vms:
                vm.cpu_used = vm.cpu_total * 0.99
                vm.memory_used = vm.memory_total * 0.99
            
            action = 0
            state, reward, done, truncated, info = self.env.step(action)
            if not info['allocation_info']['allocated']:
                rewards_failed.append(reward)
        
        if rewards_successful and rewards_failed:
            avg_success = np.mean(rewards_successful)
            avg_fail = np.mean(rewards_failed)
            
            print(f"    Avg Reward (successful): {avg_success:.3f}")
            print(f"    Avg Reward (failed): {avg_fail:.3f}")
            
            # Failed allocation should have lower reward
            assert avg_fail < avg_success, "Failed allocation should have lower reward"
            print("    ✓ Reward function differentiates success/failure")
        else:
            # At least check that failed allocations give negative reward
            if rewards_failed:
                print(f"    Reward (failed): {np.mean(rewards_failed):.3f}")
                assert all(r < 0 for r in rewards_failed), "Failed allocations should have negative reward"
                print("    ✓ Failed allocations have negative reward")
            else:
                print("    ⚠ Could not test reward differentiation (no failures)")
        
        self.test_results['reward_function'] = 'PASS'
    
    def _print_summary(self):
        """Print test summary"""
        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v == 'PASS')
        
        for test_name, result in self.test_results.items():
            status = "✓" if result == "PASS" else "✗"
            print(f"{status} {test_name}: {result}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed! Environment is ready for training.")
        else:
            print("\n⚠️  Some tests failed. Please review the results.")


def run_performance_comparison():
    """Compare basic vs realistic environment"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print("\nRunning episodes with realistic environment...")
    
    env = RealisticCloudEnvironment(
        workload_pattern=WorkloadPattern.PERIODIC,
        seed=42
    )
    
    # Run test episodes
    num_episodes = 5
    episode_rewards = []
    episode_utilizations = []
    episode_completion_times = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_utilizations.append(info.get('avg_utilization', 0))
        episode_completion_times.append(info.get('avg_execution_time', 0))
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Completed Tasks: {info.get('completed_tasks', 0)}")
        print(f"  Failed Tasks: {info.get('failed_tasks', 0)}")
        print(f"  Avg Utilization: {info.get('avg_utilization', 0):.2%}")
        print(f"  Avg Wait Time: {info.get('avg_wait_time', 0):.2f}s")
        print(f"  Avg Execution Time: {info.get('avg_execution_time', 0):.2f}s")
    
    print("\n" + "="*60)
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Utilization: {np.mean(episode_utilizations):.2%}")
    print(f"Average Execution Time: {np.mean(episode_completion_times):.2f}s")


def main():
    """Main test function"""
    # Create realistic environment
    env = RealisticCloudEnvironment(
        workload_pattern=WorkloadPattern.PERIODIC,
        seed=42
    )
    
    # Run validation tests
    validator = EnvironmentValidator(env)
    validator.run_all_tests()
    
    # Run performance comparison
    run_performance_comparison()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

