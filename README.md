# Cloud Resource Allocation using Reinforcement Learning

## 🎯 Project Overview

This project implements a **comparative study of RL vs traditional scheduling algorithms** for cloud resource allocation. We compare **4 RL algorithms (DDQN, PPO, A3C, DDPG)** against **6 traditional baselines** on a **realistic cloud simulation environment**.

### Research Goals

1. Compare state-of-the-art RL algorithms for cloud resource allocation
2. Benchmark against traditional scheduling strategies
3. Provide comprehensive performance analysis across multiple metrics
4. Demonstrate advantages and trade-offs of RL approaches

### Current Status: **Phase 2 Complete** ✅

- ✅ **Phase 1**: Realistic Cloud Environment (COMPLETE)
- ✅ **Phase 2**: Baseline Algorithms (COMPLETE) - 6 algorithms implemented & benchmarked
- ⏳ **Phase 3**: DDQN Implementation (NEXT)
- ⏳ Phases 4-8: PPO, A3C, DDPG, Comparison, Visualization

### Key Features

✨ **Realistic Cloud Simulation**
- Performance degradation modeling (CPU throttling, memory swapping)
- Resource contention and task interference
- Dynamic execution times based on VM load
- Multiple task types with different resource profiles

📊 **Multiple Workload Patterns**
- Periodic (daily patterns)
- Bursty (sudden spikes)
- Constant (steady state)
- Trending (increasing/decreasing load)
- Real trace support (Google, Alibaba)

🧠 **RL Algorithms**
- DDQN (Double Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

📊 **Baseline Algorithms** ✅ **IMPLEMENTED**
- Random (naive baseline)
- Round-Robin (cyclic assignment)
- FCFS (First-Come-First-Serve)
- SJF (Shortest Job First)
- Best Fit (bin-packing inspired)
- Least Loaded (load balancing) - **Best performer: 7.33% acceptance**

---

## 📁 Project Structure

```
Cloud-resource-allocation-using-Reinforcement-Learning/
├── config/                          # Configuration files
│   ├── env_config.yaml             # Environment parameters
│   ├── agent_config.yaml           # Agent hyperparameters
│   └── training_config.yaml        # Training settings
│
├── src/                            # Source code
│   ├── environment/                # Environment simulation ✅
│   │   ├── cloud_env.py           # Basic cloud environment
│   │   ├── realistic_cloud_env.py # Realistic simulation
│   │   ├── task_models.py         # Task and VM models
│   │   ├── workload_generator.py  # Workload patterns
│   │   └── performance_models.py  # Performance modeling
│   │
│   ├── baselines/                  # Baseline schedulers ✅
│   │   ├── base_scheduler.py      # Base class & metrics
│   │   ├── random_scheduler.py    # Random selection
│   │   ├── round_robin_scheduler.py # Round-robin
│   │   ├── fcfs_scheduler.py      # FCFS
│   │   ├── sjf_scheduler.py       # Shortest Job First
│   │   ├── best_fit_scheduler.py  # Best fit
│   │   └── least_loaded_scheduler.py # Least loaded
│   │
│   ├── agent/                      # RL agents (TODO)
│   │   ├── ddqn_agent.py          # DDQN agent
│   │   ├── ppo_agent.py           # PPO agent
│   │   ├── a3c_agent.py           # A3C agent
│   │   └── ddpg_agent.py          # DDPG agent
│   │
│   ├── networks/                   # Neural networks
│   │   └── dueling_network.py     # Dueling architecture (TODO)
│   │
│   ├── training/                   # Training pipeline
│   │   └── trainer.py             # Training loop (TODO)
│   │
│   └── utils/                      # Utilities
│       ├── logger.py              # Logging
│       ├── metrics.py             # Performance metrics
│       └── config_loader.py       # Config loading
│
├── scripts/                        # Executable scripts
│   ├── test_environment.py        # Basic environment test ✅
│   ├── test_realistic_environment.py  # Validation suite ✅
│   └── evaluate_baselines.py      # Baseline evaluation ✅
│
├── Src/                           # Research papers and presentations
│
├── docs/                          # 📚 All documentation (organized by category)
│   ├── planning/                  # Project plans, roadmaps, quick reference
│   ├── phases/                    # Phase 1-3 completion summaries
│   ├── algorithms/                # Algorithm status & training guides  
│   ├── issues/                    # Bug fixes & configuration docs
│   ├── sessions/                  # Session summaries & policy
│   └── README.md                  # Documentation index
│
└── README.md                      # This file (you are here)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) GPU for faster training

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Cloud-resource-allocation-using-Reinforcement-Learning
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Quick Start

**Test the realistic environment**:
```bash
python3 scripts/test_realistic_environment.py
```

Expected output:
```
============================================================
REALISTIC CLOUD ENVIRONMENT VALIDATION
============================================================

1. Testing Basic Functionality...
  ✓ Reset works correctly
  ✓ Step works correctly

2. Testing Workload Patterns...
  ✓ Workload pattern shows expected variation

3. Testing Resource Contention...
  ✓ Resource contention increases execution time

4. Testing Performance Degradation...
  ✓ Performance degrades at high utilization

5. Testing Task Diversity...
  ✓ Task diversity is present

6. Testing Reward Function...
  ✓ Reward function differentiates success/failure

Total: 6/6 tests passed
🎉 All tests passed! Environment is ready for training.
```

**Evaluate baseline algorithms** ✅:
```bash
# Evaluate all 6 baseline algorithms
python3 scripts/evaluate_baselines.py --episodes 100 --compare

# Evaluate specific algorithm
python3 scripts/evaluate_baselines.py --algo least-loaded --episodes 50

# Quick test
python3 scripts/evaluate_baselines.py --algo random --episodes 10
```

Expected output:
```
====================================================================================================
PERFORMANCE COMPARISON TABLE
====================================================================================================
Algorithm       | Accept%  | AvgComp  | AvgWait  | Util%    | SLA%     | Reward  
----------------------------------------------------------------------------------------------------
Least-Loaded    |     7.3% |     0.00 |     0.00 |    45.8% |     0.0% |  -425.84
SJF             |     6.8% |     0.00 |     0.00 |    44.7% |     0.0% |  -431.43
Random          |     6.7% |     0.00 |     0.00 |    43.9% |     0.0% |  -434.53
...
====================================================================================================
```

---

## 🎮 Usage

### Using the Realistic Environment

```python
from src.environment.realistic_cloud_env import RealisticCloudEnvironment
from src.environment.workload_generator import WorkloadPattern

# Create environment
env = RealisticCloudEnvironment(
    config_path="config/env_config.yaml",
    workload_pattern=WorkloadPattern.PERIODIC,
    seed=42
)

# Reset environment
state, info = env.reset()

# Run simulation
done = False
total_reward = 0

while not done:
    # Sample random action (replace with RL agent)
    action = env.action_space.sample()
    
    # Step environment
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Print metrics
    if info['allocation_info']['allocated']:
        print(f"✓ Task allocated to VM {action}")
    else:
        print(f"✗ Allocation failed: {info['allocation_info']['reason']}")

print(f"Episode finished with total reward: {total_reward:.2f}")
print(f"Completed tasks: {info['completed_tasks']}")
print(f"Average utilization: {info['avg_utilization']:.2%}")
```

### Configuration

Edit `config/env_config.yaml`:

```yaml
vm_pool:
  num_vms: 10
  vm_types:
    - {type: "small", cpu_cores: 2, memory_gb: 4}
    - {type: "medium", cpu_cores: 4, memory_gb: 8}
    - {type: "large", cpu_cores: 8, memory_gb: 16}

task:
  arrival_rate: 5

environment:
  max_queue_size: 100
  episode_length: 500
  time_step: 1.0

reward:
  completion_time_weight: 1.0
  utilization_weight: 0.5
  training_cost_weight: 0.3
  acceptance_rate_weight: 0.7
```

---

## 📊 Simulation Features

### Task Types

| Type | CPU Intensity | Memory Intensity | I/O Intensity | Characteristics |
|------|---------------|------------------|---------------|-----------------|
| **CPU-intensive** | 90% | 30% | 10% | High CPU, low I/O |
| **Memory-intensive** | 30% | 90% | 20% | Large memory footprint |
| **I/O-intensive** | 20% | 30% | 90% | Heavy I/O operations |
| **Mixed** | 50% | 50% | 50% | Balanced workload |
| **Batch** | 70% | 40% | 30% | Long-running jobs |
| **Web Service** | 40% | 50% | 60% | Bursty, interactive |

### Workload Patterns

1. **PERIODIC**: Mimics daily usage patterns
   - Peak hours: 9 AM - 5 PM (1.5x base rate)
   - Night hours: 11 PM - 5 AM (0.5x base rate)
   - Random variation: ±20%

2. **BURSTY**: Random spikes
   - 10% chance of burst (3-5x base rate)
   - 90% normal load (0.7x base rate)

3. **CONSTANT**: Steady arrival rate

4. **TRENDING**: Linear increase/decrease over time

5. **REAL_TRACE**: Load from real cloud traces

### Performance Modeling

The simulation models:
- **CPU Throttling**: >80% utilization → performance degradation
- **Memory Swapping**: >85% memory → significant slowdown
- **Cache Contention**: Multiple tasks → cache pollution
- **Task Interference**: Co-located tasks affect each other
- **Network Latency**: VM-to-VM communication delays

**Example**: A task that takes 10.18s with 1 task on a VM will take 36.03s with 2 competing tasks!

---

## 🧪 Validation Results

### Test Suite: 6/6 Passed ✅

| Test | Result | Metric |
|------|--------|--------|
| Basic Functionality | ✅ PASS | State/action spaces correct |
| Workload Patterns | ✅ PASS | Variation std: 3.21 |
| Resource Contention | ✅ PASS | 3.5x slowdown with 2 tasks |
| Performance Degradation | ✅ PASS | 30% loss at 95% util |
| Task Diversity | ✅ PASS | 6 task types generated |
| Reward Function | ✅ PASS | Differentiates success/failure |

### Random Policy Performance (Baseline)

```
Average Episode Reward: -62.09 ± 3.85
Average Utilization:    36.63%
Average Execution Time: 61.05s
Completion Rate:        1-7 tasks per 100 steps
```

*Note: Random policy provides baseline for RL improvement*

---

## 🎓 Research Context

### Addressed Research Gaps

1. **Realistic Simulation**: Most cloud RL papers use oversimplified environments
2. **Performance Modeling**: Accounts for contention and degradation
3. **Multi-Objective**: Balances multiple competing objectives
4. **Workload Diversity**: Multiple patterns and task types
5. **Validation Framework**: Comprehensive testing ensures correctness

### Key References

- Google Cluster Data (task distributions, resource requirements)
- Alibaba Cluster Traces (workload patterns)
- Cloud computing research (performance models)

See `Src/` for research papers.

---

## 🔧 Development Roadmap

### Phase 1: Environment Setup ✅ COMPLETE
- [x] Enhanced task and VM models
- [x] Realistic workload generator
- [x] Performance and contention models
- [x] Integrated realistic environment
- [x] Validation framework

### Phase 2: Agent Development (In Progress)
- [ ] DDQN agent implementation
- [ ] Dueling network architecture
- [ ] Experience replay buffer
- [ ] Target network updates

### Phase 3: Training Pipeline
- [ ] Training loop
- [ ] Tensorboard logging
- [ ] Checkpointing
- [ ] Hyperparameter tuning

### Phase 4: Evaluation & Comparison
- [ ] Baseline comparisons (Random, Round-robin, Greedy)
- [ ] Performance metrics visualization
- [ ] Statistical analysis
- [ ] Real trace evaluation

### Phase 5: Deployment (Optional)
- [ ] Real cluster integration (Slurm)
- [ ] AWS EC2 deployment
- [ ] Live monitoring dashboard

---

## 📈 Performance Metrics

The system tracks:
- **Task Completion Time**: Average time from arrival to completion
- **Resource Utilization**: CPU and memory usage across VMs
- **Task Acceptance Ratio**: Successfully allocated tasks / total tasks
- **Queue Length**: Number of waiting tasks
- **Load Balancing**: Standard deviation of VM utilizations
- **Deadline Met Ratio**: Tasks completed before deadline
- **Training Convergence**: Episode rewards over time

---

## 🤝 Contributing

This is a research project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📝 License

[Add your license here]

---

## 📧 Contact

[Add your contact information]

---

## 🙏 Acknowledgments

- Google for cluster trace data
- Alibaba for workload traces
- OpenAI Gymnasium for RL environment framework
- Research papers in `Src/` directory

---

## 📚 Documentation

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**: Detailed 7-phase implementation plan
- **[REALISTIC_SIMULATION_SUMMARY.md](REALISTIC_SIMULATION_SUMMARY.md)**: Complete simulation documentation
- **[CLUSTER_SETUP.md](CLUSTER_SETUP.md)**: HPC cluster setup with AWS and Slurm
- **[FREE_CLUSTER_SETUP.md](FREE_CLUSTER_SETUP.md)**: Free tier cluster setup guide

---

**Status**: Phase 1 Complete ✅ | Ready for RL Agent Development 🚀

*Last Updated: October 30, 2025*
