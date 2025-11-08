# 📊 Baseline Benchmarks

## Overview
This directory contains baseline algorithm performance benchmarks that serve as targets for RL algorithms.

## Benchmark Configuration
- **Environment:** `env_config_training.yaml`
- **VMs:** 40 (mixed small/medium/large)
- **Episodes:** 20 per algorithm
- **Tasks/Episode:** ~500
- **Date:** October 31, 2025

## Results Summary

### 🏆 Algorithm Rankings

| Rank | Algorithm | Acceptance | Avg Reward | Completion Time | Utilization |
|------|-----------|------------|------------|-----------------|-------------|
| 1st | **SJF** | **32.4%** | **-175.28** | 258.54s | **51.2%** |
| 2nd | **Least-Loaded** | 30.9% | -187.87 | **248.62s** | 49.8% |
| 3rd | **Round-Robin** | 30.4% | -199.99 | 269.47s | 44.0% |
| 4th | Random | 29.0% | -215.72 | 278.50s | 42.0% |
| 5th | FCFS | 28.0% | -235.02 | 306.95s | 49.3% |
| 6th | Best-Fit | 26.4% | -259.95 | 339.39s | 46.8% |

### 📈 Best Performers
- **Highest Acceptance:** SJF (32.4%)
- **Best Reward:** SJF (-175.28)
- **Fastest Completion:** Least-Loaded (248.62s)
- **Highest Utilization:** SJF (51.2%)

### 📊 Baseline Statistics
- **Avg Acceptance:** 29.5% ± 2.0%
- **Avg Reward:** -212.3 ± 28.6
- **Avg Completion:** 283.6s ± 31.0s
- **Avg Utilization:** 47.2% ± 3.3%

## 🎯 RL Training Targets

To demonstrate clear superiority, RL algorithms should achieve:

| Metric | Baseline Best | RL Target | Improvement |
|--------|---------------|-----------|-------------|
| Acceptance Rate | 32.4% | **45-55%** | **+40-70%** |
| Avg Reward | -175.28 | **-50 to +50** | **2-5x** |
| Completion Time | 248.62s | **200-230s** | **-10-20%** |
| Utilization | 51.2% | **55-65%** | **+5-15%** |

## Files

- `baseline_benchmarks_training.json` - Complete benchmark results
- `README.md` - This file

## Usage

Compare RL training results against these benchmarks:

```python
import json

# Load baseline benchmarks
with open('results/benchmarks/baseline_benchmarks_training.json', 'r') as f:
    baselines = json.load(f)

# Get best baseline (SJF)
best_baseline = baselines['SJF']
print(f"Best baseline acceptance: {best_baseline['acceptance_rate']:.2%}")
print(f"Best baseline reward: {best_baseline['avg_reward']:.2f}")

# Compare with your RL results
rl_acceptance = 0.45  # Example
improvement = (rl_acceptance / best_baseline['acceptance_rate'] - 1) * 100
print(f"RL improvement: +{improvement:.1f}%")
```

---

**Baseline to Beat:** SJF with 32.4% acceptance and -175.28 avg reward 🎯
