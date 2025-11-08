# 🚀 Launch Training Guide

## ✅ Pre-Flight Verification Complete

All systems checked and verified. Ready for parallel training!

### System Status
- **GPU**: NVIDIA RTX 3050 (4GB) ✅
- **Disk Space**: 13GB available ✅  
- **Virtual Env**: Active ✅
- **All Agents**: Verified ✅
- **Checkpoints**: Configured (every 50 episodes) ✅
- **Baseline**: SJF @ 32.4% acceptance (target to beat) ✅

---

## 🎯 Training Configuration

**Training Plan:**
- **Algorithms**: DDQN, PPO, A3C, DDPG (parallel)
- **Episodes**: 1000 per algorithm
- **Duration**: 8-12 hours with GPU
- **Checkpoints**: Every 50 episodes (20 saves per algorithm)
- **Evaluation**: Every 50 episodes (10 test episodes)

**Targets to Beat:**
- Acceptance Rate: 32.4% → **45-55%** (+40-70%)
- Avg Reward: -175.28 → **-50 to +50** (2-5x better)
- Utilization: 51.2% → **55-65%** (+5-15%)

---

## 🚀 START TRAINING

```bash
./scripts/train_all_parallel.sh
```

This will:
1. Create all necessary directories
2. Start 4 training processes in parallel
3. Display process IDs for monitoring
4. Run in background (safe to close terminal)

---

## 📊 MONITORING

### Watch Training Progress
```bash
# All logs (will be busy!)
tail -f logs/*.log

# Specific algorithm
tail -f logs/ddqn_train.log
tail -f logs/ppo_train.log
tail -f logs/a3c_train.log
tail -f logs/ddpg_train.log
```

### TensorBoard (RECOMMENDED)
```bash
tensorboard --logdir=results/logs --port=6006 --bind_all
```
Then open: http://localhost:6006

### Check Processes
```bash
# See if training is running
ps aux | grep train_

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor disk usage
watch -n 60 du -sh results/checkpoints/*
```

---

## 🛑 STOP TRAINING (if needed)

Training runs in background. To stop:

```bash
# The script will show PIDs like:
# DDQN: 12345
# PPO:  12346
# A3C:  12347
# DDPG: 12348

# Kill specific algorithm
kill 12345  # Replace with actual PID

# Kill all (emergency stop)
pkill -f train_ddqn
pkill -f train_ppo
pkill -f train_a3c
pkill -f train_ddpg

# Or find PIDs and kill
ps aux | grep "train_" | grep -v grep | awk '{print $2}' | xargs kill
```

---

## 📁 OUTPUT FILES

### Checkpoints (can resume from these)
```
results/checkpoints/ddqn/
  ├── checkpoint_ep50.pt
  ├── checkpoint_ep100.pt
  ├── ...
  ├── checkpoint_ep1000.pt
  ├── best_model_epXXX.pt  (best performance)
  └── final_model.pt       (at completion)
```

### TensorBoard Logs (for visualization)
```
results/logs/
  ├── ddqn/    (TensorBoard events)
  ├── ppo/
  ├── a3c/
  └── ddpg/
```

### Training Logs (text output)
```
logs/
  ├── ddqn_train.log
  ├── ppo_train.log
  ├── a3c_train.log
  └── ddpg_train.log
```

---

## 🔄 RESUME TRAINING (if interrupted)

If training stops (power loss, error, etc.):

```bash
# Resume from last checkpoint (automatically finds latest)
python scripts/train_ddqn.py --episodes 1000 --resume results/checkpoints/ddqn/checkpoint_ep500.pt

# Or manually for all algorithms
./scripts/train_all_parallel.sh  # Will resume if checkpoints exist
```

---

## ✅ VERIFY COMPLETION

Training is complete when you see in logs:

```
Training completed!
Final evaluation: ...
Model saved to: results/checkpoints/.../final_model.pt
```

Check all 4 algorithms:
```bash
ls -lh results/checkpoints/*/final_model.pt
```

Should see 4 final models:
- results/checkpoints/ddqn/final_model.pt
- results/checkpoints/ppo/final_model.pt
- results/checkpoints/a3c/final_model.pt
- results/checkpoints/ddpg/final_model.pt

---

## 🎉 AFTER TRAINING

Once complete:

1. **View Results in TensorBoard**
   - Compare learning curves
   - Analyze acceptance rates
   - Check reward progression

2. **Evaluate Best Models**
   ```bash
   python scripts/evaluate_best_models.py
   ```

3. **Generate Comparison Report**
   ```bash
   python scripts/analyze_results.py
   ```

4. **Commit to Git**
   ```bash
   git add results/benchmarks results/checkpoints
   git commit -m "Complete RL training - 1000 episodes x 4 algorithms"
   ```

---

## ⚠️ TROUBLESHOOTING

### GPU Out of Memory
- Training will auto-fallback to CPU (slower but works)
- Or reduce batch sizes in config files

### Process Dies
- Check logs for errors: `tail -100 logs/algorithm_train.log`
- Resume from last checkpoint

### Disk Full
- Training will fail if disk fills up
- Need ~2GB total space
- Current free space: 13GB ✅

---

## 📞 QUICK REFERENCE

| Command | Purpose |
|---------|---------|
| `./scripts/train_all_parallel.sh` | Start training |
| `tail -f logs/*.log` | Watch progress |
| `tensorboard --logdir=results/logs` | Visualize metrics |
| `ps aux \| grep train_` | Check running |
| `kill <PID>` | Stop training |
| `nvidia-smi` | Check GPU |

---

**Ready to launch?** 🚀

Run: `./scripts/train_all_parallel.sh`

See you in 8-12 hours with trained models! 🎉
