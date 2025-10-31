# 🚀 Start Training - Quick Guide

## ✅ All 4 Algorithms Ready!

**DDQN** ✅ | **PPO** ✅ | **A3C** ✅ | **DDPG** ✅

---

## 🎯 Choose Your Training Mode

### **Option 1: Parallel Training** ⭐ FASTEST (5-6 hours)

```bash
cd "/home/nightfury653/Documents/BTP Project/Cloud-resource-allocation-using-Reinforcement-Learning"
./scripts/train_all_parallel.sh
```

**What it does**:
- Trains all 4 algorithms simultaneously
- Runs in background
- Saves logs to `logs/` directory
- Completes in ~5-6 hours

**Requirements**: ~8-12 GB GPU RAM

---

### **Option 2: Sequential Training** (16-20 hours)

```bash
cd "/home/nightfury653/Documents/BTP Project/Cloud-resource-allocation-using-Reinforcement-Learning"
./scripts/train_all.sh
```

**What it does**:
- Trains one algorithm at a time
- Lower GPU memory usage (~2-3 GB)
- Completes in ~16-20 hours

---

### **Option 3: Manual Selection** (Custom)

Train specific algorithms:

```bash
cd "/home/nightfury653/Documents/BTP Project/Cloud-resource-allocation-using-Reinforcement-Learning"
source venv/bin/activate

# DDQN (resume from episode 50)
python scripts/train_ddqn.py --episodes 1000 \
    --resume results/checkpoints/ddqn/checkpoint_ep50.pt

# PPO (fresh start)
python scripts/train_ppo.py --episodes 1000

# A3C (fresh start)
python scripts/train_a3c.py --episodes 1000

# DDPG (fresh start)
python scripts/train_ddpg.py --episodes 1000
```

---

## 📊 Monitoring Training

### **View Progress**:
```bash
# Watch log files
tail -f logs/ddqn_train.log
tail -f logs/ppo_train.log
tail -f logs/a3c_train.log
tail -f logs/ddpg_train.log

# Or all at once
tail -f logs/*.log
```

### **TensorBoard** (Real-time Visualization):
```bash
# In separate terminal
tensorboard --logdir results/logs

# Open browser: http://localhost:6006
```

### **Check GPU Usage**:
```bash
watch -n 1 nvidia-smi
```

---

## ⏱️ Expected Timeline

| Algorithm | Episodes | Time | Status |
|-----------|----------|------|--------|
| DDQN | 50 → 1000 | ~4-5h | Resume |
| PPO | 0 → 1000 | ~4-5h | Fresh |
| A3C | 0 → 1000 | ~4-5h | Fresh |
| DDPG | 0 → 1000 | ~4-5h | Fresh |

**Parallel**: ~5-6 hours total  
**Sequential**: ~16-20 hours total

---

## 📈 Expected Results

### By Tomorrow Morning:

**DDQN**:
- ✅ Should beat baseline (7.33%)
- Target: >10% acceptance
- Confidence: 85%

**PPO**:
- ✅ Likely best performer
- Target: >10-12% acceptance
- Confidence: 90%

**A3C**:
- ✅ Should match baseline
- Target: ~8-10% acceptance
- Confidence: 75%

**DDPG**:
- ⚠️ Experimental (adapted)
- Target: >7% acceptance
- Confidence: 60%

---

## 🛑 Stopping Training

### If you need to stop:

```bash
# Find process IDs
ps aux | grep train_

# Kill specific algorithm
kill <PID>

# Or kill all training
pkill -f "train_"
```

### Resume later:
All algorithms save checkpoints every 50 episodes. To resume:

```bash
python scripts/train_ddqn.py --episodes 1000 \
    --resume results/checkpoints/ddqn/checkpoint_ep500.pt
```

---

## 📊 After Training Completes

### 1. Evaluate All Algorithms:
```bash
python scripts/train_ddqn.py --eval-only --resume results/checkpoints/ddqn/best_model.pt
python scripts/train_ppo.py --eval-only --resume results/checkpoints/ppo/best_model.pt
python scripts/train_a3c.py --eval-only --resume results/checkpoints/a3c/best_model.pt
python scripts/train_ddpg.py --eval-only --resume results/checkpoints/ddpg/best_model.pt
```

### 2. Compare with Baselines:
```bash
python scripts/evaluate_baselines.py --episodes 100 --compare
```

### 3. Analyze Results:
```bash
python scripts/analyze_results.py
tensorboard --logdir results/logs
```

---

## 💡 Tips

### **Optimize GPU Usage**:
If running out of memory, reduce batch sizes in configs:
- DDQN: `batch_size: 64` → `32`
- PPO: `batch_size: 64` → `32`
- DDPG: `batch_size: 64` → `32`

### **Speed Up Training**:
- Use GPU (already enabled)
- Reduce episodes for testing: `--episodes 100`
- Train on faster machine if available

### **Ensure Stability**:
- Close other GPU applications
- Monitor disk space (logs can grow)
- Keep laptop plugged in

---

## 🎯 READY TO START?

### **Recommended Command** (Parallel, Fastest):

```bash
cd "/home/nightfury653/Documents/BTP Project/Cloud-resource-allocation-using-Reinforcement-Learning"
./scripts/train_all_parallel.sh
```

Then go to sleep! By morning, all 4 algorithms will be trained! 😴

---

## 📞 Troubleshooting

### **"Command not found"**:
```bash
chmod +x scripts/train_all_parallel.sh
```

### **"No module named..."**:
```bash
source venv/bin/activate
```

### **"CUDA out of memory"**:
- Train 2 at a time instead of 4
- Or use sequential training

### **Training seems stuck**:
- Check logs: `tail logs/*.log`
- Check GPU: `nvidia-smi`
- May be in warmup phase (first 10K steps)

---

## ✅ Final Checklist

Before starting:
- [ ] Virtual environment activated
- [ ] GPU available (`nvidia-smi`)
- [ ] Enough disk space (~10 GB)
- [ ] Scripts executable (`chmod +x`)
- [ ] Laptop plugged in (if applicable)

---

**Status**: ✅ READY TO TRAIN  
**Next**: Run the parallel training script  
**Then**: Sleep well, check results tomorrow! 😴

🚀 **LET'S GO!**
