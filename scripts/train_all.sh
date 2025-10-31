#!/bin/bash
# Train all RL algorithms overnight

echo "=========================================="
echo "Training All RL Algorithms"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Create results directories
mkdir -p results/checkpoints/{ddqn,ppo,a3c,ddpg}
mkdir -p results/logs/{ddqn,ppo,a3c,ddpg}

# Function to train algorithm
train_algorithm() {
    local algo=$1
    local episodes=$2
    local resume_flag=$3
    
    echo ""
    echo "=========================================="
    echo "Training $algo for $episodes episodes"
    echo "=========================================="
    
    if [ "$resume_flag" = "resume" ]; then
        # Find latest checkpoint
        latest=$(ls -t results/checkpoints/$algo/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "Resuming from: $latest"
            python3 scripts/train_$algo.py --episodes $episodes --resume "$latest" 2>&1 | tee results/logs/${algo}_training.log
        else
            echo "No checkpoint found, starting fresh"
            python3 scripts/train_$algo.py --episodes $episodes 2>&1 | tee results/logs/${algo}_training.log
        fi
    else
        python3 scripts/train_$algo.py --episodes $episodes 2>&1 | tee results/logs/${algo}_training.log
    fi
    
    echo "✓ $algo training complete"
}

# Training configuration
EPISODES=1000  # Episodes per algorithm

# Train each algorithm sequentially
# (To run in parallel, use & at end and add 'wait' command)

echo "Starting training pipeline..."
echo "Total episodes per algorithm: $EPISODES"
echo "Estimated time: 4-6 hours per algorithm"
echo ""

# DDQN (resume from 50 episodes)
train_algorithm "ddqn" $EPISODES "resume"

# PPO
train_algorithm "ppo" $EPISODES "new"

# A3C
train_algorithm "a3c" $EPISODES "new"

# DDPG
train_algorithm "ddpg" $EPISODES "new"

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo ""
echo "View results with TensorBoard:"
echo "  tensorboard --logdir results/logs"
echo ""
echo "Compare algorithms:"
echo "  python scripts/compare_algorithms.py"
echo ""

