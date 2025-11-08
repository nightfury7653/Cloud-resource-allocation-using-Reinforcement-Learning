#!/bin/bash
# Train all 4 RL algorithms in parallel

echo "=========================================="
echo "Training All 4 RL Algorithms in Parallel"
echo "=========================================="

# Create necessary directories
mkdir -p logs results/checkpoints/{ddqn,ppo,a3c,ddpg} results/logs/{ddqn,ppo,a3c,ddpg}

# Activate virtual environment
source venv/bin/activate

echo "Starting parallel training..."
echo "Monitor with: tail -f logs/*.log"
echo "Or use TensorBoard: tensorboard --logdir results/logs"
echo ""

# Start all trainings in background
echo "Starting DDQN..."
nohup python scripts/train_ddqn.py --episodes 1000 \
    --env-config config/env_config_training.yaml \
    --eval-frequency 50 --eval-episodes 10 \
    > logs/ddqn_train.log 2>&1 &
DDQN_PID=$!

echo "Starting PPO..."
nohup python scripts/train_ppo.py --episodes 1000 \
    --env-config config/env_config_training.yaml \
    --eval-frequency 50 --eval-episodes 10 \
    > logs/ppo_train.log 2>&1 &
PPO_PID=$!

echo "Starting A3C..."
nohup python scripts/train_a3c.py --episodes 1000 \
    --env-config config/env_config_training.yaml \
    --eval-frequency 50 --eval-episodes 10 \
    > logs/a3c_train.log 2>&1 &
A3C_PID=$!

echo "Starting DDPG..."
nohup python scripts/train_ddpg.py --episodes 1000 \
    --env-config config/env_config_training.yaml \
    --eval-frequency 50 --eval-episodes 10 \
    > logs/ddpg_train.log 2>&1 &
DDPG_PID=$!

echo ""
echo "=========================================="
echo "All algorithms started!"
echo "=========================================="
echo "Process IDs:"
echo "  DDQN: $DDQN_PID"
echo "  PPO:  $PPO_PID"
echo "  A3C:  $A3C_PID"
echo "  DDPG: $DDPG_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/*.log"
echo "  tensorboard --logdir results/logs"
echo ""
echo "Check if running:"
echo "  ps aux | grep train_"
echo ""
echo "Kill all if needed:"
echo "  kill $DDQN_PID $PPO_PID $A3C_PID $DDPG_PID"
echo ""
echo "Expected completion: ~8-12 hours (with GPU)"
echo "Checkpoints saved every 50 episodes"
echo "=========================================="

