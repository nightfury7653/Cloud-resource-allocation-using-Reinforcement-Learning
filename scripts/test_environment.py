import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.environment.cloud_env import CloudEnvironment
from src.utils.logger import Logger
import numpy as np

def test_environment(num_episodes: int = 5, max_steps: int = 100):
    """
    Test the cloud environment with random actions
    """
    # Initialize environment and logger
    env = CloudEnvironment()
    logger = Logger(experiment_name="env_test")
    
    logger.log_info("Starting environment test...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        logger.log_info(f"\nEpisode {episode + 1}")
        logger.log_info(f"Initial state: {state}")
        
        done = False
        while not done and episode_steps < max_steps:
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Log step information
            logger.log_info(
                f"\nStep {episode_steps + 1}:\n"
                f"Action: {action}\n"
                f"Reward: {reward:.4f}\n"
                f"Next state: {next_state}\n"
                f"Info: {info}"
            )
            
            episode_reward += reward
            episode_steps += 1
            
            state = next_state
            
        # Log episode summary
        logger.log_info(
            f"\nEpisode {episode + 1} finished:\n"
            f"Total reward: {episode_reward:.4f}\n"
            f"Steps: {episode_steps}"
        )
        
        # Log metrics
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_steps,
            "avg_reward": episode_reward / episode_steps,
            "final_queue_length": len(env.task_queue),
            "completed_tasks": len(env.completed_tasks)
        }
        logger.log_episode_metrics(metrics, episode)
        
    logger.log_info("\nEnvironment test completed!")
    logger.close()

if __name__ == "__main__":
    test_environment()
