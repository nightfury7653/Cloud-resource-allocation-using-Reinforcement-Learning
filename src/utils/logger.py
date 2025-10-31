import logging
from pathlib import Path
from typing import Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import yaml
from datetime import datetime

class Logger:
    """
    Unified logging class for the project.
    Handles both TensorBoard logging and traditional file logging.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        config: Optional[dict] = None
    ):
        # Create log directory
        self.log_dir = Path(log_dir)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logging()
        
        # Setup TensorBoard
        self.writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))
        
        # Save config if provided
        if config is not None:
            self.save_config(config)
        
        self.episode = 0
        self.step = 0
        
    def setup_file_logging(self):
        """Setup traditional file logging"""
        log_file = self.experiment_dir / "training.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def save_config(self, config: dict):
        """Save configuration to YAML file"""
        config_file = self.experiment_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics to both TensorBoard and file
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (if not provided, uses internal counter)
        """
        if step is None:
            step = self.step
            
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
        
    def log_episode_metrics(self, metrics: dict, episode: Optional[int] = None):
        """
        Log episode-level metrics
        
        Args:
            metrics: Dictionary of metric names and values
            episode: Optional episode number (if not provided, uses internal counter)
        """
        if episode is None:
            episode = self.episode
            
        # Add episode/ prefix to metric names for TensorBoard organization
        tb_metrics = {f"episode/{k}": v for k, v in metrics.items()}
        
        # Log to TensorBoard
        for name, value in tb_metrics.items():
            self.writer.add_scalar(name, value, episode)
        
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Episode {episode} | {metric_str}")
        
        self.episode += 1
        
    def log_model_graph(self, model: torch.nn.Module, input_size: tuple):
        """Log model architecture to TensorBoard"""
        dummy_input = torch.zeros(input_size)
        self.writer.add_graph(model, dummy_input)
        
    def log_hyperparameters(self, hparams: dict):
        """Log hyperparameters"""
        # Save to file
        hparam_file = self.experiment_dir / "hyperparameters.json"
        with open(hparam_file, 'w') as f:
            json.dump(hparams, f, indent=4)
            
        # Log to TensorBoard
        self.writer.add_hparams(hparams, {})
        
    def log_info(self, message: str):
        """Log information message"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
        
    def increment_step(self):
        """Increment internal step counter"""
        self.step += 1
        
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
