"""Logging utilities for deepfake detection."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'deepfake_detection',
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_dir: str = './logs'
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Log file name (auto-generated if None)
        log_level: Logging level
        log_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified or auto-generated)
    if log_file is not None or log_dir is not None:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename if not provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'deepfake_detection_{timestamp}.log'
        
        file_path = log_path / log_file
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {file_path}")
    
    return logger


def get_logger(name: str = 'deepfake_detection') -> logging.Logger:
    """
    Get existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with default configuration
    if not logger.handlers:
        setup_logger(name)
    
    return logger


class ExperimentLogger:
    """Logger for tracking experiment results and parameters."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = './logs/experiments'
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            name=f'experiment_{experiment_name}',
            log_file=f'{experiment_name}.log',
            log_dir=str(self.experiment_dir)
        )
        
        self.logger.info(f"Experiment '{experiment_name}' started")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def log_parameters(self, params: dict):
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.logger.info("Experiment Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save parameters to file
        import json
        params_file = self.experiment_dir / 'parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        self.logger.info(f"Parameters saved to {params_file}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log experiment metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        step_str = f"Step {step} - " if step is not None else ""
        self.logger.info(f"{step_str}Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = 'file'):
        """
        Log artifact (file, model, etc.).
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact
        """
        self.logger.info(f"Artifact saved: {artifact_path} (type: {artifact_type})")
    
    def finalize(self):
        """Finalize experiment logging."""
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Results saved in: {self.experiment_dir}")


def log_system_info():
    """Log system and environment information."""
    import platform
    import torch
    
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("System Information")
    logger.info("="*60)
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info("="*60)
