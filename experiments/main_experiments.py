"""Main experiments for emergent language evolution research.

Ubuntu-optimized with GPU acceleration and multiprocessing.
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
from pathlib import Path
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device(cfg):
    """Setup optimal device for Ubuntu."""
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            # Optimize CUDA settings for Ubuntu
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU")
            # Optimize CPU settings for Ubuntu
            torch.set_num_threads(min(8, mp.cpu_count()))
    else:
        device = torch.device(cfg.training.device)
    
    return device

def print_system_info():
    """Print system information."""
    logger.info("🌍 Virtual Earth Language Evolution")
    logger.info(f"🐧 OS: {os.uname().sysname} {os.uname().release}")
    logger.info(f"🐍 Python: {sys.version}")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"💾 CPU cores: {mp.cpu_count()}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"🚀 GPU: {gpu_props.name}")
        logger.info(f"📊 GPU Memory: {gpu_props.total_memory // 1024**3}GB")

@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """Run main experiments."""
    print_system_info()
    device = setup_device(cfg)
    
    logger.info("⚡ Starting experiments...")
    logger.info(f"📋 Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Import here to avoid issues if modules aren't complete
    try:
        from src.envs import ReferentialGame
        logger.info("✅ Environment modules loaded successfully")
        
        # Create a simple test environment
        env = ReferentialGame(
            attributes=cfg.environment.attributes,
            max_episode_steps=cfg.environment.max_episode_steps
        )
        
        # Test environment
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"🎮 Environment test successful!")
        logger.info(f"🎯 Observation shape: {obs.shape}")
        logger.info(f"🎲 Action space: {env.action_space}")
        logger.info(f"🏆 Test reward: {reward}")
        
    except ImportError as e:
        logger.warning(f"⚠️ Could not import all modules: {e}")
        logger.info("📝 This is normal for initial setup")
    
    # Save configuration
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Create a simple results file
    results = {
        'status': 'experiment_setup_complete',
        'device': str(device),
        'config': OmegaConf.to_yaml(cfg)
    }
    
    results_file = output_dir / "experiment_results.yaml"
    with open(results_file, 'w') as f:
        f.write(OmegaConf.to_yaml(results))
    
    logger.info("✅ Experiment setup completed successfully!")
    logger.info(f"📊 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
