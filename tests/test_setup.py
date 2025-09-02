"""Test basic setup on Ubuntu."""

import pytest
import torch
import numpy as np
from src.envs import ReferentialGame

def test_torch_cuda():
    """Test PyTorch CUDA setup."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")

def test_environment():
    """Test environment creation."""
    env = ReferentialGame(
        attributes={'color': 4, 'shape': 3},
        max_episode_steps=10
    )
    
    obs, info = env.reset()
    assert obs.shape == (12,)
    assert np.sum(obs) == 1.0
    
    action = np.random.randint(12)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert reward in [0.0, 1.0]
    assert 'success' in info
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

if __name__ == "__main__":
    test_torch_cuda()
    test_environment()
    print("✅ All tests passed!")
