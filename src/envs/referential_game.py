"""Referential game environment for emergent communication.

Ubuntu-optimized with multiprocessing support and CUDA compatibility.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class ReferentialGame(gym.Env):
    """
    Referential game where agents must communicate about objects
    with multiple attributes (color, shape, size, position).
    
    Ubuntu-optimized with parallel processing support.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        attributes: Dict[str, int],
        max_episode_steps: int = 100,
        geography: Optional['GeographyModule'] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = 'auto'
    ):
        super().__init__()
        
        self.attributes = attributes
        self.max_episode_steps = max_episode_steps
        self.geography = geography
        self.current_step = 0
        
        # Auto-detect device on Ubuntu
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Multiprocessing setup for Ubuntu
        self.num_workers = min(4, mp.cpu_count())
        
        if seed is not None:
            self.seed(seed)
        
        # Create attribute spaces
        self.attribute_dims = list(attributes.values())
        self.total_objects = np.prod(self.attribute_dims)
        
        # Observation space: one-hot encoding of target object
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.total_objects,), dtype=np.float32
        )
        
        # Action space: discrete choice among all objects
        self.action_space = spaces.Discrete(self.total_objects)
        
        self.reset()
        print(f"🎮 ReferentialGame initialized on {self.device}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seeds for reproducibility on Ubuntu."""
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        return [seed]
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and return initial observation."""
        self.current_step = 0
        
        # Sample random target object
        self.target_attributes = {}
        for attr_name, attr_size in self.attributes.items():
            self.target_attributes[attr_name] = np.random.randint(attr_size)
        
        # Convert to flat index
        self.target_index = self._attributes_to_index(self.target_attributes)
        
        # Create one-hot observation
        observation = np.zeros(self.total_objects, dtype=np.float32)
        observation[self.target_index] = 1.0
        
        info = {
            'target_attributes': self.target_attributes,
            'target_index': self.target_index,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (observation, reward, terminated, truncated, info)."""
        self.current_step += 1
        
        # Calculate reward
        reward = 1.0 if action == self.target_index else 0.0
        
        # Check if done
        terminated = reward > 0
        truncated = self.current_step >= self.max_episode_steps
        
        # Next observation (new target)
        if not (terminated or truncated):
            next_obs, _ = self.reset()
            observation = next_obs
        else:
            observation = np.zeros(self.total_objects, dtype=np.float32)
        
        info = {
            'target_attributes': self.target_attributes,
            'target_index': self.target_index,
            'success': reward > 0,
            'episode_step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _attributes_to_index(self, attributes: Dict[str, int]) -> int:
        """Convert attribute dictionary to flat index."""
        index = 0
        multiplier = 1
        
        for attr_name in reversed(list(self.attributes.keys())):
            index += attributes[attr_name] * multiplier
            multiplier *= self.attributes[attr_name]
        
        return index
    
    def _index_to_attributes(self, index: int) -> Dict[str, int]:
        """Convert flat index to attribute dictionary."""
        attributes = {}
        remaining = index
        
        for attr_name in reversed(list(self.attributes.keys())):
            attr_size = self.attributes[attr_name]
            attributes[attr_name] = remaining % attr_size
            remaining //= attr_size
        
        return attributes
    
    def get_semantic_distance(self, obj1: int, obj2: int) -> float:
        """Calculate semantic distance between two objects."""
        attr1 = self._index_to_attributes(obj1)
        attr2 = self._index_to_attributes(obj2)
        
        distance = 0
        for attr_name in self.attributes:
            if attr1[attr_name] != attr2[attr_name]:
                distance += 1
        
        return distance / len(self.attributes)
    
    def render(self, mode='human'):
        """Render environment (placeholder for Ubuntu display)."""
        if mode == 'human':
            print(f"Target: {self.target_attributes} (index: {self.target_index})")
        elif mode == 'rgb_array':
            # Return placeholder RGB array
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
