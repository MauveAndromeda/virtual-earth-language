"""Terrain and elevation system."""

from typing import Dict, Any
import numpy as np


class TerrainSystem:
    """Terrain and elevation management."""

    def __init__(self):
        self.terrain_types = {
            'ocean': {'elevation_range': (-11000, -100)},
            'coastal': {'elevation_range': (-100, 50)},
            'plains': {'elevation_range': (0, 500)},
            'hills': {'elevation_range': (500, 1500)},
            'mountains': {'elevation_range': (1500, 8900)}
        }

    def get_terrain_type(self, elevation_m: float) -> str:
        """Determine terrain type from elevation."""
        for terrain, data in self.terrain_types.items():
            min_elev, max_elev = data['elevation_range']
            if min_elev <= elevation_m < max_elev:
                return terrain
        return 'unknown'

    def get_elevation(self, coord) -> float:
        """Get elevation at coordinate (simplified)."""
        # Simplified elevation model
        lat = abs(coord.latitude)

        if lat < 5:  # Near equator - varies
            return np.random.uniform(-100, 2000)
        elif lat > 60:  # Polar - generally lower
            return np.random.uniform(0, 500)
        else:  # Mid-latitudes
            return np.random.uniform(0, 3000)
