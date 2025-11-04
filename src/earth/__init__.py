"""
Virtual Earth - Real World Simulation System
============================================

A comprehensive, highly realistic simulation of planet Earth including:
- Real geography (continents, countries, cities, terrain)
- Real climate systems (temperature, precipitation, seasons)
- Real ecology (biomes, species, ecosystems)
- Real human civilization (countries, cities, populations, economies, cultures)
- Real-time systems (day/night cycles, seasons, weather)

This module aims to achieve 99.9999% accuracy in representing the real Earth.
"""

from .core.earth import Earth, EarthConfig
from .geography.real_world import RealWorldGeography
from .climate.climate_system import ClimateSystem
from .ecology.biomes import BiomeSystem
from .society.civilization import CivilizationSystem

__version__ = "2.0.0"
__all__ = [
    'Earth',
    'EarthConfig',
    'RealWorldGeography',
    'ClimateSystem',
    'BiomeSystem',
    'CivilizationSystem'
]
