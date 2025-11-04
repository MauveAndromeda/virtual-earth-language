"""
Ecology and Biome System
=========================

Manages Earth's ecosystems and biodiversity.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class EcosystemType(Enum):
    """Major ecosystem types."""
    TROPICAL_RAINFOREST = "Tropical Rainforest"
    TEMPERATE_FOREST = "Temperate Forest"
    BOREAL_FOREST = "Boreal Forest (Taiga)"
    GRASSLAND_SAVANNA = "Grassland/Savanna"
    DESERT = "Desert"
    TUNDRA = "Tundra"
    MARINE = "Marine"
    FRESHWATER = "Freshwater"
    WETLAND = "Wetland"


@dataclass
class Biome:
    """Biome definition with characteristics."""
    name: str
    ecosystem_type: EcosystemType
    avg_temperature_c: float
    avg_precipitation_mm: float
    biodiversity_index: float  # 0-1
    primary_vegetation: List[str]
    characteristic_animals: List[str]


class BiomeSystem:
    """Manages Earth's biomes and ecosystems."""

    def __init__(self, config):
        """Initialize biome system."""
        self.config = config
        self._initialize_biomes()
        print("   🌿 Ecology system initialized")

    def _initialize_biomes(self):
        """Initialize biome definitions."""
        self.biomes = {
            'tropical_rainforest': Biome(
                name="Tropical Rainforest",
                ecosystem_type=EcosystemType.TROPICAL_RAINFOREST,
                avg_temperature_c=27.0,
                avg_precipitation_mm=2000,
                biodiversity_index=1.0,
                primary_vegetation=["Broadleaf trees", "Vines", "Epiphytes"],
                characteristic_animals=["Jaguars", "Monkeys", "Parrots", "Insects"]
            ),
            'temperate_forest': Biome(
                name="Temperate Forest",
                ecosystem_type=EcosystemType.TEMPERATE_FOREST,
                avg_temperature_c=12.0,
                avg_precipitation_mm=800,
                biodiversity_index=0.7,
                primary_vegetation=["Deciduous trees", "Conifers", "Ferns"],
                characteristic_animals=["Deer", "Bears", "Birds", "Squirrels"]
            ),
            'grassland': Biome(
                name="Grassland",
                ecosystem_type=EcosystemType.GRASSLAND_SAVANNA,
                avg_temperature_c=20.0,
                avg_precipitation_mm=500,
                biodiversity_index=0.5,
                primary_vegetation=["Grasses", "Scattered trees"],
                characteristic_animals=["Bison", "Lions", "Zebras", "Eagles"]
            ),
            'desert': Biome(
                name="Desert",
                ecosystem_type=EcosystemType.DESERT,
                avg_temperature_c=30.0,
                avg_precipitation_mm=100,
                biodiversity_index=0.2,
                primary_vegetation=["Cacti", "Succulents", "Dry shrubs"],
                characteristic_animals=["Camels", "Lizards", "Scorpions", "Snakes"]
            ),
            'tundra': Biome(
                name="Tundra",
                ecosystem_type=EcosystemType.TUNDRA,
                avg_temperature_c=-5.0,
                avg_precipitation_mm=200,
                biodiversity_index=0.3,
                primary_vegetation=["Mosses", "Lichens", "Small shrubs"],
                characteristic_animals=["Caribou", "Arctic foxes", "Polar bears"]
            )
        }

    def get_biome(self, coord) -> Dict[str, Any]:
        """Determine biome at given coordinate."""
        # Simplified biome determination based on latitude
        lat = abs(coord.latitude)

        if lat < 15:
            biome_key = 'tropical_rainforest'
        elif lat < 30:
            biome_key = 'grassland'
        elif lat < 60:
            biome_key = 'temperate_forest'
        else:
            biome_key = 'tundra'

        biome = self.biomes.get(biome_key)

        return {
            'biome_type': biome.name,
            'ecosystem': biome.ecosystem_type.value,
            'biodiversity_index': biome.biodiversity_index,
            'vegetation': biome.primary_vegetation,
            'animals': biome.characteristic_animals
        }

    def update(self, delta_seconds: float, climate_system) -> Dict[str, Any]:
        """Update ecology system."""
        return {
            'ecosystems_stable': True,
            'biodiversity_trend': 'stable'
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get ecology statistics."""
        return {
            'total_biomes': len(self.biomes),
            'ecosystem_types': len(EcosystemType)
        }
