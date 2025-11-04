"""City database and management."""

from typing import Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.world_data import WORLD_CITIES


class CityDatabase:
    """Database of world cities."""

    def __init__(self):
        self.cities = WORLD_CITIES

    def get(self, name: str) -> Dict[str, Any]:
        return self.cities.get(name, {})

    def all(self) -> Dict[str, Dict[str, Any]]:
        return self.cities
