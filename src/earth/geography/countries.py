"""Country database and management."""

from typing import Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.world_data import WORLD_COUNTRIES


class CountryDatabase:
    """Database of world countries."""

    def __init__(self):
        self.countries = WORLD_COUNTRIES

    def get(self, name: str) -> Dict[str, Any]:
        return self.countries.get(name, {})

    def all(self) -> Dict[str, Dict[str, Any]]:
        return self.countries
