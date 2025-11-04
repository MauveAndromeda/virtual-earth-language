"""
Human Civilization and Society System
======================================

Manages countries, cities, populations, economies, and cultures.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Import data
sys.path.append(str(Path(__file__).parent.parent))
from data.world_data import WORLD_COUNTRIES, WORLD_CITIES


@dataclass
class CountryInfo:
    """Country information."""
    name: str
    population: int
    gdp_usd: float
    capital: str
    government_type: str


@dataclass
class EconomicData:
    """Economic data for a region."""
    gdp_usd: float
    gdp_per_capita_usd: float
    unemployment_rate: float
    inflation_rate: float


class CivilizationSystem:
    """
    Human civilization management system.

    Manages:
    - Countries and governments
    - Cities and urban populations
    - Economies and trade
    - Cultures and languages
    """

    def __init__(self, config, time_system):
        """Initialize civilization system."""
        self.config = config
        self.time = time_system

        # Load country and city data
        self.countries = WORLD_COUNTRIES
        self.cities = WORLD_CITIES

        print(f"   🏛️  Civilization system initialized")
        print(f"      {len(self.countries)} countries loaded")

    def get_country(self, country_name: str) -> Optional[Dict[str, Any]]:
        """Get country information."""
        country = self.countries.get(country_name)
        if not country:
            return None

        # Calculate economic metrics
        population = country['population']
        gdp = country['gdp_usd']
        gdp_per_capita = gdp / population if population > 0 else 0

        return {
            'name': country_name,
            'official_name': country.get('official_name', country_name),
            'capital': country.get('capital'),
            'population': population,
            'area_km2': country.get('area_km2'),
            'gdp_usd': gdp,
            'gdp_per_capita_usd': gdp_per_capita,
            'currency': country.get('currency'),
            'languages': country.get('languages', []),
            'government_type': country.get('government_type'),
            'continent': country.get('continent'),
            'independence_year': country.get('independence_year')
        }

    def get_local_info(self, country_name: str, coord) -> Dict[str, Any]:
        """Get local society information for a coordinate."""
        country = self.get_country(country_name)

        if not country:
            return {}

        return {
            'country': country_name,
            'population_density': country['population'] / country['area_km2'],
            'development_level': 'high' if country['gdp_per_capita_usd'] > 20000 else 'medium',
            'languages': country['languages'],
            'currency': country['currency']
        }

    def update(self, delta_seconds: float) -> Dict[str, Any]:
        """Update civilization system."""
        # Minimal updates - populations and economies change slowly
        return {
            'events': [],
            'global_population': sum(c['population'] for c in self.countries.values())
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global civilization statistics."""
        total_population = sum(c['population'] for c in self.countries.values())
        total_gdp = sum(c['gdp_usd'] for c in self.countries.values())

        return {
            'total_countries': len(self.countries),
            'total_population': total_population,
            'total_gdp_usd': total_gdp,
            'avg_gdp_per_capita': total_gdp / total_population if total_population > 0 else 0,
            'total_cities': len(self.cities)
        }
