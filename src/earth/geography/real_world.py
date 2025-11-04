"""
Real World Geography System
===========================

Manages real-world geographic data and provides query interfaces.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Import data
sys.path.append(str(Path(__file__).parent.parent))
from data.world_data import WORLD_COUNTRIES, WORLD_CITIES, CONTINENTS, OCEANS
from ..core.earth import EarthConfig, CoordinateSystem


class RealWorldGeography:
    """
    Real-world geography management system.

    Provides access to:
    - Countries and their properties
    - Cities and urban areas
    - Continents and oceans
    - Terrain and elevation data
    """

    def __init__(self, config: EarthConfig):
        """Initialize geography system."""
        self.config = config
        self.countries = WORLD_COUNTRIES
        self.cities = WORLD_CITIES
        self.continents = CONTINENTS
        self.oceans = OCEANS

        # Create spatial indices for fast lookup
        self._build_spatial_indices()

        print(f"   📍 Loaded {len(self.countries)} countries")
        print(f"   🏙️  Loaded {len(self.cities)} cities")

    def _build_spatial_indices(self) -> None:
        """Build spatial indices for efficient geographic queries."""
        # Build country spatial index (simplified - in production use R-tree)
        self.country_index = {}

        # Build city spatial index
        self.city_coords = []
        self.city_names = []

        for city_name, city_data in self.cities.items():
            self.city_coords.append([city_data['latitude'], city_data['longitude']])
            self.city_names.append(city_name)

        self.city_coords = np.array(self.city_coords)

    def query_point(self, coord: CoordinateSystem) -> Dict[str, Any]:
        """
        Query geographic information for a specific point.

        Returns country, nearest city, continent, terrain type, etc.
        """
        result = {
            'coordinate': coord.to_dict(),
            'country': None,
            'nearest_city': None,
            'continent': None,
            'ocean': None,
            'terrain_type': 'unknown'
        }

        # Find country (simplified - check country bounding boxes)
        country = self._find_country_at_point(coord)
        if country:
            result['country'] = country
            result['continent'] = self.countries[country].get('continent', 'Unknown')

        # Find nearest city
        nearest_city = self._find_nearest_city(coord)
        if nearest_city:
            city_data = self.cities[nearest_city]
            distance = coord.distance_to(
                CoordinateSystem(city_data['latitude'], city_data['longitude'])
            )
            result['nearest_city'] = {
                'name': nearest_city,
                'distance_km': distance,
                **city_data
            }

        # Determine if in ocean
        if not country:
            result['ocean'] = self._find_ocean_at_point(coord)
            result['terrain_type'] = 'water'
        else:
            result['terrain_type'] = 'land'

        return result

    def _find_country_at_point(self, coord: CoordinateSystem) -> Optional[str]:
        """Find which country contains the given point (simplified)."""
        # Simplified implementation - checks against country center coordinates
        # In production, use proper polygon containment checks
        min_distance = float('inf')
        closest_country = None

        for country_name, country_data in self.countries.items():
            country_coord = country_data.get('coordinates')
            if country_coord:
                country_center = CoordinateSystem(
                    latitude=country_coord['latitude'],
                    longitude=country_coord['longitude']
                )
                distance = coord.distance_to(country_center)

                # Approximate country radius based on area
                approx_radius = np.sqrt(country_data['area_km2'] / np.pi)

                if distance < approx_radius and distance < min_distance:
                    min_distance = distance
                    closest_country = country_name

        return closest_country

    def _find_nearest_city(self, coord: CoordinateSystem, max_results: int = 1) -> Optional[str]:
        """Find nearest city to given coordinate."""
        if len(self.city_coords) == 0:
            return None

        # Calculate distances to all cities
        point = np.array([[coord.latitude, coord.longitude]])

        # Simple Euclidean distance (approximate)
        distances = np.sqrt(np.sum((self.city_coords - point)**2, axis=1))

        # Get nearest
        nearest_idx = np.argmin(distances)
        return self.city_names[nearest_idx]

    def _find_ocean_at_point(self, coord: CoordinateSystem) -> Optional[str]:
        """Determine which ocean the point is in (simplified)."""
        # Simplified ocean detection based on latitude/longitude ranges
        lat, lon = coord.latitude, coord.longitude

        # Pacific Ocean
        if (lon > 100 or lon < -70) and -60 < lat < 60:
            return 'Pacific Ocean'
        # Atlantic Ocean
        elif -70 < lon < 20 and -60 < lat < 60:
            return 'Atlantic Ocean'
        # Indian Ocean
        elif 20 < lon < 100 and -60 < lat < 30:
            return 'Indian Ocean'
        # Arctic Ocean
        elif lat > 60:
            return 'Arctic Ocean'
        # Southern Ocean
        elif lat < -60:
            return 'Southern Ocean'

        return None

    def get_country(self, country_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a country."""
        return self.countries.get(country_name)

    def get_city(self, city_name: str, country: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a city."""
        city_data = self.cities.get(city_name)

        if city_data and country:
            # Verify city is in specified country
            if city_data.get('country') != country:
                return None

        return city_data

    def get_continent(self, continent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a continent."""
        return self.continents.get(continent_name)

    def get_ocean(self, ocean_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an ocean."""
        return self.oceans.get(ocean_name)

    def find_cities_in_radius(
        self,
        coord: CoordinateSystem,
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """Find all cities within given radius of coordinate."""
        results = []

        for city_name, city_data in self.cities.items():
            city_coord = CoordinateSystem(
                latitude=city_data['latitude'],
                longitude=city_data['longitude']
            )

            distance = coord.distance_to(city_coord)

            if distance <= radius_km:
                results.append({
                    'name': city_name,
                    'distance_km': distance,
                    **city_data
                })

        # Sort by distance
        results.sort(key=lambda x: x['distance_km'])

        return results

    def get_country_borders(self, country1: str, country2: str) -> bool:
        """Check if two countries share a border (simplified)."""
        # This would require detailed border data
        # Simplified implementation
        data1 = self.countries.get(country1)
        data2 = self.countries.get(country2)

        if not data1 or not data2:
            return False

        # Check if on same continent (rough approximation)
        return data1.get('continent') == data2.get('continent')

    def get_statistics(self) -> Dict[str, Any]:
        """Get geographic statistics."""
        return {
            'countries': {
                'total': len(self.countries),
                'by_continent': self._count_by_continent()
            },
            'cities': {
                'total': len(self.cities),
                'by_country': self._count_cities_by_country()
            },
            'continents': {
                'total': len(self.continents),
                'total_land_area_km2': sum(c['area_km2'] for c in self.continents.values())
            },
            'oceans': {
                'total': len(self.oceans),
                'total_ocean_area_km2': sum(o['area_km2'] for o in self.oceans.values())
            }
        }

    def _count_by_continent(self) -> Dict[str, int]:
        """Count countries by continent."""
        counts = {}
        for country_data in self.countries.values():
            continent = country_data.get('continent', 'Unknown')
            counts[continent] = counts.get(continent, 0) + 1
        return counts

    def _count_cities_by_country(self) -> Dict[str, int]:
        """Count cities by country."""
        counts = {}
        for city_data in self.cities.values():
            country = city_data.get('country', 'Unknown')
            counts[country] = counts.get(country, 0) + 1
        return counts
