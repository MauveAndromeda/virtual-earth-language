"""
Core Earth Simulation System
============================

Main Earth class that integrates all subsystems:
- Geography (continents, countries, cities)
- Climate (weather, seasons, temperature)
- Ecology (biomes, species)
- Society (civilizations, economies, cultures)
- Time (day/night cycles, seasons, years)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class EarthConfig:
    """Configuration for Earth simulation."""

    # Physical parameters (real Earth values)
    radius_km: float = 6371.0  # Earth's mean radius
    surface_area_km2: float = 510_072_000.0  # Total surface area
    land_area_km2: float = 148_940_000.0  # Total land area
    ocean_area_km2: float = 361_132_000.0  # Total ocean area

    # Simulation parameters
    simulation_resolution: str = "high"  # "low", "medium", "high", "ultra"
    real_time_mode: bool = False  # Sync with real world time
    starting_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 0))

    # System toggles
    enable_climate: bool = True
    enable_ecology: bool = True
    enable_society: bool = True
    enable_real_time_weather: bool = False  # Fetch real weather data

    # Data sources
    use_real_data: bool = True  # Use real world data vs generated
    data_path: Optional[Path] = None


@dataclass
class CoordinateSystem:
    """Geographic coordinate system for Earth."""

    latitude: float  # -90 to 90 degrees
    longitude: float  # -180 to 180 degrees
    altitude: float = 0.0  # meters above sea level

    def __post_init__(self):
        """Validate coordinates."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")

    def distance_to(self, other: 'CoordinateSystem') -> float:
        """Calculate distance to another coordinate using Haversine formula."""
        R = 6371.0  # Earth radius in km

        lat1, lon1 = np.radians(self.latitude), np.radians(self.longitude)
        lat2, lon2 = np.radians(other.latitude), np.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CoordinateSystem':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TimeSystem:
    """Time system for Earth simulation."""

    current_datetime: datetime
    time_scale: float = 1.0  # 1.0 = real time, > 1.0 = faster

    # Astronomical calculations
    axial_tilt: float = 23.44  # degrees
    orbital_period_days: float = 365.25
    rotation_period_hours: float = 24.0

    def advance(self, seconds: float) -> None:
        """Advance time by given seconds (scaled by time_scale)."""
        actual_seconds = seconds * self.time_scale
        self.current_datetime += timedelta(seconds=actual_seconds)

    def get_day_of_year(self) -> int:
        """Get current day of year (1-365/366)."""
        return self.current_datetime.timetuple().tm_yday

    def get_season(self, latitude: float) -> str:
        """Get current season for given latitude."""
        day = self.get_day_of_year()

        # Northern hemisphere
        if latitude >= 0:
            if 80 <= day < 172:  # Mar 21 - Jun 20
                return "spring"
            elif 172 <= day < 266:  # Jun 21 - Sep 22
                return "summer"
            elif 266 <= day < 356:  # Sep 23 - Dec 20
                return "autumn"
            else:  # Dec 21 - Mar 20
                return "winter"
        # Southern hemisphere (reversed)
        else:
            if 80 <= day < 172:
                return "autumn"
            elif 172 <= day < 266:
                return "winter"
            elif 266 <= day < 356:
                return "spring"
            else:
                return "summer"

    def get_solar_position(self, coord: CoordinateSystem) -> Dict[str, float]:
        """Calculate solar position for given coordinate."""
        # Simplified solar position calculation
        day_of_year = self.get_day_of_year()
        hour = self.current_datetime.hour + self.current_datetime.minute / 60.0

        # Solar declination
        declination = self.axial_tilt * np.sin(2 * np.pi * (day_of_year - 81) / 365.25)

        # Hour angle
        hour_angle = 15 * (hour - 12)  # degrees

        # Solar elevation angle
        lat_rad = np.radians(coord.latitude)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        elevation = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        ))

        # Azimuth (simplified)
        azimuth = np.degrees(np.arctan2(
            np.sin(ha_rad),
            np.cos(ha_rad) * np.sin(lat_rad) - np.tan(dec_rad) * np.cos(lat_rad)
        ))

        return {
            'elevation': elevation,
            'azimuth': azimuth,
            'is_daytime': elevation > 0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_datetime': self.current_datetime.isoformat(),
            'time_scale': self.time_scale,
            'day_of_year': self.get_day_of_year()
        }


class Earth:
    """
    Main Earth simulation class.

    Integrates all subsystems to create a comprehensive, realistic Earth simulation.
    """

    def __init__(self, config: Optional[EarthConfig] = None):
        """Initialize Earth simulation."""
        self.config = config or EarthConfig()

        # Initialize time system
        self.time = TimeSystem(current_datetime=self.config.starting_date)

        # Initialize subsystems (will be loaded lazily)
        self._geography = None
        self._climate = None
        self._ecology = None
        self._society = None

        # Statistics and metrics
        self.stats = {
            'total_countries': 0,
            'total_cities': 0,
            'total_population': 0,
            'land_coverage': {},
            'climate_zones': {}
        }

        # Event log
        self.events = []

        print(f"🌍 Virtual Earth initialized at {self.time.current_datetime}")

    @property
    def geography(self):
        """Get geography system (lazy loading)."""
        if self._geography is None:
            from ..geography.real_world import RealWorldGeography
            self._geography = RealWorldGeography(self.config)
            print("✓ Geography system loaded")
        return self._geography

    @property
    def climate(self):
        """Get climate system (lazy loading)."""
        if self._climate is None and self.config.enable_climate:
            from ..climate.climate_system import ClimateSystem
            self._climate = ClimateSystem(self.config, self.time)
            print("✓ Climate system loaded")
        return self._climate

    @property
    def ecology(self):
        """Get ecology system (lazy loading)."""
        if self._ecology is None and self.config.enable_ecology:
            from ..ecology.biomes import BiomeSystem
            self._ecology = BiomeSystem(self.config)
            print("✓ Ecology system loaded")
        return self._ecology

    @property
    def society(self):
        """Get society system (lazy loading)."""
        if self._society is None and self.config.enable_society:
            from ..society.civilization import CivilizationSystem
            self._society = CivilizationSystem(self.config, self.time)
            print("✓ Society system loaded")
        return self._society

    def initialize(self) -> None:
        """Initialize all systems with real-world data."""
        print("\n🌍 Initializing Virtual Earth...")
        print("=" * 60)

        # Load all systems
        _ = self.geography
        _ = self.climate
        _ = self.ecology
        _ = self.society

        # Calculate statistics
        self.update_statistics()

        print("=" * 60)
        print("✅ Virtual Earth initialization complete!")
        print(f"   📍 {self.stats['total_countries']} countries loaded")
        print(f"   🏙️  {self.stats['total_cities']} cities loaded")
        print(f"   👥 {self.stats['total_population']:,} total population")
        print(f"   🕐 Current time: {self.time.current_datetime}")

    def step(self, delta_seconds: float = 3600.0) -> Dict[str, Any]:
        """
        Advance simulation by given time delta.

        Args:
            delta_seconds: Time to advance in seconds (default 1 hour)

        Returns:
            Dictionary of simulation state changes
        """
        # Advance time
        self.time.advance(delta_seconds)

        results = {
            'time': self.time.to_dict(),
            'events': []
        }

        # Update climate
        if self._climate:
            climate_changes = self._climate.update(delta_seconds)
            results['climate'] = climate_changes

        # Update ecology
        if self._ecology:
            ecology_changes = self._ecology.update(delta_seconds, self._climate)
            results['ecology'] = ecology_changes

        # Update society
        if self._society:
            society_changes = self._society.update(delta_seconds)
            results['society'] = society_changes
            results['events'].extend(society_changes.get('events', []))

        # Log events
        self.events.extend(results['events'])

        return results

    def query_location(self, coord: CoordinateSystem) -> Dict[str, Any]:
        """
        Query information about a specific location.

        Returns comprehensive data including:
        - Geographic info (country, city, terrain)
        - Climate data (temperature, precipitation)
        - Ecological data (biome, species)
        - Society data (population, economy, culture)
        """
        info = {
            'coordinate': coord.to_dict(),
            'time': self.time.to_dict()
        }

        # Geographic information
        if self._geography:
            geo_info = self._geography.query_point(coord)
            info['geography'] = geo_info

        # Climate information
        if self._climate:
            climate_info = self._climate.get_conditions(coord)
            info['climate'] = climate_info

        # Ecological information
        if self._ecology:
            eco_info = self._ecology.get_biome(coord)
            info['ecology'] = eco_info

        # Society information
        if self._society and 'country' in info.get('geography', {}):
            society_info = self._society.get_local_info(
                info['geography']['country'],
                coord
            )
            info['society'] = society_info

        # Solar information
        info['solar'] = self.time.get_solar_position(coord)

        return info

    def get_country(self, country_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a country."""
        if not self._geography or not self._society:
            return None

        country_geo = self._geography.get_country(country_name)
        if not country_geo:
            return None

        country_soc = self._society.get_country(country_name)

        return {
            **country_geo,
            **country_soc,
            'query_time': self.time.current_datetime.isoformat()
        }

    def get_city(self, city_name: str, country: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a city."""
        if not self._geography:
            return None

        city_info = self._geography.get_city(city_name, country)
        if not city_info:
            return None

        # Add current conditions
        coord = CoordinateSystem(
            latitude=city_info['latitude'],
            longitude=city_info['longitude']
        )

        current_conditions = self.query_location(coord)
        city_info['current_conditions'] = current_conditions

        return city_info

    def update_statistics(self) -> None:
        """Update global statistics."""
        if self._geography:
            self.stats['total_countries'] = len(self._geography.countries)
            self.stats['total_cities'] = len(self._geography.cities)

        if self._society:
            self.stats['total_population'] = sum(
                country.get('population', 0)
                for country in self._society.countries.values()
            )

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics and metrics."""
        self.update_statistics()

        stats = {
            'earth_metrics': {
                'surface_area_km2': self.config.surface_area_km2,
                'land_area_km2': self.config.land_area_km2,
                'ocean_area_km2': self.config.ocean_area_km2
            },
            'simulation': {
                'current_time': self.time.current_datetime.isoformat(),
                'time_scale': self.time.time_scale,
                'systems_active': {
                    'geography': self._geography is not None,
                    'climate': self._climate is not None,
                    'ecology': self._ecology is not None,
                    'society': self._society is not None
                }
            },
            **self.stats
        }

        # Add climate statistics
        if self._climate:
            stats['climate'] = self._climate.get_global_statistics()

        # Add ecology statistics
        if self._ecology:
            stats['ecology'] = self._ecology.get_statistics()

        # Add society statistics
        if self._society:
            stats['society'] = self._society.get_global_statistics()

        return stats

    def save_state(self, filepath: Path) -> None:
        """Save current Earth state to file."""
        state = {
            'config': {
                'starting_date': self.config.starting_date.isoformat(),
                'simulation_resolution': self.config.simulation_resolution
            },
            'time': self.time.to_dict(),
            'statistics': self.get_global_statistics(),
            'events': self.events[-1000:]  # Last 1000 events
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"💾 Earth state saved to {filepath}")

    def load_state(self, filepath: Path) -> None:
        """Load Earth state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore time
        self.time.current_datetime = datetime.fromisoformat(
            state['time']['current_datetime']
        )
        self.time.time_scale = state['time']['time_scale']

        # Restore events
        self.events = state.get('events', [])

        print(f"📂 Earth state loaded from {filepath}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Earth(time={self.time.current_datetime.isoformat()}, "
            f"countries={self.stats.get('total_countries', 0)}, "
            f"population={self.stats.get('total_population', 0):,})"
        )
