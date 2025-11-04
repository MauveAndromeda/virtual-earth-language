"""
Climate System - Realistic weather and climate modeling
======================================================

Implements:
- Köppen climate classification
- Seasonal variations
- Temperature and precipitation patterns
- Weather dynamics
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ClimateZone(Enum):
    """Köppen climate classification zones."""
    TROPICAL_RAINFOREST = "Af"           # Tropical rainforest
    TROPICAL_MONSOON = "Am"              # Tropical monsoon
    TROPICAL_SAVANNA = "Aw"              # Tropical savanna
    ARID_DESERT = "BWh"                  # Hot desert
    ARID_STEPPE = "BSh"                  # Hot semi-arid (steppe)
    TEMPERATE_OCEANIC = "Cfb"            # Temperate oceanic
    TEMPERATE_HUMID_SUBTROPICAL = "Cfa"  # Humid subtropical
    TEMPERATE_MEDITERRANEAN = "Csa"      # Hot-summer Mediterranean
    CONTINENTAL_HUMID = "Dfa"            # Hot-summer humid continental
    CONTINENTAL_SUBARCTIC = "Dfc"        # Subarctic
    POLAR_TUNDRA = "ET"                  # Tundra
    POLAR_ICE_CAP = "EF"                 # Ice cap


@dataclass
class WeatherConditions:
    """Current weather conditions at a location."""
    temperature_c: float
    precipitation_mm: float
    humidity_percent: float
    wind_speed_kmh: float
    cloud_cover_percent: float
    pressure_hpa: float
    description: str


class ClimateSystem:
    """
    Realistic climate and weather system.

    Provides:
    - Temperature calculations based on latitude, season, altitude
    - Precipitation patterns
    - Climate zone classification
    - Dynamic weather simulation
    """

    def __init__(self, config, time_system):
        """Initialize climate system."""
        self.config = config
        self.time = time_system

        # Climate constants
        self.avg_global_temp_c = 15.0
        self.temp_range_pole_to_equator = 60.0

        print("   🌡️  Climate system initialized")

    def get_conditions(self, coord) -> Dict[str, Any]:
        """Get current weather/climate conditions for coordinate."""
        # Calculate base temperature from latitude
        temp = self._calculate_temperature(coord)

        # Calculate precipitation
        precip = self._calculate_precipitation(coord)

        # Determine climate zone
        climate_zone = self._classify_climate(coord, temp, precip)

        # Create weather conditions
        weather = WeatherConditions(
            temperature_c=temp,
            precipitation_mm=precip,
            humidity_percent=self._calculate_humidity(coord, temp),
            wind_speed_kmh=self._calculate_wind_speed(coord),
            cloud_cover_percent=self._calculate_cloud_cover(coord, precip),
            pressure_hpa=self._calculate_pressure(coord),
            description=self._get_weather_description(temp, precip)
        )

        return {
            'temperature_c': temp,
            'temperature_f': temp * 9/5 + 32,
            'feels_like_c': temp - 2,  # Simplified
            'precipitation_mm': precip,
            'humidity_percent': weather.humidity_percent,
            'wind_speed_kmh': weather.wind_speed_kmh,
            'cloud_cover_percent': weather.cloud_cover_percent,
            'pressure_hpa': weather.pressure_hpa,
            'climate_zone': climate_zone.value,
            'climate_zone_name': climate_zone.name,
            'season': self.time.get_season(coord.latitude),
            'description': weather.description
        }

    def _calculate_temperature(self, coord) -> float:
        """Calculate temperature based on latitude, season, and altitude."""
        # Base temperature from latitude
        lat_factor = np.cos(np.radians(abs(coord.latitude)))
        base_temp = self.avg_global_temp_c + (self.temp_range_pole_to_equator / 2) * (lat_factor - 0.5)

        # Seasonal variation
        day_of_year = self.time.get_day_of_year()
        season_factor = np.cos(2 * np.pi * (day_of_year - 172) / 365.25)  # Peak in summer

        # Adjust for hemisphere
        if coord.latitude < 0:
            season_factor = -season_factor

        seasonal_variation = 15.0 * season_factor * (1 - abs(coord.latitude) / 90)
        temp = base_temp + seasonal_variation

        # Altitude adjustment (-6.5°C per 1000m)
        temp -= (coord.altitude / 1000.0) * 6.5

        # Diurnal variation
        hour = self.time.current_datetime.hour
        diurnal_variation = 5.0 * np.cos(2 * np.pi * (hour - 14) / 24)
        temp += diurnal_variation

        return round(temp, 1)

    def _calculate_precipitation(self, coord) -> float:
        """Calculate average monthly precipitation (mm)."""
        # Tropical regions (near equator) have high precipitation
        lat = abs(coord.latitude)

        if lat < 10:  # Tropical
            base_precip = 200
        elif lat < 23.5:  # Subtropical
            base_precip = 50   # Arid
        elif lat < 60:  # Temperate
            base_precip = 80
        else:  # Polar
            base_precip = 25

        # Seasonal variation
        season = self.time.get_season(coord.latitude)
        if season in ['spring', 'autumn']:
            seasonal_factor = 1.3
        elif season == 'summer':
            seasonal_factor = 1.1
        else:  # winter
            seasonal_factor = 0.8

        precip = base_precip * seasonal_factor

        # Add randomness for weather variability
        precip *= np.random.uniform(0.7, 1.3)

        return round(precip, 1)

    def _classify_climate(self, coord, temp: float, precip: float) -> ClimateZone:
        """Classify climate zone using Köppen classification."""
        lat = abs(coord.latitude)

        # Polar climates
        if temp < -10:
            return ClimateZone.POLAR_ICE_CAP
        elif temp < 10 and lat > 60:
            return ClimateZone.POLAR_TUNDRA

        # Arid climates
        elif precip < 250:
            if temp > 18:
                return ClimateZone.ARID_DESERT
            else:
                return ClimateZone.ARID_STEPPE

        # Tropical climates (near equator)
        elif lat < 15:
            if precip > 180:
                return ClimateZone.TROPICAL_RAINFOREST
            elif precip > 100:
                return ClimateZone.TROPICAL_MONSOON
            else:
                return ClimateZone.TROPICAL_SAVANNA

        # Temperate climates
        elif 23.5 < lat < 60:
            if temp > 22:
                return ClimateZone.TEMPERATE_HUMID_SUBTROPICAL
            elif precip < 80:
                return ClimateZone.TEMPERATE_MEDITERRANEAN
            else:
                return ClimateZone.TEMPERATE_OCEANIC

        # Continental climates
        elif lat > 40:
            if temp < 0:
                return ClimateZone.CONTINENTAL_SUBARCTIC
            else:
                return ClimateZone.CONTINENTAL_HUMID

        # Default to temperate oceanic
        return ClimateZone.TEMPERATE_OCEANIC

    def _calculate_humidity(self, coord, temp: float) -> float:
        """Calculate relative humidity percentage."""
        # Higher humidity in tropics and coastal areas
        lat = abs(coord.latitude)

        if lat < 15:  # Tropical
            base_humidity = 80
        elif lat < 60:  # Temperate
            base_humidity = 65
        else:  # Polar
            base_humidity = 70

        # Temperature affects humidity
        temp_factor = (temp + 10) / 30.0
        humidity = base_humidity * (0.8 + 0.4 * temp_factor)

        return round(min(100, max(10, humidity)), 1)

    def _calculate_wind_speed(self, coord) -> float:
        """Calculate wind speed (km/h)."""
        # Higher winds at mid-latitudes and coasts
        lat = abs(coord.latitude)

        if 30 < lat < 60:  # Mid-latitudes (prevailing westerlies)
            base_wind = 25
        elif lat > 60:  # Polar easterlies
            base_wind = 20
        else:  # Trade winds
            base_wind = 15

        # Add randomness
        wind = base_wind * np.random.uniform(0.5, 1.5)

        return round(wind, 1)

    def _calculate_cloud_cover(self, coord, precip: float) -> float:
        """Calculate cloud cover percentage."""
        # More clouds with higher precipitation
        base_cover = min(80, precip / 3)

        # Add daily variation
        hour = self.time.current_datetime.hour
        diurnal_var = 20 * np.sin(2 * np.pi * (hour - 6) / 24)

        cloud_cover = max(0, min(100, base_cover + diurnal_var))

        return round(cloud_cover, 1)

    def _calculate_pressure(self, coord) -> float:
        """Calculate atmospheric pressure (hPa)."""
        # Standard sea level pressure
        sea_level_pressure = 1013.25

        # Decrease with altitude (~12 hPa per 100m)
        pressure = sea_level_pressure - (coord.altitude / 100.0) * 12

        # Small random variation
        pressure += np.random.uniform(-5, 5)

        return round(pressure, 1)

    def _get_weather_description(self, temp: float, precip: float) -> str:
        """Get human-readable weather description."""
        # Temperature description
        if temp < 0:
            temp_desc = "freezing"
        elif temp < 10:
            temp_desc = "cold"
        elif temp < 20:
            temp_desc = "cool"
        elif temp < 30:
            temp_desc = "warm"
        else:
            temp_desc = "hot"

        # Precipitation description
        if precip < 10:
            precip_desc = "dry"
        elif precip < 50:
            precip_desc = "partly cloudy"
        elif precip < 150:
            precip_desc = "rainy"
        else:
            precip_desc = "heavy rain"

        return f"{temp_desc.capitalize()} and {precip_desc}"

    def update(self, delta_seconds: float) -> Dict[str, Any]:
        """Update climate system (called each simulation step)."""
        # Climate changes slowly, so most updates are minimal
        return {
            'time_advanced': delta_seconds,
            'global_avg_temp': self.avg_global_temp_c
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global climate statistics."""
        return {
            'avg_global_temperature_c': self.avg_global_temp_c,
            'climate_zones_count': len(ClimateZone),
            'current_season_northern': self.time.get_season(45.0),
            'current_season_southern': self.time.get_season(-45.0)
        }
