#!/usr/bin/env python3
"""Quick test script for Virtual Earth 2.0"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from earth import Earth, EarthConfig
from earth.core.earth import CoordinateSystem
from datetime import datetime


def test_basic_initialization():
    """Test basic Earth initialization."""
    print("Testing Earth initialization...")
    config = EarthConfig()
    earth = Earth(config)
    earth.initialize()
    print("✓ Earth initialized successfully\n")
    return earth


def test_country_queries(earth):
    """Test country queries."""
    print("Testing country queries...")
    countries = ["United States", "China", "Japan", "Germany"]

    for country in countries:
        info = earth.get_country(country)
        assert info is not None, f"Failed to get info for {country}"
        assert 'population' in info
        assert 'capital' in info
        print(f"✓ {country}: {info['capital']}, pop: {info['population']:,}")

    print()


def test_city_queries(earth):
    """Test city queries."""
    print("Testing city queries...")
    cities = ["Tokyo", "London", "New York", "Paris"]

    for city in cities:
        info = earth.get_city(city)
        assert info is not None, f"Failed to get info for {city}"
        assert 'country' in info
        assert 'population' in info
        print(f"✓ {city}: {info['country']}, pop: {info['population']:,}")

    print()


def test_location_queries(earth):
    """Test location queries."""
    print("Testing location queries...")

    locations = [
        ("New York", 40.7128, -74.0060),
        ("Tokyo", 35.6762, 139.6503),
        ("Sydney", -33.8688, 151.2093),
        ("North Pole", 90.0, 0.0),
    ]

    for name, lat, lon in locations:
        coord = CoordinateSystem(latitude=lat, longitude=lon)
        info = earth.query_location(coord)

        assert 'geography' in info
        assert 'climate' in info
        assert 'ecology' in info

        print(f"✓ {name}: {info['climate']['temperature_c']:.1f}°C, {info['climate']['climate_zone_name']}")

    print()


def test_time_simulation(earth):
    """Test time advancement."""
    print("Testing time simulation...")

    initial_time = earth.time.current_datetime
    print(f"  Initial time: {initial_time}")

    # Advance 24 hours
    earth.step(24 * 3600)
    new_time = earth.time.current_datetime

    assert new_time > initial_time
    print(f"  After 24h: {new_time}")
    print("✓ Time simulation working\n")


def test_statistics(earth):
    """Test global statistics."""
    print("Testing global statistics...")

    stats = earth.get_global_statistics()

    assert 'total_countries' in stats
    assert 'total_population' in stats

    print(f"✓ Countries: {stats['total_countries']}")
    print(f"✓ Population: {stats['total_population']:,}")
    print(f"✓ Cities: {stats['total_cities']}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 VIRTUAL EARTH 2.0 - TEST SUITE")
    print("=" * 60 + "\n")

    try:
        earth = test_basic_initialization()
        test_country_queries(earth)
        test_city_queries(earth)
        test_location_queries(earth)
        test_time_simulation(earth)
        test_statistics(earth)

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
