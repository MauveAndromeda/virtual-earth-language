#!/usr/bin/env python3
"""Quick demonstration of Virtual Earth 2.0 capabilities"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from earth import Earth, EarthConfig
from earth.core.earth import CoordinateSystem
from datetime import datetime


def main():
    print("\n" + "=" * 70)
    print("🌍 VIRTUAL EARTH 2.0 - Quick Demo")
    print("=" * 70 + "\n")

    # Initialize
    print("Initializing Virtual Earth...")
    earth = Earth(EarthConfig())
    earth.initialize()

    # Demo 1: Query famous locations
    print("\n📍 Querying Famous Locations")
    print("-" * 70)

    locations = [
        ("Eiffel Tower, Paris", 48.8584, 2.2945),
        ("Statue of Liberty, New York", 40.6892, -74.0445),
        ("Great Wall of China", 40.4319, 116.5704),
        ("Taj Mahal, India", 27.1751, 78.0421),
    ]

    for name, lat, lon in locations:
        coord = CoordinateSystem(latitude=lat, longitude=lon)
        info = earth.query_location(coord)

        print(f"\n📌 {name}")
        print(f"   🌍 {info['geography'].get('country', 'Unknown')}")
        print(f"   🌡️  {info['climate']['temperature_c']}°C - {info['climate']['description']}")
        print(f"   🌿 {info['ecology']['biome_type']}")
        print(f"   {'☀️ ' if info['solar']['is_daytime'] else '🌙'} {info['climate']['season'].capitalize()}")

    # Demo 2: Country comparison
    print("\n\n🏛️  Top Countries by Population")
    print("-" * 70)

    countries = ["China", "India", "United States"]
    for name in countries:
        country = earth.get_country(name)
        if country:
            print(f"\n{name}:")
            print(f"   Population: {country['population']:,}")
            print(f"   GDP: ${country['gdp_usd']:,.0f}")
            print(f"   Capital: {country['capital']}")

    # Demo 3: Climate zones
    print("\n\n🌡️  Temperature Across Latitudes")
    print("-" * 70)

    for lat in [-60, -30, 0, 30, 60]:
        coord = CoordinateSystem(latitude=lat, longitude=0)
        info = earth.query_location(coord)
        print(f"\nLatitude {lat:3d}°: {info['climate']['temperature_c']:5.1f}°C "
              f"({info['climate']['climate_zone_name']})")

    # Demo 4: Time simulation
    print("\n\n⏰ Time Simulation")
    print("-" * 70)

    tokyo = CoordinateSystem(35.6762, 139.6503)

    print(f"\nCurrent time: {earth.time.current_datetime}")
    info1 = earth.query_location(tokyo)
    print(f"Tokyo: {info1['climate']['temperature_c']}°C, "
          f"{'Daytime ☀️' if info1['solar']['is_daytime'] else 'Nighttime 🌙'}")

    earth.step(12 * 3600)  # 12 hours later

    print(f"\n12 hours later: {earth.time.current_datetime}")
    info2 = earth.query_location(tokyo)
    print(f"Tokyo: {info2['climate']['temperature_c']}°C, "
          f"{'Daytime ☀️' if info2['solar']['is_daytime'] else 'Nighttime 🌙'}")

    # Final stats
    print("\n\n📊 Global Statistics")
    print("-" * 70)

    stats = earth.get_global_statistics()
    print(f"\n🌍 Earth Surface: {stats['earth_metrics']['surface_area_km2']:,.0f} km²")
    print(f"🏛️  Countries: {stats['total_countries']}")
    print(f"🏙️  Cities: {stats['total_cities']}")
    print(f"👥 Population: {stats['total_population']:,}")

    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("\nTry interactive mode: python virtual_earth_main.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
