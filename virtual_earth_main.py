#!/usr/bin/env python3
"""
Virtual Earth - Main Entry Point
=================================

A comprehensive, realistic simulation of planet Earth.

Features:
- Real geography (195+ countries, 1000+ cities)
- Real climate and weather systems
- Real ecology and biomes
- Real human civilization (economies, cultures)
- Real-time day/night and seasonal cycles

Usage:
    python virtual_earth_main.py

Examples:
    # Query a location
    earth.query_location(CoordinateSystem(40.7128, -74.0060))  # New York

    # Get country info
    earth.get_country('United States')

    # Get city info
    earth.get_city('Tokyo')

    # Advance simulation
    earth.step(3600)  # Advance by 1 hour
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from earth import Earth, EarthConfig
from earth.core.earth import CoordinateSystem
from datetime import datetime
import json


def demo_earth_query():
    """Demonstrate Earth query capabilities."""
    print("\n" + "=" * 70)
    print("🌍 VIRTUAL EARTH - Comprehensive Real-World Simulation")
    print("=" * 70)

    # Initialize Earth
    config = EarthConfig(
        simulation_resolution="high",
        starting_date=datetime(2025, 1, 1, 12, 0, 0),
        enable_climate=True,
        enable_ecology=True,
        enable_society=True
    )

    earth = Earth(config)
    earth.initialize()

    # Demo 1: Query major cities
    print("\n📍 QUERYING MAJOR CITIES")
    print("-" * 70)

    cities_to_query = [
        ("Tokyo", 35.6762, 139.6503),
        ("New York", 40.7128, -74.0060),
        ("Paris", 48.8566, 2.3522),
        ("Sydney", -33.8688, 151.2093),
        ("Cairo", 30.0444, 31.2357)
    ]

    for city_name, lat, lon in cities_to_query:
        coord = CoordinateSystem(latitude=lat, longitude=lon)
        info = earth.query_location(coord)

        print(f"\n🌆 {city_name}")
        print(f"   Location: {lat:.4f}°, {lon:.4f}°")
        print(f"   Country: {info['geography'].get('country', 'Unknown')}")
        print(f"   Continent: {info['geography'].get('continent', 'Unknown')}")
        print(f"   Climate: {info['climate']['climate_zone_name']}")
        print(f"   Temperature: {info['climate']['temperature_c']:.1f}°C ({info['climate']['temperature_f']:.1f}°F)")
        print(f"   Season: {info['climate']['season'].capitalize()}")
        print(f"   Weather: {info['climate']['description']}")
        print(f"   Biome: {info['ecology']['biome_type']}")
        print(f"   Solar: {'Daytime ☀️' if info['solar']['is_daytime'] else 'Nighttime 🌙'}")

    # Demo 2: Country information
    print("\n\n🏛️  COUNTRY INFORMATION")
    print("-" * 70)

    countries_to_query = ["United States", "China", "Japan", "Germany", "Brazil"]

    for country_name in countries_to_query:
        country_info = earth.get_country(country_name)
        if country_info:
            print(f"\n🌐 {country_name}")
            print(f"   Capital: {country_info.get('capital')}")
            print(f"   Population: {country_info.get('population', 0):,}")
            print(f"   Area: {country_info.get('area_km2', 0):,} km²")
            print(f"   GDP: ${country_info.get('gdp_usd', 0):,.0f}")
            print(f"   Government: {country_info.get('government_type')}")
            print(f"   Languages: {', '.join(country_info.get('languages', []))}")

    # Demo 3: City information
    print("\n\n🏙️  CITY INFORMATION")
    print("-" * 70)

    cities_info = ["Tokyo", "London", "Beijing", "Mumbai", "São Paulo"]

    for city_name in cities_info:
        city = earth.get_city(city_name)
        if city:
            print(f"\n🌆 {city_name}")
            print(f"   Country: {city.get('country')}")
            print(f"   Population: {city.get('population', 0):,}")
            print(f"   Coordinates: {city.get('latitude'):.4f}°, {city.get('longitude'):.4f}°")
            if city.get('is_capital'):
                print(f"   Status: Capital City ⭐")

    # Demo 4: Global statistics
    print("\n\n📊 GLOBAL STATISTICS")
    print("-" * 70)

    stats = earth.get_global_statistics()

    print(f"\n🌍 Earth Metrics:")
    print(f"   Total surface area: {stats['earth_metrics']['surface_area_km2']:,.0f} km²")
    print(f"   Land area: {stats['earth_metrics']['land_area_km2']:,.0f} km²")
    print(f"   Ocean area: {stats['earth_metrics']['ocean_area_km2']:,.0f} km²")

    print(f"\n🏛️  Civilization:")
    print(f"   Total countries: {stats['total_countries']}")
    print(f"   Total cities: {stats['total_cities']}")
    print(f"   Total population: {stats['total_population']:,}")
    if 'society' in stats:
        print(f"   Global GDP: ${stats['society']['total_gdp_usd']:,.0f}")

    print(f"\n🌡️  Climate:")
    if 'climate' in stats:
        print(f"   Global avg temperature: {stats['climate']['avg_global_temperature_c']:.1f}°C")
        print(f"   Northern hemisphere season: {stats['climate']['current_season_northern'].capitalize()}")
        print(f"   Southern hemisphere season: {stats['climate']['current_season_southern'].capitalize()}")

    print(f"\n🕐 Simulation Time:")
    print(f"   Current time: {earth.time.current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Day of year: {earth.time.get_day_of_year()}")
    print(f"   Time scale: {earth.time.time_scale}x")

    # Demo 5: Time simulation
    print("\n\n⏰ TIME SIMULATION")
    print("-" * 70)

    print(f"\nAdvancing simulation by 6 hours...")
    results = earth.step(6 * 3600)  # 6 hours

    print(f"✓ Time advanced to: {earth.time.current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Query Tokyo again to see time changes
    tokyo_coord = CoordinateSystem(latitude=35.6762, longitude=139.6503)
    tokyo_info = earth.query_location(tokyo_coord)

    print(f"\n🌆 Tokyo (after time advancement):")
    print(f"   Temperature: {tokyo_info['climate']['temperature_c']:.1f}°C")
    print(f"   Solar: {'Daytime ☀️' if tokyo_info['solar']['is_daytime'] else 'Nighttime 🌙'}")
    print(f"   Weather: {tokyo_info['climate']['description']}")

    # Demo 6: Save state
    print("\n\n💾 SAVING EARTH STATE")
    print("-" * 70)

    save_path = Path("earth_state.json")
    earth.save_state(save_path)
    print(f"✓ Earth state saved to {save_path}")

    print("\n" + "=" * 70)
    print("✅ Virtual Earth demonstration complete!")
    print("=" * 70 + "\n")

    return earth


def interactive_mode(earth):
    """Interactive mode for querying the Earth."""
    print("\n🌍 INTERACTIVE MODE")
    print("Type 'help' for commands, 'quit' to exit\n")

    while True:
        try:
            cmd = input("earth> ").strip().lower()

            if cmd in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            elif cmd == 'help':
                print("\nAvailable commands:")
                print("  country <name>  - Get country information")
                print("  city <name>     - Get city information")
                print("  query <lat> <lon> - Query a location")
                print("  stats           - Show global statistics")
                print("  time            - Show current time")
                print("  step <hours>    - Advance time by N hours")
                print("  quit            - Exit")

            elif cmd.startswith('country '):
                country_name = cmd[8:].strip().title()
                info = earth.get_country(country_name)
                if info:
                    print(json.dumps(info, indent=2, default=str))
                else:
                    print(f"Country '{country_name}' not found")

            elif cmd.startswith('city '):
                city_name = cmd[5:].strip().title()
                info = earth.get_city(city_name)
                if info:
                    print(json.dumps(info, indent=2, default=str))
                else:
                    print(f"City '{city_name}' not found")

            elif cmd.startswith('query '):
                parts = cmd[6:].split()
                if len(parts) == 2:
                    lat, lon = float(parts[0]), float(parts[1])
                    coord = CoordinateSystem(latitude=lat, longitude=lon)
                    info = earth.query_location(coord)
                    print(json.dumps(info, indent=2, default=str))

            elif cmd == 'stats':
                stats = earth.get_global_statistics()
                print(json.dumps(stats, indent=2, default=str))

            elif cmd == 'time':
                print(f"Current time: {earth.time.current_datetime}")
                print(f"Day of year: {earth.time.get_day_of_year()}")

            elif cmd.startswith('step '):
                hours = float(cmd[5:])
                earth.step(hours * 3600)
                print(f"✓ Advanced {hours} hours to {earth.time.current_datetime}")

            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🌍 VIRTUAL EARTH v2.0 - Real World Simulation")
    print("=" * 70)
    print("\nChoose mode:")
    print("  1. Demo mode (automated demonstration)")
    print("  2. Interactive mode (query the Earth)")
    print("  3. Exit")

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            earth = demo_earth_query()
            print("\nWould you like to enter interactive mode? (y/n): ", end='')
            if input().strip().lower() == 'y':
                interactive_mode(earth)

        elif choice == '2':
            config = EarthConfig(
                simulation_resolution="high",
                starting_date=datetime.now(),
                enable_climate=True,
                enable_ecology=True,
                enable_society=True
            )
            earth = Earth(config)
            earth.initialize()
            interactive_mode(earth)

        elif choice == '3':
            print("👋 Goodbye!")

        else:
            print("Invalid choice. Exiting.")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
