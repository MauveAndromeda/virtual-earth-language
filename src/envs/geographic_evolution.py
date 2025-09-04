# src/envs/geographic_evolution.py
"""
Geographic Language Evolution Environment

Implements geographic constraints on language evolution:
- Mountain ranges create communication barriers
- Islands enable isolated dialect development  
- Rivers facilitate trade and language contact
- Population density affects language change rates
- Migration patterns influence linguistic borrowing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import random
from pathlib import Path

@dataclass
class GeographicCell:
    """Individual cell in the geographic grid."""
    x: int
    y: int
    elevation: float
    terrain_type: str  # 'plains', 'mountain', 'water', 'forest'
    population_density: float
    language_variants: List[str] = field(default_factory=list)
    contact_frequency: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
@dataclass
class PopulationGroup:
    """Group of agents sharing similar language patterns."""
    group_id: str
    location: Tuple[int, int]
    size: int
    language_features: Dict[str, Any]
    migration_tendency: float
    contact_openness: float
    linguistic_conservatism: float
    
@dataclass
class LanguageContact:
    """Record of language contact between populations."""
    time_step: int
    group1: str
    group2: str
    contact_intensity: float
    features_exchanged: List[str]
    geographic_distance: float

class GeographicEnvironment:
    """
    Main geographic environment for language evolution simulation.
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (100, 100),
        terrain_type: str = "mixed",
        population_capacity: int = 10000,
        migration_rate: float = 0.1,
        contact_probability: float = 0.05,
        barrier_strength: float = 0.8,
        seed: Optional[int] = None
    ):
        self.width, self.height = size
        self.terrain_type = terrain_type
        self.population_capacity = population_capacity
        self.migration_rate = migration_rate
        self.contact_probability = contact_probability
        self.barrier_strength = barrier_strength
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize geographic grid
        self.grid = self._create_geographic_grid()
        
        # Initialize populations
        self.population_groups = {}
        self.language_contact_history = []
        
        # Tracking variables
        self.current_time_step = 0
        self.dialect_boundaries = {}
        self.language_family_tree = nx.DiGraph()
        
        # Metrics
        self.geographic_metrics = {
            'isolation_index': {},
            'contact_networks': {},
            'dialect_diversity': {},
            'migration_flows': defaultdict(list)
        }
        
    def _create_geographic_grid(self) -> np.ndarray:
        """Create geographic grid with specified terrain features."""
        
        grid = np.zeros((self.height, self.width), dtype=object)
        
        # Generate elevation map
        elevation_map = self._generate_elevation_map()
        
        # Generate terrain types based on elevation and configuration
        terrain_map = self._generate_terrain_map(elevation_map)
        
        # Populate grid with GeographicCell objects
        for y in range(self.height):
            for x in range(self.width):
                cell = GeographicCell(
                    x=x, y=y,
                    elevation=elevation_map[y, x],
                    terrain_type=terrain_map[y, x],
                    population_density=self._calculate_population_density(x, y, terrain_map[y, x])
                )
                grid[y, x] = cell
        
        return grid
    
    def _generate_elevation_map(self) -> np.ndarray:
        """Generate elevation map based on terrain type."""
        
        if self.terrain_type == "mountains":
            return self._create_mountain_terrain()
        elif self.terrain_type == "islands":
            return self._create_island_terrain()  
        elif self.terrain_type == "rivers":
            return self._create_river_terrain()
        elif self.terrain_type == "plains":
            return self._create_plain_terrain()
        else:  # mixed
            return self._create_mixed_terrain()
    
    def _create_mountain_terrain(self) -> np.ndarray:
        """Create mountainous terrain with valleys and peaks."""
        
        elevation = np.zeros((self.height, self.width))
        
        # Create mountain ranges
        num_ranges = random.randint(2, 4)
        
        for _ in range(num_ranges):
            # Random mountain range parameters
            center_x = random.randint(10, self.width - 10)
            center_y = random.randint(10, self.height - 10)
            length = random.randint(30, 60)
            width = random.randint(8, 15)
            orientation = random.uniform(0, np.pi)
            
            # Generate mountain range
            for i in range(length):
                x = int(center_x + i * np.cos(orientation))
                y = int(center_y + i * np.sin(orientation))
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Create mountain profile
                    for dx in range(-width, width + 1):
                        for dy in range(-width, width + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                distance = np.sqrt(dx**2 + dy**2)
                                if distance <= width:
                                    height = (1 - distance / width) * random.uniform(0.7, 1.0)
                                    elevation[ny, nx] = max(elevation[ny, nx], height)
        
        # Smooth elevation map
        elevation = gaussian_filter(elevation, sigma=1.5)
        
        return elevation
    
    def _create_island_terrain(self) -> np.ndarray:
        """Create archipelago with multiple islands."""
        
        elevation = np.full((self.height, self.width), -0.2)  # Ocean floor
        
        # Create islands
        num_islands = random.randint(5, 12)
        island_sizes = [random.randint(15, 40) for _ in range(num_islands)]
        
        for i, size in enumerate(island_sizes):
            # Random island position (avoid edges)
            center_x = random.randint(size, self.width - size)
            center_y = random.randint(size, self.height - size)
            
            # Create circular island with random shape variation
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Add random shape variation
                        angle = np.arctan2(dy, dx)
                        noise = 0.1 * np.sin(4 * angle) + 0.05 * np.sin(8 * angle)
                        actual_size = size * (1 + noise)
                        
                        if distance <= actual_size:
                            # Island height profile
                            height = (1 - distance / actual_size) * random.uniform(0.3, 0.8)
                            elevation[y, x] = max(elevation[y, x], height)
        
        # Smooth islands
        elevation = gaussian_filter(elevation, sigma=1.0)
        
        return elevation
    
    def _create_river_terrain(self) -> np.ndarray:
        """Create terrain with river systems."""
        
        # Base elevation with gentle hills
        elevation = np.random.uniform(0.1, 0.4, (self.height, self.width))
        elevation = gaussian_filter(elevation, sigma=3.0)
        
        # Create river systems
        num_rivers = random.randint(2, 4)
        
        for _ in range(num_rivers):
            # River source (high elevation)
            start_x = random.randint(0, self.width - 1)
            start_y = random.randint(0, self.height // 4)  # Start near top
            
            # River path (flows downward and toward edges)
            current_x, current_y = start_x, start_y
            river_length = random.randint(40, 80)
            
            river_path = [(current_x, current_y)]
            
            for step in range(river_length):
                # Bias toward flowing downward and outward
                dx = random.choice([-1, 0, 1])
                dy = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
                
                current_x = max(0, min(self.width - 1, current_x + dx))
                current_y = max(0, min(self.height - 1, current_y + dy))
                
                river_path.append((current_x, current_y))
                
                # Stop if reached edge
                if current_y >= self.height - 1:
                    break
            
            # Carve river valley
            for x, y in river_path:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= 2:
                                valley_depth = (1 - distance / 2) * 0.15
                                elevation[ny, nx] = max(-0.1, elevation[ny, nx] - valley_depth)
        
        return elevation
    
    def _create_plain_terrain(self) -> np.ndarray:
        """Create mostly flat plains with gentle variations."""
        
        elevation = np.random.uniform(0.0, 0.2, (self.height, self.width))
        elevation = gaussian_filter(elevation, sigma=5.0)
        
        return elevation
    
    def _create_mixed_terrain(self) -> np.ndarray:
        """Create mixed terrain with various geographic features."""
        
        elevation = np.zeros((self.height, self.width))
        
        # Base plains
        base_elevation = np.random.uniform(0.0, 0.1, (self.height, self.width))
        base_elevation = gaussian_filter(base_elevation, sigma=4.0)
        elevation += base_elevation
        
        # Add some mountains
        mountain_elevation = self._create_mountain_terrain() * 0.6
        elevation += mountain_elevation
        
        # Add river valleys
        river_effect = self._create_river_terrain() * 0.3
        elevation = np.minimum(elevation, elevation + river_effect - 0.1)
        
        return elevation
    
    def _generate_terrain_map(self, elevation_map: np.ndarray) -> np.ndarray:
        """Generate terrain types based on elevation."""
        
        terrain_map = np.full((self.height, self.width), 'plains', dtype=object)
        
        for y in range(self.height):
            for x in range(self.width):
                elev = elevation_map[y, x]
                
                if elev < 0:
                    terrain_map[y, x] = 'water'
                elif elev > 0.6:
                    terrain_map[y, x] = 'mountain'
                elif elev > 0.3:
                    terrain_map[y, x] = 'forest'
                else:
                    terrain_map[y, x] = 'plains'
        
        return terrain_map
    
    def _calculate_population_density(self, x: int, y: int, terrain: str) -> float:
        """Calculate population density based on terrain suitability."""
        
        base_densities = {
            'plains': 1.0,
            'forest': 0.6,
            'mountain': 0.2,
            'water': 0.0
        }
        
        base_density = base_densities.get(terrain, 0.5)
        
        # Add some random variation
        variation = random.uniform(0.8, 1.2)
        
        # Distance from center effect (more populated near center)
        center_x, center_y = self.width // 2, self.height // 2
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        center_effect = 1.0 - (distance_from_center / max_distance) * 0.5
        
        final_density = base_density * variation * center_effect
        return max(0.0, final_density)
    
    def initialize_populations(self, num_groups: int = 20) -> Dict[str, PopulationGroup]:
        """Initialize population groups across the geographic grid."""
        
        population_groups = {}
        
        for i in range(num_groups):
            # Find suitable location for population group
            attempts = 0
            while attempts < 100:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                
                cell = self.grid[y, x]
                if cell.population_density > 0.1:  # Suitable for habitation
                    break
                attempts += 1
            
            # Create population group
            group = PopulationGroup(
                group_id=f"group_{i:03d}",
                location=(x, y),
                size=int(cell.population_density * random.randint(100, 500)),
                language_features=self._generate_initial_language_features(),
                migration_tendency=random.uniform(0.0, 0.3),
                contact_openness=random.uniform(0.2, 0.8),
                linguistic_conservatism=random.uniform(0.1, 0.9)
            )
            
            population_groups[group.group_id] = group
        
        self.population_groups = population_groups
        return population_groups
    
    def _generate_initial_language_features(self) -> Dict[str, Any]:
        """Generate initial language features for a population group."""
        
        features = {
            'phonology': {
                'consonants': random.randint(15, 35),
                'vowels': random.randint(3, 12),
                'tone': random.choice(['none', 'simple', 'complex'])
            },
            'morphology': {
                'case_marking': random.choice(['none', 'nominative', 'ergative']),
                'verb_agreement': random.choice(['minimal', 'moderate', 'extensive']),
                'word_order': random.choice(['SVO', 'SOV', 'VSO', 'VOS', 'OVS', 'OSV'])
            },
            'lexicon': {
                'basic_vocabulary': random.randint(800, 1500),
                'color_terms': random.randint(3, 11),
                'kinship_complexity': random.choice(['simple', 'moderate', 'complex'])
            },
            'communication_patterns': {
                'slot_preference': [random.random() for _ in range(5)],  # For our interpretable system
                'message_complexity': random.uniform(0.3, 0.9),
                'redundancy_tolerance': random.uniform(0.1, 0.7)
            }
        }
        
        return features
    
    def step(self) -> Dict[str, Any]:
        """Execute one time step of geographic language evolution."""
        
        self.current_time_step += 1
        
        # 1. Calculate inter-group contact probabilities
        contact_matrix = self._calculate_contact_matrix()
        
        # 2. Execute population migrations
        migration_events = self._process_migrations()
        
        # 3. Process language contact events
        contact_events = self._process_language_contacts(contact_matrix)
        
        # 4. Apply language change due to geographic factors
        change_events = self._apply_geographic_language_change()
        
        # 5. Update dialect boundaries
        self._update_dialect_boundaries()
        
        # 6. Calculate geographic metrics
        metrics = self._calculate_geographic_metrics()
        
        step_results = {
            'time_step': self.current_time_step,
            'migration_events': migration_events,
            'contact_events': contact_events,
            'change_events': change_events,
            'dialect_boundaries': self.dialect_boundaries.copy(),
            'metrics': metrics
        }
        
        return step_results
    
    def _calculate_contact_matrix(self) -> np.ndarray:
        """Calculate contact probabilities between all population groups."""
        
        groups = list(self.population_groups.values())
        n_groups = len(groups)
        contact_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i != j:
                    contact_prob = self._calculate_pairwise_contact(group1, group2)
                    contact_matrix[i, j] = contact_prob
        
        return contact_matrix
    
    def _calculate_pairwise_contact(self, group1: PopulationGroup, group2: PopulationGroup) -> float:
        """Calculate contact probability between two specific groups."""
        
        x1, y1 = group1.location
        x2, y2 = group2.location
        
        # Geographic distance
        euclidean_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Calculate terrain barrier effect
        barrier_effect = self._calculate_terrain_barrier(group1.location, group2.location)
        
        # Effective distance considering barriers
        effective_distance = euclidean_distance * (1 + barrier_effect * self.barrier_strength)
        
        # Base contact probability (decreases with distance)
        base_contact = self.contact_probability * np.exp(-effective_distance / 20.0)
        
        # Modify by group characteristics
        openness_factor = (group1.contact_openness + group2.contact_openness) / 2
        size_factor = np.sqrt((group1.size * group2.size) / 10000)  # Larger groups more contact
        
        final_contact = base_contact * openness_factor * size_factor
        
        return min(1.0, final_contact)
    
    def _calculate_terrain_barrier(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> float:
        """Calculate terrain barrier effect between two locations."""
        
        x1, y1 = loc1
        x2, y2 = loc2
        
        # Sample points along the path
        num_samples = int(max(abs(x2 - x1), abs(y2 - y1)))
        if num_samples == 0:
            return 0.0
        
        barrier_accumulation = 0.0
        
        for i in range(num_samples):
            t = i / num_samples
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < self.width and 0 <= y < self.height:
                cell = self.grid[y, x]
                
                # Different terrain types create different barriers
                if cell.terrain_type == 'mountain':
                    barrier_accumulation += 0.5
                elif cell.terrain_type == 'water':
                    barrier_accumulation += 1.0  # Water is major barrier
                elif cell.terrain_type == 'forest':
                    barrier_accumulation += 0.2
                # Plains add no barrier
        
        # Average barrier strength along path
        average_barrier = barrier_accumulation / num_samples if num_samples > 0 else 0.0
        
        return average_barrier
    
    def _process_migrations(self) -> List[Dict[str, Any]]:
        """Process population migration events."""
        
        migration_events = []
        
        for group_id, group in self.population_groups.items():
            if random.random() < group.migration_tendency * self.migration_rate:
                
                # Find potential migration destination
                current_x, current_y = group.location
                
                # Search in radius for better location
                best_location = None
                best_score = self._evaluate_location_quality(current_x, current_y)
                
                search_radius = 10
                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        new_x = current_x + dx
                        new_y = current_y + dy
                        
                        if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                            (dx != 0 or dy != 0)):
                            
                            score = self._evaluate_location_quality(new_x, new_y)
                            if score > best_score * 1.2:  # Need significant improvement
                                best_score = score
                                best_location = (new_x, new_y)
                
                # Execute migration if better location found
                if best_location:
                    old_location = group.location
                    group.location = best_location
                    
                    migration_event = {
                        'group_id': group_id,
                        'from': old_location,
                        'to': best_location,
                        'distance': np.sqrt((old_location[0] - best_location[0])**2 + 
                                          (old_location[1] - best_location[1])**2),
                        'reason': 'better_location'
                    }
                    
                    migration_events.append(migration_event)
                    
                    # Record in metrics
                    self.geographic_metrics['migration_flows'][self.current_time_step].append(migration_event)
        
        return migration_events
    
    def _evaluate_location_quality(self, x: int, y: int) -> float:
        """Evaluate the quality of a location for population settlement."""
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        cell = self.grid[y, x]
        
        # Base quality from population density
        quality = cell.population_density
        
        # Bonus for being near water (but not in water)
        water_bonus = 0.0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx].terrain_type == 'water':
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= 3:
                            water_bonus += (1 - distance / 3) * 0.2
        
        quality += min(water_bonus, 0.3)  # Cap water bonus
        
        return quality
    
    def _process_language_contacts(self, contact_matrix: np.ndarray) -> List[LanguageContact]:
        """Process language contact events between groups."""
        
        contact_events = []
        groups = list(self.population_groups.values())
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                
                contact_prob = contact_matrix[i, j]
                
                if random.random() < contact_prob:
                    # Language contact occurs
                    
                    # Calculate contact intensity
                    intensity = contact_prob * random.uniform(0.5, 1.0)
                    
                    # Determine which features are exchanged
                    features_exchanged = self._determine_feature_exchange(group1, group2, intensity)
                    
                    # Apply language change
                    self._apply_contact_induced_change(group1, group2, features_exchanged, intensity)
                    
                    # Record contact event
                    geographic_distance = np.sqrt(
                        (group1.location[0] - group2.location[0])**2 +
                        (group1.location[1] - group2.location[1])**2
                    )
                    
                    contact_event = LanguageContact(
                        time_step=self.current_time_step,
                        group1=group1.group_id,
                        group2=group2.group_id,
                        contact_intensity=intensity,
                        features_exchanged=features_exchanged,
                        geographic_distance=geographic_distance
                    )
                    
                    contact_events.append(contact_event)
                    self.language_contact_history.append(contact_event)
        
        return contact_events
    
    def _determine_feature_exchange(
        self, 
        group1: PopulationGroup, 
        group2: PopulationGroup, 
        intensity: float
    ) -> List[str]:
        """Determine which linguistic features are exchanged in contact."""
        
        exchangeable_features = [
            'lexical_borrowing',
            'phonological_influence', 
            'morphological_borrowing',
            'syntactic_influence',
            'communication_patterns'
        ]
        
        features_exchanged = []
        
        for feature in exchangeable_features:
            # Probability of exchange depends on intensity and feature type
            exchange_probs = {
                'lexical_borrowing': 0.8,      # Most easily borrowed
                'phonological_influence': 0.4,
                'communication_patterns': 0.6,  # For our interpretable system
                'morphological_borrowing': 0.2,
                'syntactic_influence': 0.1      # Most resistant to borrowing
            }
            
            if random.random() < intensity * exchange_probs[feature]:
                features_exchanged.append(feature)
        
        return features_exchanged
    
    def _apply_contact_induced_change(
        self,
        group1: PopulationGroup,
        group2: PopulationGroup, 
        features_exchanged: List[str],
        intensity: float
    ):
        """Apply language changes due to contact between groups."""
        
        for feature in features_exchanged:
            
            if feature == 'lexical_borrowing':
                # Exchange some vocabulary
                change_rate = intensity * 0.1
                self._exchange_lexical_features(group1, group2, change_rate)
                
            elif feature == 'communication_patterns':
                # This is key for our interpretable system
                self._exchange_communication_patterns(group1, group2, intensity)
                
            elif feature == 'phonological_influence':
                # Phonological convergence
                change_rate = intensity * 0.05
                self._converge_phonological_features(group1, group2, change_rate)
            
            # Add other feature exchange types as needed
    
    def _exchange_communication_patterns(
        self, 
        group1: PopulationGroup, 
        group2: PopulationGroup, 
        intensity: float
    ):
        """Exchange communication patterns relevant to interpretable system."""
        
        # Get current communication patterns
        patterns1 = group1.language_features['communication_patterns']
        patterns2 = group2.language_features['communication_patterns']
        
        # Exchange slot preferences (key for interpretable system)
        slot_prefs1 = patterns1['slot_preference']
        slot_prefs2 = patterns2['slot_preference']
        
        convergence_rate = intensity * 0.1
        
        for i in range(len(slot_prefs1)):
            # Mutual influence
            avg_pref = (slot_prefs1[i] + slot_prefs2[i]) / 2
            slot_prefs1[i] += convergence_rate * (avg_pref - slot_prefs1[i])
            slot_prefs2[i] += convergence_rate * (avg_pref - slot_prefs2[i])
        
        # Exchange message complexity preferences
        avg_complexity = (patterns1['message_complexity'] + patterns2['message_complexity']) / 2
        patterns1['message_complexity'] += convergence_rate * (avg_complexity - patterns1['message_complexity'])
        patterns2['message_complexity'] += convergence_rate * (avg_complexity - patterns2['message_complexity'])
    
    def _exchange_lexical_features(
        self, 
        group1: PopulationGroup, 
        group2: PopulationGroup, 
        change_rate: float
    ):
        """Exchange lexical features between groups."""
        
        lex1 = group1.language_features['lexicon']
        lex2 = group2.language_features['lexicon']
        
        # Vocabulary size can increase through borrowing
        size_increase1 = int(change_rate * lex2['basic_vocabulary'])
        size_increase2 = int(change_rate * lex1['basic_vocabulary'])
        
        lex1['basic_vocabulary'] += size_increase1
        lex2['basic_vocabulary'] += size_increase2
        
        # Color terms can be borrowed (but conservatively)
        if change_rate > 0.05:
            avg_color_terms = (lex1['color_terms'] + lex2['color_terms']) / 2
            lex1['color_terms'] = int(lex1['color_terms'] + change_rate * (avg_color_terms - lex1['color_terms']))
            lex2['color_terms'] = int(lex2['color_terms'] + change_rate * (avg_color_terms - lex2['color_terms']))
    
    def _converge_phonological_features(
        self, 
        group1: PopulationGroup, 
        group2: PopulationGroup, 
        change_rate: float
    ):
        """Apply phonological convergence between groups."""
        
        phon1 = group1.language_features['phonology']
        phon2 = group2.language_features['phonology']
        
        # Consonant and vowel inventories can converge
        avg_consonants = (phon1['consonants'] + phon2['consonants']) / 2
        avg_vowels = (phon1['vowels'] + phon2['vowels']) / 2
        
        phon1['consonants'] = int(phon1['consonants'] + change_rate * (avg_consonants - phon1['consonants']))
        phon1['vowels'] = int(phon1['vowels'] + change_rate * (avg_vowels - phon1['vowels']))
        
        phon2['consonants'] = int(phon2['consonants'] + change_rate * (avg_consonants - phon2['consonants']))
        phon2['vowels'] = int(phon2['vowels'] + change_rate * (avg_vowels - phon2['vowels']))
    
    def _apply_geographic_language_change(self) -> List[Dict[str, Any]]:
        """Apply language changes due to geographic factors."""
        
        change_events = []
        
        for group_id, group in self.population_groups.items():
            x, y = group.location
            cell = self.grid[y, x]
            
            # Geographic isolation promotes language change/conservation
            isolation_level = self._calculate_isolation_level(group)
            
            # Isolated groups change more slowly (linguistic conservatism)
            if isolation_level > 0.7:
                # Apply conservative change
                change_event = self._apply_conservative_change(group, isolation_level)
                if change_event:
                    change_events.append(change_event)
            
            # Groups in contact-rich areas change faster
            elif isolation_level < 0.3:
                # Apply innovative change
                change_event = self._apply_innovative_change(group, isolation_level)
                if change_event:
                    change_events.append(change_event)
            
            # Terrain-specific changes
            terrain_change = self._apply_terrain_specific_change(group, cell.terrain_type)
            if terrain_change:
                change_events.append(terrain_change)
        
        return change_events
    
    def _calculate_isolation_level(self, group: PopulationGroup) -> float:
        """Calculate how geographically isolated a group is."""
        
        x, y = group.location
        
        # Count accessible neighbors within radius
        accessible_neighbors = 0
        total_positions_checked = 0
        
        radius = 15
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    total_positions_checked += 1
                    
                    # Check if position is accessible (not blocked by major barriers)
                    barrier_strength = self._calculate_terrain_barrier((x, y), (nx, ny))
                    if barrier_strength < 0.5:  # Accessible
                        cell = self.grid[ny, nx]
                        if cell.population_density > 0.1:  # Has potential for contact
                            accessible_neighbors += 1
        
        # Isolation is inverse of accessibility
        accessibility = accessible_neighbors / max(total_positions_checked, 1)
        isolation = 1.0 - accessibility
        
        return isolation
    
    def _apply_conservative_change(self, group: PopulationGroup, isolation: float) -> Optional[Dict[str, Any]]:
        """Apply conservative linguistic changes in isolated groups."""
        
        if random.random() < 0.1:  # Low rate of change
            
            # Isolated groups tend to preserve archaic features
            # But also develop unique innovations
            
            change_type = random.choice(['phonological_drift', 'lexical_loss', 'morphological_simplification'])
            
            change_event = {
                'group_id': group.group_id,
                'type': 'conservative_change',
                'subtype': change_type,
                'isolation_level': isolation,
                'time_step': self.current_time_step
            }
            
            # Apply the change to language features
            if change_type == 'lexical_loss':
                # Isolated groups may lose some vocabulary
                lexicon = group.language_features['lexicon']
                loss_rate = 0.02 * isolation
                lexicon['basic_vocabulary'] = int(lexicon['basic_vocabulary'] * (1 - loss_rate))
            
            return change_event
        
        return None
    
    def _apply_innovative_change(self, group: PopulationGroup, isolation: float) -> Optional[Dict[str, Any]]:
        """Apply innovative changes in contact-rich groups."""
        
        if random.random() < 0.2:  # Higher rate of change
            
            change_type = random.choice(['lexical_expansion', 'phonological_innovation', 'syntactic_innovation'])
            
            change_event = {
                'group_id': group.group_id,
                'type': 'innovative_change', 
                'subtype': change_type,
                'contact_level': 1.0 - isolation,
                'time_step': self.current_time_step
            }
            
            # Apply the change
            if change_type == 'lexical_expansion':
                # Contact-rich groups expand vocabulary
                lexicon = group.language_features['lexicon']
                expansion_rate = 0.05 * (1.0 - isolation)
                lexicon['basic_vocabulary'] = int(lexicon['basic_vocabulary'] * (1 + expansion_rate))
            
            return change_event
        
        return None
    
    def _apply_terrain_specific_change(self, group: PopulationGroup, terrain_type: str) -> Optional[Dict[str, Any]]:
        """Apply language changes specific to terrain type."""
        
        terrain_effects = {
            'mountain': {'phonological': 'ejectives', 'lexical': 'altitude_terms'},
            'forest': {'phonological': 'nasalization', 'lexical': 'botanical_terms'},
            'water': {'lexical': 'maritime_vocabulary', 'morphological': 'directional_markers'},
            'plains': {'syntactic': 'distance_marking', 'lexical': 'spatial_terms'}
        }
        
        if terrain_type in terrain_effects and random.random() < 0.05:
            effects = terrain_effects[terrain_type]
            
            change_event = {
                'group_id': group.group_id,
                'type': 'terrain_adaptation',
                'terrain': terrain_type,
                'effects': effects,
                'time_step': self.current_time_step
            }
            
            return change_event
        
        return None
    
    def _update_dialect_boundaries(self):
        """Update dialect boundary detection based on linguistic similarity."""
        
        groups = list(self.population_groups.values())
        
        # Calculate linguistic distances between all groups
        linguistic_distances = {}
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                distance = self._calculate_linguistic_distance(group1, group2)
                linguistic_distances[(group1.group_id, group2.group_id)] = distance
        
        # Identify dialect boundaries (high linguistic distance despite geographic proximity)
        boundaries = []
        
        for (group1_id, group2_id), ling_dist in linguistic_distances.items():
            group1 = self.population_groups[group1_id]
            group2 = self.population_groups[group2_id]
            
            geographic_dist = np.sqrt(
                (group1.location[0] - group2.location[0])**2 +
                (group1.location[1] - group2.location[1])**2
            )
            
            # Boundary detected if linguistic distance is high despite geographic proximity
            if ling_dist > 0.7 and geographic_dist < 20:
                boundaries.append({
                    'group1': group1_id,
                    'group2': group2_id,
                    'linguistic_distance': ling_dist,
                    'geographic_distance': geographic_dist,
                    'boundary_strength': ling_dist / (geographic_dist / 20 + 0.1)
                })
        
        self.dialect_boundaries[self.current_time_step] = boundaries
    
    def _calculate_linguistic_distance(self, group1: PopulationGroup, group2: PopulationGroup) -> float:
        """Calculate linguistic distance between two groups."""
        
        features1 = group1.language_features
        features2 = group2.language_features
        
        total_distance = 0.0
        component_count = 0
        
        # Phonological distance
        phon1, phon2 = features1['phonology'], features2['phonology']
        consonant_diff = abs(phon1['consonants'] - phon2['consonants']) / 30
        vowel_diff = abs(phon1['vowels'] - phon2['vowels']) / 10
        phon_distance = (consonant_diff + vowel_diff) / 2
        total_distance += phon_distance
        component_count += 1
        
        # Lexical distance
        lex1, lex2 = features1['lexicon'], features2['lexicon']
        vocab_diff = abs(lex1['basic_vocabulary'] - lex2['basic_vocabulary']) / 1000
        color_diff = abs(lex1['color_terms'] - lex2['color_terms']) / 10
        lex_distance = (vocab_diff + color_diff) / 2
        total_distance += lex_distance
        component_count += 1
        
        # Communication patterns distance (key for interpretable system)
        comm1, comm2 = features1['communication_patterns'], features2['communication_patterns']
        slot_prefs1, slot_prefs2 = comm1['slot_preference'], comm2['slot_preference']
        slot_distance = np.mean([abs(p1 - p2) for p1, p2 in zip(slot_prefs1, slot_prefs2)])
        
        complexity_diff = abs(comm1['message_complexity'] - comm2['message_complexity'])
        comm_distance = (slot_distance + complexity_diff) / 2
        total_distance += comm_distance
        component_count += 1
        
        return total_distance / component_count
    
    def _calculate_geographic_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive geographic metrics."""
        
        metrics = {
            'population_distribution': self._analyze_population_distribution(),
            'isolation_indices': self._calculate_all_isolation_indices(),
            'contact_network_density': self._calculate_contact_network_density(),
            'dialect_diversity_index': self._calculate_dialect_diversity(),
            'terrain_language_correlation': self._analyze_terrain_language_correlation()
        }
        
        return metrics
    
    def _analyze_population_distribution(self) -> Dict[str, float]:
        """Analyze population distribution across terrain types."""
        
        terrain_populations = defaultdict(list)
        
        for group in self.population_groups.values():
            x, y = group.location
            terrain = self.grid[y, x].terrain_type
            terrain_populations[terrain].append(group.size)
        
        distribution = {}
        total_pop = sum(sum(pops) for pops in terrain_populations.values())
        
        for terrain, populations in terrain_populations.items():
            terrain_total = sum(populations)
            distribution[f'{terrain}_proportion'] = terrain_total / total_pop if total_pop > 0 else 0
            distribution[f'{terrain}_avg_size'] = np.mean(populations) if populations else 0
        
        return distribution
    
    def _calculate_all_isolation_indices(self) -> Dict[str, float]:
        """Calculate isolation indices for all groups."""
        
        indices = {}
        for group_id, group in self.population_groups.items():
            indices[group_id] = self._calculate_isolation_level(group)
        
        return indices
    
    def _calculate_contact_network_density(self) -> float:
        """Calculate density of the inter-group contact network."""
        
        n_groups = len(self.population_groups)
        if n_groups < 2:
            return 0.0
        
        # Count actual contacts in recent history
        recent_contacts = [
            contact for contact in self.language_contact_history
            if self.current_time_step - contact.time_step < 10
        ]
        
        actual_edges = len(set((c.group1, c.group2) for c in recent_contacts))
        max_possible_edges = n_groups * (n_groups - 1) / 2
        
        return actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    
    def _calculate_dialect_diversity(self) -> float:
        """Calculate dialect diversity index."""
        
        if len(self.population_groups) < 2:
            return 0.0
        
        # Calculate pairwise linguistic distances
        groups = list(self.population_groups.values())
        distances = []
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                distance = self._calculate_linguistic_distance(group1, group2)
                distances.append(distance)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def _analyze_terrain_language_correlation(self) -> Dict[str, float]:
        """Analyze correlation between terrain and language features."""
        
        terrain_language_data = defaultdict(lambda: defaultdict(list))
        
        for group in self.population_groups.values():
            x, y = group.location
            terrain = self.grid[y, x].terrain_type
            
            # Collect language features by terrain
            features = group.language_features
            
            terrain_language_data[terrain]['consonants'].append(features['phonology']['consonants'])
            terrain_language_data[terrain]['vowels'].append(features['phonology']['vowels'])
            terrain_language_data[terrain]['vocabulary'].append(features['lexicon']['basic_vocabulary'])
            terrain_language_data[terrain]['complexity'].append(features['communication_patterns']['message_complexity'])
        
        # Calculate correlations
        correlations = {}
        
        for terrain in terrain_language_data:
            data = terrain_language_data[terrain]
            if len(data['consonants']) > 1:
                # Example correlation (simplified)
                consonants = np.array(data['consonants'])
                vocabulary = np.array(data['vocabulary'])
                
                if len(consonants) > 1 and len(vocabulary) > 1:
                    corr = np.corrcoef(consonants, vocabulary)[0, 1]
                    if not np.isnan(corr):
                        correlations[f'{terrain}_consonant_vocab_corr'] = float(corr)
        
        return correlations
    
    def visualize_geography(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Create visualization of the geographic environment."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Geographic Language Evolution Environment', fontsize=16)
        
        # 1. Elevation and terrain map
        ax1 = axes[0, 0]
        elevation_map = np.array([[self.grid[y, x].elevation for x in range(self.width)] 
                                 for y in range(self.height)])
        
        im1 = ax1.imshow(elevation_map, cmap='terrain', origin='lower')
        ax1.set_title('Elevation Map')
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # 2. Population density
        ax2 = axes[0, 1]
        pop_density_map = np.array([[self.grid[y, x].population_density for x in range(self.width)]
                                   for y in range(self.height)])
        
        im2 = ax2.imshow(pop_density_map, cmap='YlOrRd', origin='lower')
        ax2.set_title('Population Density')
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        # Add population groups
        for group in self.population_groups.values():
            x, y = group.location
            ax2.scatter(x, y, c='blue', s=group.size/20, alpha=0.7)
        
        # 3. Linguistic distance network
        ax3 = axes[1, 0]
        
        if len(self.population_groups) > 1:
            # Create network graph
            G = nx.Graph()
            groups = list(self.population_groups.values())
            
            # Add nodes
            for group in groups:
                G.add_node(group.group_id, pos=group.location)
            
            # Add edges for recent contacts
            recent_contacts = [c for c in self.language_contact_history 
                             if self.current_time_step - c.time_step < 5]
            
            for contact in recent_contacts:
                if not G.has_edge(contact.group1, contact.group2):
                    G.add_edge(contact.group1, contact.group2, 
                             weight=contact.contact_intensity)
            
            # Draw network
            pos = nx.get_node_attributes(G, 'pos')
            if pos:
                nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=50, alpha=0.7)
                nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.5)
        
        ax3.set_title('Language Contact Network')
        ax3.set_xlim(0, self.width)
        ax3.set_ylim(0, self.height)
        
        # 4. Dialect boundaries
        ax4 = axes[1, 1]
        
        # Show terrain as background
        terrain_colors = {'plains': 0, 'forest': 1, 'mountain': 2, 'water': 3}
        terrain_map = np.array([[terrain_colors.get(self.grid[y, x].terrain_type, 0) 
                               for x in range(self.width)] for y in range(self.height)])
        
        ax4.imshow(terrain_map, cmap='Set3', alpha=0.3, origin='lower')
        
        # Draw dialect boundaries
        if self.current_time_step in self.dialect_boundaries:
            boundaries = self.dialect_boundaries[self.current_time_step]
            for boundary in boundaries:
                group1 = self.population_groups[boundary['group1']]
                group2 = self.population_groups[boundary['group2']]
                
                x1, y1 = group1.location
                x2, y2 = group2.location
                
                # Draw boundary line
                ax4.plot([x1, x2], [y1, y2], 'r-', 
                        linewidth=boundary['boundary_strength']*3, alpha=0.7)
        
        ax4.set_title('Dialect Boundaries')
        ax4.set_xlim(0, self.width)
        ax4.set_ylim(0, self.height)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics of the simulation."""
        
        stats = {
            'environment': {
                'size': (self.width, self.height),
                'terrain_type': self.terrain_type,
                'current_time_step': self.current_time_step
            },
            'population': {
                'num_groups': len(self.population_groups),
                'total_population': sum(group.size for group in self.population_groups.values()),
                'avg_group_size': np.mean([group.size for group in self.population_groups.values()]),
                'geographic_spread': self._calculate_geographic_spread()
            },
            'linguistic_diversity': {
                'dialect_diversity_index': self._calculate_dialect_diversity(),
                'num_recent_contacts': len([c for c in self.language_contact_history 
                                          if self.current_time_step - c.time_step < 10]),
                'avg_isolation_level': np.mean([self._calculate_isolation_level(g) 
                                              for g in self.population_groups.values()])
            },
            'geographic_effects': {
                'terrain_distribution': self._get_terrain_distribution(),
                'migration_events': len(self.geographic_metrics['migration_flows']),
                'contact_network_density': self._calculate_contact_network_density()
            }
        }
        
        return stats
    
    def _calculate_geographic_spread(self) -> float:
        """Calculate how spread out the population groups are."""
        
        if len(self.population_groups) < 2:
            return 0.0
        
        locations = [group.location for group in self.population_groups.values()]
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)
        
        max_possible_spread = np.sqrt(self.width**2 + self.height**2)
        actual_spread = np.sqrt(x_spread**2 + y_spread**2)
        
        return actual_spread / max_possible_spread
    
    def _get_terrain_distribution(self) -> Dict[str, float]:
        """Get distribution of terrain types in the environment."""
        
        terrain_counts = defaultdict(int)
        total_cells = self.width * self.height
        
        for y in range(self.height):
            for x in range(self.width):
                terrain = self.grid[y, x].terrain_type
                terrain_counts[terrain] += 1
        
        return {terrain: count / total_cells for terrain, count in terrain_counts.items()}

# Convenience functions for easy use

def create_mountain_environment(size: Tuple[int, int] = (100, 100), **kwargs) -> GeographicEnvironment:
    """Create environment with mountain ranges."""
    return GeographicEnvironment(size=size, terrain_type="mountains", **kwargs)

def create_island_environment(size: Tuple[int, int] = (120, 120), **kwargs) -> GeographicEnvironment:
    """Create archipelago environment."""
    return GeographicEnvironment(size=size, terrain_type="islands", **kwargs)

def create_river_environment(size: Tuple[int, int] = (80, 80), **kwargs) -> GeographicEnvironment:
    """Create environment with river systems."""
    return GeographicEnvironment(size=size, terrain_type="rivers", **kwargs)
