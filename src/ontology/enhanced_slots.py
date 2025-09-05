"""
Enhanced Slot System for Interpretable Language Evolution

Key Innovation: Structured semantic slots with morphology rules
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
import numpy as np
from collections import defaultdict

@dataclass
class SlotDefinition:
    """Definition of a semantic slot with vocabulary and rules."""
    name: str
    vocabulary: List[str]
    morphology_rules: Dict[str, str] = field(default_factory=dict)
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    anchor_words: List[str] = field(default_factory=list)  # Fixed vocabulary core

@dataclass
class MorphologyRule:
    """Morphological transformation rule."""
    pattern: str
    replacement: str
    context: Optional[str] = None

class EnhancedSlotSystem:
    """
    Enhanced slot system supporting interpretable language evolution.
    
    Features:
    - Rich semantic slots with structured vocabulary
    - Morphology rules for productive word formation  
    - Anchor words to prevent semantic drift
    - Semantic distance calculation
    - Validation and consistency checking
    """
    
    def __init__(self):
        self.slots = self._initialize_slots()
        self.slot_order = ["ACTION", "OBJECT", "ATTRIBUTE", "LOCATION", "MODIFIER"]
        self.morphology_rules = self._initialize_morphology_rules()
        self.anchor_words = self._initialize_anchor_words()
        
        # Semantic relationships
        self.semantic_hierarchies = self._build_semantic_hierarchies()
        self.semantic_distances = self._precompute_semantic_distances()
    
    def _initialize_slots(self) -> List[SlotDefinition]:
        """Initialize comprehensive slot definitions."""
        
        slots = [
            SlotDefinition(
                name="ACTION",
                vocabulary=[
                    "MOVE", "TAKE", "DROP", "GIVE", "POINT", "LOOK", "SCAN", "WAIT",
                    "NAVIGATE", "SEARCH", "COLLECT", "PLACE", "ACTIVATE", "DEACTIVATE",
                    "OPEN", "CLOSE", "PUSH", "PULL", "ROTATE", "FLIP"
                ],
                semantic_features={
                    "transitivity": {"MOVE": "intransitive", "TAKE": "transitive", "GIVE": "ditransitive"},
                    "duration": {"MOVE": "extended", "LOOK": "instantaneous", "WAIT": "extended"},
                    "causality": {"PUSH": "causative", "MOVE": "non-causative"}
                },
                anchor_words=["MOVE", "TAKE", "LOOK"]  # Core actions that should remain stable
            ),
            
            SlotDefinition(
                name="OBJECT", 
                vocabulary=[
                    "CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND", "STAR", "CROSS", "HEXAGON", "OVAL",
                    "BOX", "SPHERE", "CUBE", "CYLINDER", "CONE", "PYRAMID", "PRISM",
                    "AGENT", "MARKER", "TOOL", "CONTAINER", "BARRIER", "TARGET", "RESOURCE"
                ],
                semantic_features={
                    "shape_class": {"CIRCLE": "2d", "SPHERE": "3d", "BOX": "3d"},
                    "animacy": {"AGENT": "animate", "CIRCLE": "inanimate"},
                    "manipulability": {"TOOL": "manipulable", "BARRIER": "fixed"}
                },
                anchor_words=["CIRCLE", "SQUARE", "TRIANGLE"]
            ),
            
            SlotDefinition(
                name="ATTRIBUTE",
                vocabulary=[
                    "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK", "CYAN",
                    "BLACK", "WHITE", "GRAY", "BROWN", "MAGENTA", "LIME", "NAVY", "SILVER",
                    "SMALL", "LARGE", "MEDIUM", "TINY", "HUGE", "NORMAL",
                    "BRIGHT", "DARK", "SHINY", "MATTE", "TRANSPARENT", "OPAQUE",
                    "HOT", "COLD", "WARM", "COOL", "SOFT", "HARD", "SMOOTH", "ROUGH"
                ],
                semantic_features={
                    "attribute_type": {
                        "RED": "color", "BLUE": "color", "SMALL": "size", 
                        "BRIGHT": "luminance", "HOT": "temperature", "SOFT": "texture"
                    },
                    "gradability": {"SMALL": True, "RED": False, "HOT": True}
                },
                anchor_words=["RED", "BLUE", "GREEN", "SMALL", "LARGE"]
            ),
            
            SlotDefinition(
                name="LOCATION",
                vocabulary=[
                    "LEFT", "RIGHT", "UP", "DOWN", "CENTER", "TOP", "BOTTOM", "MIDDLE",
                    "NEAR", "FAR", "CLOSE", "DISTANT", "ADJACENT", "OPPOSITE", "BESIDE",
                    "FRONT", "BACK", "BEHIND", "AHEAD", "AROUND", "INSIDE", "OUTSIDE",
                    "CORNER", "EDGE", "BORDER", "BOUNDARY", "ZONE_A", "ZONE_B", "ZONE_C"
                ],
                semantic_features={
                    "dimension": {"LEFT": "horizontal", "UP": "vertical", "NEAR": "distance"},
                    "reference_frame": {"LEFT": "relative", "ZONE_A": "absolute"},
                    "topology": {"INSIDE": "containment", "ADJACENT": "proximity"}
                },
                anchor_words=["LEFT", "RIGHT", "CENTER", "NEAR", "FAR"]
            ),
            
            SlotDefinition(
                name="MODIFIER",
                vocabulary=[
                    "NOT", "VERY", "SLIGHTLY", "QUITE", "EXTREMELY", "BARELY", "ALMOST",
                    "EXACTLY", "APPROXIMATELY", "ROUGHLY", "PRECISELY", "ABOUT",
                    "QUICKLY", "SLOWLY", "CAREFULLY", "FORCEFULLY", "GENTLY", "RAPIDLY",
                    "IMMEDIATELY", "EVENTUALLY", "SOON", "LATER", "BEFORE", "AFTER"
                ],
                semantic_features={
                    "modifier_type": {
                        "NOT": "negation", "VERY": "intensifier", "QUICKLY": "manner",
                        "EXACTLY": "precision", "SOON": "temporal"
                    },
                    "scope": {"NOT": "semantic", "VERY": "gradable_adjectives"},
                    "polarity": {"NOT": "negative", "VERY": "positive"}
                },
                anchor_words=["NOT", "VERY", "QUICKLY"]
            )
        ]
        
        return slots
    
    def _initialize_morphology_rules(self) -> Dict[str, List[MorphologyRule]]:
        """Initialize morphological transformation rules."""
        
        rules = {
            "ACTION": [
                MorphologyRule("(.+)", r"\1_ING", "progressive"),  # MOVE -> MOVE_ING
                MorphologyRule("(.+)", r"\1_ED", "past"),          # TAKE -> TAKE_ED
                MorphologyRule("(.+)", r"RE_\1", "repetitive"),    # MOVE -> RE_MOVE
            ],
            "ATTRIBUTE": [
                MorphologyRule("(.+)", r"\1_ER", "comparative"),   # RED -> RED_ER
                MorphologyRule("(.+)", r"\1_EST", "superlative"),  # BIG -> BIG_EST
                MorphologyRule("(.+)", r"UN_\1", "negation"),      # BRIGHT -> UN_BRIGHT
            ],
            "LOCATION": [
                MorphologyRule("(.+)", r"\1_WARD", "direction"),   # UP -> UP_WARD
                MorphologyRule("(.+)", r"\1_SIDE", "area"),        # LEFT -> LEFT_SIDE
            ]
        }
        
        return rules
    
    def _initialize_anchor_words(self) -> Dict[str, List[str]]:
        """Initialize anchor words to prevent semantic drift."""
        
        anchor_words = {}
        for slot in self.slots:
            anchor_words[slot.name] = slot.anchor_words.copy()
        
        return anchor_words
    
    def _build_semantic_hierarchies(self) -> Dict[str, Dict[str, List[str]]]:
        """Build semantic hierarchies for distance calculation."""
        
        hierarchies = {
            "ACTION": {
                "motion": ["MOVE", "NAVIGATE", "ROTATE", "FLIP"],
                "manipulation": ["TAKE", "DROP", "GIVE", "PLACE", "PUSH", "PULL"],
                "observation": ["LOOK", "SCAN", "SEARCH"],
                "control": ["ACTIVATE", "DEACTIVATE", "OPEN", "CLOSE"]
            },
            "OBJECT": {
                "2d_shapes": ["CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND", "STAR"],
                "3d_shapes": ["SPHERE", "CUBE", "CYLINDER", "CONE", "PYRAMID"],
                "agents": ["AGENT", "MARKER"],
                "tools": ["TOOL", "CONTAINER"]
            },
            "ATTRIBUTE": {
                "colors": ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE"],
                "sizes": ["SMALL", "MEDIUM", "LARGE", "TINY", "HUGE"],
                "textures": ["SOFT", "HARD", "SMOOTH", "ROUGH"],
                "temperatures": ["HOT", "COLD", "WARM", "COOL"]
            },
            "LOCATION": {
                "cardinal": ["LEFT", "RIGHT", "UP", "DOWN"],
                "proximity": ["NEAR", "FAR", "CLOSE", "DISTANT"],
                "containment": ["INSIDE", "OUTSIDE", "CENTER"],
                "zones": ["ZONE_A", "ZONE_B", "ZONE_C"]
            },
            "MODIFIER": {
                "negation": ["NOT"],
                "intensifiers": ["VERY", "EXTREMELY", "SLIGHTLY"],
                "manner": ["QUICKLY", "SLOWLY", "CAREFULLY"],
                "temporal": ["SOON", "LATER", "IMMEDIATELY"]
            }
        }
        
        return hierarchies
    
    def _precompute_semantic_distances(self) -> Dict[Tuple[str, str], float]:
        """Precompute semantic distances between vocabulary items."""
        
        distances = {}
        
        for slot in self.slots:
            slot_name = slot.name
            vocab = slot.vocabulary
            
            # Distance within same semantic category
            if slot_name in self.semantic_hierarchies:
                categories = self.semantic_hierarchies[slot_name]
                
                for word1 in vocab:
                    for word2 in vocab:
                        if word1 == word2:
                            distances[(word1, word2)] = 0.0
                        else:
                            # Find categories for each word
                            cat1 = None
                            cat2 = None
                            
                            for cat, words in categories.items():
                                if word1 in words:
                                    cat1 = cat
                                if word2 in words:
                                    cat2 = cat
                            
                            # Same category = closer distance
                            if cat1 == cat2 and cat1 is not None:
                                distances[(word1, word2)] = 0.3
                            # Different categories = farther distance
                            elif cat1 is not None and cat2 is not None:
                                distances[(word1, word2)] = 0.7
                            # Unknown category = maximum distance
                            else:
                                distances[(word1, word2)] = 1.0
        
        return distances
    
    def get_slot_vocabulary(self, slot_name: str) -> List[str]:
        """Get vocabulary for a specific slot."""
        for slot in self.slots:
            if slot.name == slot_name:
                return slot.vocabulary.copy()
        return []
    
    def get_slot_definition(self, slot_name: str) -> Optional[SlotDefinition]:
        """Get complete slot definition."""
        for slot in self.slots:
            if slot.name == slot_name:
                return slot
        return None
    
    def validate_semantics(self, semantics: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate semantic dictionary for completeness and correctness."""
        
        errors = []
        
        # Check required slots
        for slot_name in self.slot_order:
            if slot_name not in semantics:
                errors.append(f"Missing required slot: {slot_name}")
                continue
            
            # Check vocabulary validity
            value = semantics[slot_name]
            vocab = self.get_slot_vocabulary(slot_name)
            
            if value not in vocab:
                errors.append(f"Invalid value '{value}' for slot '{slot_name}'. Valid options: {vocab[:5]}...")
        
        # Check for unknown slots
        for slot_name in semantics:
            if slot_name not in self.slot_order:
                errors.append(f"Unknown slot: {slot_name}")
        
        return len(errors) == 0, errors
    
    def semantic_distance(self, sem1: Dict[str, str], sem2: Dict[str, str]) -> float:
        """Calculate normalized semantic distance between two semantics."""
        
        if not sem1 or not sem2:
            return 1.0
        
        total_distance = 0.0
        compared_slots = 0
        
        # Compare each slot
        for slot_name in self.slot_order:
            if slot_name in sem1 and slot_name in sem2:
                val1, val2 = sem1[slot_name], sem2[slot_name]
                
                # Use precomputed distances if available
                if (val1, val2) in self.semantic_distances:
                    distance = self.semantic_distances[(val1, val2)]
                elif (val2, val1) in self.semantic_distances:
                    distance = self.semantic_distances[(val2, val1)]
                else:
                    # Fallback: exact match = 0, different = 1
                    distance = 0.0 if val1 == val2 else 1.0
                
                total_distance += distance
                compared_slots += 1
            elif slot_name in sem1 or slot_name in sem2:
                # One has slot, other doesn't = maximum distance
                total_distance += 1.0
                compared_slots += 1
        
        return total_distance / max(compared_slots, 1)
    
    def apply_morphology(self, slot_name: str, word: str, rule_type: str) -> str:
        """Apply morphological rule to word."""
        
        if slot_name not in self.morphology_rules:
            return word
        
        rules = self.morphology_rules[slot_name]
        
        for rule in rules:
            if rule.context == rule_type:
                import re
                return re.sub(rule.pattern, rule.replacement, word)
        
        return word
    
    def generate_semantic_variants(self, base_semantics: Dict[str, str], num_variants: int = 5) -> List[Dict[str, str]]:
        """Generate semantic variants for training/testing."""
        
        variants = []
        
        for _ in range(num_variants):
            variant = base_semantics.copy()
            
            # Randomly modify 1-2 slots
            slots_to_modify = random.sample(list(variant.keys()), min(2, len(variant)))
            
            for slot_name in slots_to_modify:
                vocab = self.get_slot_vocabulary(slot_name)
                current_value = variant[slot_name]
                
                # Choose different value from same semantic category if possible
                if slot_name in self.semantic_hierarchies:
                    categories = self.semantic_hierarchies[slot_name]
                    current_category = None
                    
                    # Find current category
                    for cat, words in categories.items():
                        if current_value in words:
                            current_category = cat
                            break
                    
                    # Sample from same category first
                    if current_category:
                        same_category_words = [w for w in categories[current_category] if w != current_value]
                        if same_category_words:
                            variant[slot_name] = random.choice(same_category_words)
                            continue
                
                # Fallback: random from vocabulary
                other_values = [v for v in vocab if v != current_value]
                if other_values:
                    variant[slot_name] = random.choice(other_values)
            
            variants.append(variant)
        
        return variants
    
    def get_anchor_word_usage_rate(self, semantics_list: List[Dict[str, str]]) -> float:
        """Calculate anchor word usage rate."""
        
        total_words = 0
        anchor_usage = 0
        
        for semantics in semantics_list:
            for slot_name, value in semantics.items():
                total_words += 1
                if (slot_name in self.anchor_words and 
                    value in self.anchor_words[slot_name]):
                    anchor_usage += 1
        
        return anchor_usage / max(total_words, 1)

# Global instance
ENHANCED_SLOT_SYSTEM = EnhancedSlotSystem()

def sample_semantics() -> Dict[str, str]:
    """Sample random valid semantics."""
    semantics = {}
    
    for slot_name in ENHANCED_SLOT_SYSTEM.slot_order:
        vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_name)
        if vocab:
            semantics[slot_name] = random.choice(vocab)
    
    return semantics

def sample_semantics_with_constraints(constraints: Dict[str, List[str]]) -> Dict[str, str]:
    """Sample semantics with specific constraints."""
    semantics = {}
    
    for slot_name in ENHANCED_SLOT_SYSTEM.slot_order:
        if slot_name in constraints:
            # Use constrained vocabulary
            vocab = constraints[slot_name]
        else:
            # Use full vocabulary
            vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_name)
        
        if vocab:
            semantics[slot_name] = random.choice(vocab)
    
    return semantics
