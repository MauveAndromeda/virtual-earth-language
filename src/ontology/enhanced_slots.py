print('hello')
    def _initialize_morphology_rules(self) -> Dict[str, List[MorphologyRule]]:
        rules = {
            "ACTION": [
                MorphologyRule("(.+)", r"\1_ING", "progressive"),
                MorphologyRule("(.+)", r"\1_ED", "past"),
                MorphologyRule("(.+)", r"RE_\1", "repetitive")
            ],
            "ATTRIBUTE": [
                MorphologyRule("(.+)", r"\1_ER", "comparative"),
                MorphologyRule("(.+)", r"\1_EST", "superlative"),
                MorphologyRule("(.+)", r"UN_\1", "negation")
            ],
            "LOCATION": [
                MorphologyRule("(.+)", r"\1_WARD", "direction"),
                MorphologyRule("(.+)", r"\1_SIDE", "area")
            ]
        }
        return rules

    def _initialize_anchor_words(self) -> Dict[str, List[str]]:
        anchor_words = {}
        for slot in self.slots:
            anchor_words[slot.name] = slot.anchor_words.copy()
        return anchor_words

    def _build_semantic_hierarchies(self) -> Dict[str, Dict[str, List[str]]]:
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
        distances = {}
        for slot in self.slots:
            slot_name = slot.name
            vocab = slot.vocabulary
            if slot_name in self.semantic_hierarchies:
                categories = self.semantic_hierarchies[slot_name]
                for word1 in vocab:
                    for word2 in vocab:
                        if word1 == word2:
                            distances[(word1, word2)] = 0.0
                        else:
                            cat1 = None
                            cat2 = None
                            for cat, words in categories.items():
                                if word1 in words:
                                    cat1 = cat
                                if word2 in words:
                                    cat2 = cat
                            if cat1 == cat2 and cat1 is not None:
                                distances[(word1, word2)] = 0.3
                            elif cat1 is not None and cat2 is not None:
                                distances[(word1, word2)] = 0.7
                            else:
                                distances[(word1, word2)] = 1.0
        return distances

    def get_slot_vocabulary(self, slot_name: str) -> List[str]:
        for slot in self.slots:
            if slot.name == slot_name:
                return slot.vocabulary.copy()
        return []

    def get_slot_definition(self, slot_name: str) -> Optional[SlotDefinition]:
        for slot in self.slots:
            if slot.name == slot_name:
                return slot
        return None

    def validate_semantics(self, semantics: Dict[str, str]) -> Tuple[bool, List[str]]:
        errors = []
        for slot_name in self.slot_order:
            if slot_name not in semantics:
                errors.append(f"Missing required slot: {slot_name}")
                continue
            value = semantics[slot_name]
            vocab = self.get_slot_vocabulary(slot_name)
            if value not in vocab:
                errors.append(f"Invalid value '{value}' for slot '{slot_name}'. Valid options: {vocab[:5]}...")
        for slot_name in semantics:
            if slot_name not in self.slot_order:
                errors.append(f"Unknown slot: {slot_name}")
        return len(errors) == 0, errors
    def semantic_distance(self, sem1: Dict[str, str], sem2: Dict[str, str]) -> float:
        if not sem1 or not sem2:
            return 1.0
        total_distance = 0.0
        compared_slots = 0
        for slot_name in self.slot_order:
            if slot_name in sem1 and slot_name in sem2:
                val1, val2 = sem1[slot_name], sem2[slot_name]
                if (val1, val2) in self.semantic_distances:
                    distance = self.semantic_distances[(val1, val2)]
                elif (val2, val1) in self.semantic_distances:
                    distance = self.semantic_distances[(val2, val1)]
                else:
                    distance = 0.0 if val1 == val2 else 1.0
                total_distance += distance
                compared_slots += 1
            elif slot_name in sem1 or slot_name in sem2:
                total_distance += 1.0
                compared_slots += 1
        return total_distance / max(compared_slots, 1)

    def apply_morphology(self, slot_name: str, word: str, rule_type: str) -> str:
        if slot_name not in self.morphology_rules:
            return word
        rules = self.morphology_rules[slot_name]
        for rule in rules:
            if rule.context == rule_type:
                import re
                return re.sub(rule.pattern, rule.replacement, word)
        return word

    def generate_semantic_variants(self, base_semantics: Dict[str, str], num_variants: int = 5) -> List[Dict[str, str]]:
        variants = []
        for _ in range(num_variants):
            variant = base_semantics.copy()
            slots_to_modify = random.sample(list(variant.keys()), min(2, len(variant)))
            for slot_name in slots_to_modify:
                vocab = self.get_slot_vocabulary(slot_name)
                current_value = variant[slot_name]
                if slot_name in self.semantic_hierarchies:
                    categories = self.semantic_hierarchies[slot_name]
                    current_category = None
                    for cat, words in categories.items():
                        if current_value in words:
                            current_category = cat
                            break
                    if current_category:
                        same_category_words = [w for w in categories[current_category] if w != current_value]
                        if same_category_words:
                            variant[slot_name] = random.choice(same_category_words)
                            continue
                other_values = [v for v in vocab if v != current_value]
                if other_values:
                    variant[slot_name] = random.choice(other_values)
            variants.append(variant)
        return variants

    def get_anchor_word_usage_rate(self, semantics_list: List[Dict[str, str]]) -> float:
        total_words = 0
        anchor_usage = 0
        for semantics in semantics_list:
            for slot_name, value in semantics.items():
                total_words += 1
                if slot_name in self.anchor_words and value in self.anchor_words[slot_name]:
                    anchor_usage += 1
        return anchor_usage / max(total_words, 1)

ENHANCED_SLOT_SYSTEM = EnhancedSlotSystem()

def sample_semantics() -> Dict[str, str]:
    semantics = {}
    for slot_name in ENHANCED_SLOT_SYSTEM.slot_order:
        vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_name)
        if vocab:
            semantics[slot_name] = random.choice(vocab)
    return semantics

def sample_semantics_with_constraints(constraints: Dict[str, List[str]]) -> Dict[str, str]:
    semantics = {}
    for slot_name in ENHANCED_SLOT_SYSTEM.slot_order:
        if slot_name in constraints:
            vocab = constraints[slot_name]
        else:
            vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_name)
        if vocab:
            semantics[slot_name] = random.choice(vocab)
    return semantics
