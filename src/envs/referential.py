import random
from ontology.slots import SLOTS, VOCAB

def sample_semantics() -> dict:
    return {k: random.choice(VOCAB[k]) for k in SLOTS}

def topo_distance(s1: dict, s2: dict) -> int:
    return sum(int(s1[k] != s2[k]) for k in SLOTS)
