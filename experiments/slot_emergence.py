"""
Slot-Based Grammar Emergence Experiment

Tracks how slot-structured grammar emerges during language evolution.
Measures whether agents develop positional semantics that align with slots.

Key Metrics:
- Slot alignment score (position-meaning correlation)
- Slot consistency across messages
- Slot vocabulary specialization
- Emergence timeline (when do slots stabilize)
"""

import sys
import random
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ontology.slots import SLOTS, VOCAB
from envs.referential import sample_semantics
from agents.speaker import Speaker
from explain.codec import code_from_sem, sem_from_code


def analyze_slot_alignment(messages: List[str], semantics_list: List[Dict]) -> Dict:
    """
    Analyze how well message positions align with semantic slots.

    Parameters
    ----------
    messages : List[str]
        Generated messages in format "ACT:val|OBJ:val|ATTR:val|LOC:val"
    semantics_list : List[Dict]
        Corresponding semantics

    Returns
    -------
    Dict with slot alignment metrics
    """
    slot_positions = defaultdict(lambda: defaultdict(int))

    # Parse messages and track where each slot appears
    for msg, sem in zip(messages, semantics_list):
        parts = msg.split('|')
        for pos, part in enumerate(parts):
            if ':' in part:
                slot_type, value = part.split(':', 1)
                slot_positions[slot_type][pos] += 1

    # Compute consistency: does each slot always appear in same position?
    alignment_scores = {}
    for slot in SLOTS:
        if slot in slot_positions:
            positions = slot_positions[slot]
            total = sum(positions.values())
            max_count = max(positions.values()) if positions else 0
            consistency = max_count / total if total > 0 else 0.0
            alignment_scores[slot] = consistency
        else:
            alignment_scores[slot] = 0.0

    return {
        'per_slot_alignment': alignment_scores,
        'mean_alignment': sum(alignment_scores.values()) / len(alignment_scores),
        'slot_positions': dict(slot_positions)
    }


def analyze_vocabulary_specialization(
    messages: List[str],
    semantics_list: List[Dict]
) -> Dict:
    """
    Analyze whether vocabulary specializes to specific slots.

    Returns metrics about vocabulary-slot associations.
    """
    word_slot_counts = defaultdict(lambda: defaultdict(int))

    # Track which slots each word appears in
    for msg, sem in zip(messages, semantics_list):
        for slot in SLOTS:
            if slot in sem:
                value = sem[slot]
                # Count this value appearing in this slot
                word_slot_counts[value][slot] += 1

    # Compute specialization: does each word primarily appear in one slot?
    specialization_scores = []
    for word, slot_counts in word_slot_counts.items():
        total = sum(slot_counts.values())
        max_count = max(slot_counts.values()) if slot_counts else 0
        specialization = max_count / total if total > 0 else 0.0
        specialization_scores.append(specialization)

    return {
        'word_slot_associations': dict(word_slot_counts),
        'mean_specialization': sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0,
        'num_specialized_words': sum(1 for s in specialization_scores if s > 0.8)
    }


def slot_emergence_experiment(
    num_messages: int = 200,
    seed: int = 42
) -> Dict:
    """
    Run slot emergence tracking experiment.

    Parameters
    ----------
    num_messages : int
        Number of messages to generate and analyze
    seed : int
        Random seed

    Returns
    -------
    Dict with emergence analysis
    """
    random.seed(seed)

    print(f"\n{'='*60}")
    print("SLOT-BASED GRAMMAR EMERGENCE EXPERIMENT")
    print(f"{'='*60}\n")

    # Create agent
    speaker = Speaker()

    # Generate messages
    print(f"Generating {num_messages} messages...")
    messages = []
    semantics_list = []

    for i in range(num_messages):
        sem = sample_semantics()
        code, _ = speaker.generate(sem)
        messages.append(code)
        semantics_list.append(sem)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_messages}")

    print("\nAnalyzing slot structure...")

    # Analyze slot alignment
    alignment_results = analyze_slot_alignment(messages, semantics_list)

    print("\nSlot Positional Consistency:")
    for slot, score in alignment_results['per_slot_alignment'].items():
        print(f"  {slot}: {score:.3f}")

    # Analyze vocabulary specialization
    specialization_results = analyze_vocabulary_specialization(messages, semantics_list)

    print(f"\nVocabulary Specialization:")
    print(f"  Mean specialization: {specialization_results['mean_specialization']:.3f}")
    print(f"  Highly specialized words: {specialization_results['num_specialized_words']}")

    # Combine results
    results = {
        'num_messages': num_messages,
        'alignment': alignment_results,
        'specialization': specialization_results,
        'sample_messages': messages[:5]
    }

    return results


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for slot emergence experiment."""
    if argv is None:
        argv = sys.argv[1:]

    # Parse arguments
    num_messages = 200

    if len(argv) > 0:
        num_messages = int(argv[0])

    # Run experiment
    results = slot_emergence_experiment(num_messages=num_messages)

    # Display final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Messages analyzed: {results['num_messages']}")
    print(f"")
    print(f"Mean slot alignment: {results['alignment']['mean_alignment']:.3f}")
    print(f"Mean vocabulary specialization: {results['specialization']['mean_specialization']:.3f}")
    print(f"")

    # Show sample messages
    print("Sample Messages:")
    for i, msg in enumerate(results['sample_messages'], 1):
        print(f"  {i}. {msg}")
    print(f"")

    # Emergence assessment
    mean_score = (results['alignment']['mean_alignment'] +
                  results['specialization']['mean_specialization']) / 2

    if mean_score > 0.8:
        print("✓ STRONG EMERGENCE: Clear slot-based grammar structure")
    elif mean_score > 0.6:
        print("~ MODERATE EMERGENCE: Partial slot structure visible")
    else:
        print("✗ WEAK EMERGENCE: Limited slot structure")

    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
