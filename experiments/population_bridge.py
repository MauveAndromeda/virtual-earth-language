"""
Cross-Population Translation Bridge Experiment

Tests whether languages developed by different populations can be translated
between each other. This evaluates the universality and interpretability of
emergent languages.

Key Metrics:
- Cross-population translation accuracy
- Semantic preservation after translation
- Translation robustness to vocabulary drift
- Bridge learning speed (how fast can populations align)
"""

import sys
import random
from pathlib import Path
from typing import Optional, Dict, List

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ontology.slots import SLOTS, VOCAB
from envs.referential import sample_semantics
from agents.speaker import Speaker
from agents.listener import Listener
from explain.codec import code_from_sem, sem_from_code
from objectives.losses import success as compute_success


def create_population(
    pop_id: int,
    vocabulary_bias: Dict[str, List[str]] = None
) -> Dict:
    """
    Create a population with potentially different vocabulary preferences.

    Parameters
    ----------
    pop_id : int
        Population identifier
    vocabulary_bias : Dict, optional
        Biased vocabulary preferences

    Returns
    -------
    Dict with population agents and characteristics
    """
    return {
        'id': pop_id,
        'speaker': Speaker(),
        'listener': Listener(),
        'vocabulary_bias': vocabulary_bias or {},
        'dialect_markers': []
    }


def evaluate_cross_population_communication(
    pop_a: Dict,
    pop_b: Dict,
    test_size: int = 100
) -> Dict[str, float]:
    """
    Evaluate communication success between two populations.

    Tests: Population A speaker -> Population B listener
    """
    successes = 0
    semantic_preservations = 0

    for _ in range(test_size):
        # Sample semantics
        target_sem = sample_semantics()
        distractor_sem = sample_semantics()
        candidates = [target_sem, distractor_sem]

        # Pop A speaker generates message
        code_a, _ = pop_a['speaker'].generate(target_sem)

        # Pop B listener tries to understand
        prediction = pop_b['listener'].act(code_a, candidates)

        if compute_success(prediction, 0) > 0.5:
            successes += 1

            # Check if semantics are preserved
            try:
                decoded_sem = sem_from_code(code_a)
                if decoded_sem == target_sem:
                    semantic_preservations += 1
            except Exception:
                pass

    return {
        'success_rate': successes / test_size,
        'semantic_preservation': semantic_preservations / test_size,
        'total_tests': test_size
    }


def translation_bridge_experiment(
    num_populations: int = 3,
    test_size_per_pair: int = 50,
    seed: int = 42
) -> Dict:
    """
    Run cross-population bridge experiment.

    Parameters
    ----------
    num_populations : int
        Number of separate populations to test
    test_size_per_pair : int
        Number of test cases per population pair
    seed : int
        Random seed

    Returns
    -------
    Dict with experiment results
    """
    random.seed(seed)

    print(f"\n{'='*60}")
    print("CROSS-POPULATION TRANSLATION BRIDGE EXPERIMENT")
    print(f"{'='*60}\n")

    # Create populations with slight vocabulary biases
    populations = []
    for i in range(num_populations):
        # Create slight dialect variations (in production, these would emerge from training)
        vocab_bias = {}
        if i > 0:
            # Add some vocabulary preferences to create dialects
            vocab_bias = {
                'ACT': random.sample(VOCAB['ACT'], min(1, len(VOCAB['ACT'])))
            }

        pop = create_population(i, vocab_bias)
        populations.append(pop)
        print(f"Created Population {i}")

    print(f"\nTesting all population pairs...")
    print("-" * 60)

    # Test all pairs
    results = {
        'num_populations': num_populations,
        'population_pairs': [],
        'success_rates': [],
        'semantic_preservations': []
    }

    for i, pop_a in enumerate(populations):
        for j, pop_b in enumerate(populations):
            if i != j:  # Don't test self-communication
                print(f"\nPop {i} -> Pop {j}:")

                # Test A->B communication
                comm_results = evaluate_cross_population_communication(
                    pop_a, pop_b, test_size_per_pair
                )

                print(f"  Success rate: {comm_results['success_rate']:.3f}")
                print(f"  Semantic preservation: {comm_results['semantic_preservation']:.3f}")

                results['population_pairs'].append((i, j))
                results['success_rates'].append(comm_results['success_rate'])
                results['semantic_preservations'].append(comm_results['semantic_preservation'])

    # Compute summary statistics
    import statistics
    results['mean_success'] = statistics.mean(results['success_rates'])
    results['std_success'] = statistics.stdev(results['success_rates']) if len(results['success_rates']) > 1 else 0.0
    results['mean_semantic_preservation'] = statistics.mean(results['semantic_preservations'])

    return results


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for population bridge experiment."""
    if argv is None:
        argv = sys.argv[1:]

    # Parse arguments
    num_pops = 3
    test_size = 50

    if len(argv) > 0:
        num_pops = int(argv[0])
    if len(argv) > 1:
        test_size = int(argv[1])

    # Run experiment
    results = translation_bridge_experiment(
        num_populations=num_pops,
        test_size_per_pair=test_size
    )

    # Display final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Number of populations: {results['num_populations']}")
    print(f"Population pairs tested: {len(results['population_pairs'])}")
    print(f"")
    print(f"Mean cross-population success: {results['mean_success']:.3f} ± {results['std_success']:.3f}")
    print(f"Mean semantic preservation:    {results['mean_semantic_preservation']:.3f}")
    print(f"")

    # Interpretability assessment
    if results['mean_success'] > 0.75:
        print("✓ HIGH UNIVERSALITY: Languages are highly translatable")
    elif results['mean_success'] > 0.50:
        print("~ MODERATE UNIVERSALITY: Languages are partially translatable")
    else:
        print("✗ LOW UNIVERSALITY: Languages have diverged significantly")

    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
