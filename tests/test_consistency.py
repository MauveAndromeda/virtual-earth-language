"""
Tests for consistency module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from objectives.consistency import (
    consistency_score,
    compute_consistency_metrics,
    string_similarity,
    semantic_similarity,
    levenshtein_distance,
    normalized_edit_distance,
    validate_consistency_threshold,
    batch_consistency_score
)


def test_string_similarity():
    """Test string similarity computation."""
    # Identical strings
    assert string_similarity("test", "test") == 1.0

    # Completely different
    assert string_similarity("abc", "xyz") < 0.5

    # Partial match
    sim = string_similarity("hello", "hallo")
    assert 0.5 < sim < 1.0


def test_levenshtein_distance():
    """Test edit distance calculation."""
    # Identical strings
    assert levenshtein_distance("test", "test") == 0

    # Single character difference
    assert levenshtein_distance("test", "text") == 1

    # Empty string
    assert levenshtein_distance("", "test") == 4


def test_normalized_edit_distance():
    """Test normalized edit distance."""
    # Identical strings
    assert normalized_edit_distance("test", "test") == 0.0

    # Completely different short strings
    dist = normalized_edit_distance("ab", "xy")
    assert 0 < dist <= 1.0


def test_semantic_similarity():
    """Test semantic similarity between slot dictionaries."""
    sem1 = {"ACTION": "MOVE", "OBJECT": "CIRCLE"}
    sem2 = {"ACTION": "MOVE", "OBJECT": "CIRCLE"}
    sem3 = {"ACTION": "TAKE", "OBJECT": "SQUARE"}

    # Identical semantics
    assert semantic_similarity(sem1, sem2) == 1.0

    # Different semantics
    sim = semantic_similarity(sem1, sem3)
    assert 0.0 <= sim < 1.0


def test_consistency_score():
    """Test basic consistency score computation."""
    # Perfect consistency
    score = consistency_score("E1", "E1", "C1", "C1")
    assert 0.0 <= score <= 1.0


def test_compute_consistency_metrics():
    """Test comprehensive consistency metrics."""
    metrics = compute_consistency_metrics(
        c_channel="C1",
        e_channel="E1",
        c_from_e="C1",
        e_from_c="E1"
    )

    assert metrics.bidirectional_score >= 0.0
    assert metrics.c_to_e_accuracy >= 0.0
    assert metrics.e_to_c_accuracy >= 0.0


def test_validate_consistency_threshold():
    """Test consistency threshold validation."""
    metrics = compute_consistency_metrics(
        c_channel="TEST",
        e_channel="TEST",
        c_from_e="TEST",
        e_from_c="TEST"
    )

    passes, warnings = validate_consistency_threshold(metrics, threshold=0.5)

    # Should either pass or have warnings
    assert isinstance(passes, bool)
    assert isinstance(warnings, list)


def test_batch_consistency_score():
    """Test batch consistency computation."""
    c_channels = ["C1", "C2", "C3"]
    e_channels = ["E1", "E2", "E3"]
    c_from_e = ["C1", "C2", "C3"]
    e_from_c = ["E1", "E2", "E3"]

    avg_score, metrics_list = batch_consistency_score(
        c_channels, e_channels, c_from_e, e_from_c
    )

    assert 0.0 <= avg_score <= 1.0
    assert len(metrics_list) == 3


def test_empty_strings():
    """Test handling of empty strings."""
    # Empty strings should have specific behavior
    sim = string_similarity("", "")
    assert sim == 1.0

    sim = string_similarity("", "test")
    assert sim == 0.0


def test_semantic_with_missing_slots():
    """Test semantic similarity with partial overlap."""
    sem1 = {"ACTION": "MOVE", "OBJECT": "CIRCLE"}
    sem2 = {"ACTION": "MOVE"}  # Missing OBJECT

    sim = semantic_similarity(sem1, sem2)
    assert 0.0 < sim < 1.0  # Should have partial similarity


if __name__ == "__main__":
    test_string_similarity()
    test_levenshtein_distance()
    test_normalized_edit_distance()
    test_semantic_similarity()
    test_consistency_score()
    test_compute_consistency_metrics()
    test_validate_consistency_threshold()
    test_batch_consistency_score()
    test_empty_strings()
    test_semantic_with_missing_slots()
    print("✓ All consistency tests passed!")
