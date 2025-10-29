"""
C↔E Consistency Scoring Module

This module implements comprehensive consistency checking between C-channel
(efficient codes) and E-channel (human-readable explanations) in dual-channel
communication systems.

Key Metrics:
1. Character-level consistency (edit distance)
2. AST-level consistency (structural similarity)
3. Semantic-level consistency (slot-value agreement)
4. Bidirectional translation accuracy

Consistency Target: >95% for interpretability guarantee
"""

import re
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher
import logging


logger = logging.getLogger(__name__)


@dataclass
class ConsistencyMetrics:
    """
    Comprehensive consistency metrics for C↔E channels.

    Attributes
    ----------
    bidirectional_score : float
        Overall bidirectional consistency (0-1)
    c_to_e_accuracy : float
        C→E translation accuracy
    e_to_c_accuracy : float
        E→C translation accuracy
    character_similarity : float
        Character-level similarity
    semantic_agreement : float
        Semantic slot-value agreement
    ast_similarity : float
        AST structural similarity
    confidence : float
        Confidence in consistency measurement
    """
    bidirectional_score: float
    c_to_e_accuracy: float
    e_to_c_accuracy: float
    character_similarity: float
    semantic_agreement: float
    ast_similarity: float
    confidence: float = 1.0


def consistency_score(e_from_c: str, e_gt: str, c_from_e: str, c_gt: str) -> float:
    """
    Compute basic C↔E consistency score.

    This is a simplified interface for backwards compatibility.
    For detailed metrics, use compute_consistency_metrics().

    Parameters
    ----------
    e_from_c : str
        Explanation generated from code
    e_gt : str
        Ground truth explanation
    c_from_e : str
        Code generated from explanation
    c_gt : str
        Ground truth code

    Returns
    -------
    float
        Consistency score (0-1)
    """
    metrics = compute_consistency_metrics(
        c_channel=c_gt,
        e_channel=e_gt,
        c_from_e=c_from_e,
        e_from_c=e_from_c
    )
    return metrics.bidirectional_score


def compute_consistency_metrics(
    c_channel: str,
    e_channel: str,
    c_from_e: Optional[str] = None,
    e_from_c: Optional[str] = None,
    semantics_c: Optional[Dict[str, str]] = None,
    semantics_e: Optional[Dict[str, str]] = None
) -> ConsistencyMetrics:
    """
    Compute comprehensive C↔E consistency metrics.

    Parameters
    ----------
    c_channel : str
        Original C-channel code
    e_channel : str
        Original E-channel explanation
    c_from_e : str, optional
        Code reconstructed from explanation
    e_from_c : str, optional
        Explanation reconstructed from code
    semantics_c : Dict[str, str], optional
        Semantic slots extracted from C-channel
    semantics_e : Dict[str, str], optional
        Semantic slots extracted from E-channel

    Returns
    -------
    ConsistencyMetrics
        Comprehensive consistency metrics
    """
    # C→E accuracy
    c_to_e_acc = 0.0
    if e_from_c is not None:
        c_to_e_acc = string_similarity(e_from_c, e_channel)

    # E→C accuracy
    e_to_c_acc = 0.0
    if c_from_e is not None:
        e_to_c_acc = string_similarity(c_from_e, c_channel)

    # Character-level similarity
    char_sim = string_similarity(c_channel, e_channel)

    # Semantic agreement
    sem_agreement = 1.0
    if semantics_c is not None and semantics_e is not None:
        sem_agreement = semantic_similarity(semantics_c, semantics_e)

    # AST similarity (if parseable)
    ast_sim = 0.0
    try:
        # Try to parse both channels as AST
        from explain.ast_parser import parse_explanation

        result_e = parse_explanation(e_channel)
        if e_from_c:
            result_e_from_c = parse_explanation(e_from_c)
            if result_e.parse_success and result_e_from_c.parse_success:
                ast_sim = compare_semantics(result_e.semantics, result_e_from_c.semantics)
    except Exception as e:
        logger.debug(f"AST parsing failed: {e}")
        ast_sim = 0.5  # Default moderate similarity

    # Compute overall bidirectional score
    bidirectional = (c_to_e_acc + e_to_c_acc) / 2.0 if (c_from_e and e_from_c) else 0.5

    # Weighted combination
    overall_score = (
        0.35 * bidirectional +
        0.25 * sem_agreement +
        0.25 * ast_sim +
        0.15 * char_sim
    )

    return ConsistencyMetrics(
        bidirectional_score=overall_score,
        c_to_e_accuracy=c_to_e_acc,
        e_to_c_accuracy=e_to_c_acc,
        character_similarity=char_sim,
        semantic_agreement=sem_agreement,
        ast_similarity=ast_sim,
        confidence=0.9
    )


def string_similarity(s1: str, s2: str) -> float:
    """
    Compute string similarity using normalized edit distance.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    float
        Similarity score (0-1)
    """
    if not s1 or not s2:
        return 0.0 if s1 != s2 else 1.0

    # Use SequenceMatcher for efficient similarity computation
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def semantic_similarity(sem1: Dict[str, str], sem2: Dict[str, str]) -> float:
    """
    Compute semantic similarity between two slot-value dictionaries.

    Parameters
    ----------
    sem1 : Dict[str, str]
        First semantic representation
    sem2 : Dict[str, str]
        Second semantic representation

    Returns
    -------
    float
        Similarity score (0-1)
    """
    if not sem1 or not sem2:
        return 0.0 if sem1 != sem2 else 1.0

    # Find common slots
    common_slots = set(sem1.keys()) & set(sem2.keys())
    all_slots = set(sem1.keys()) | set(sem2.keys())

    if not all_slots:
        return 1.0

    # Slot coverage
    slot_coverage = len(common_slots) / len(all_slots)

    # Value agreement for common slots
    value_agreement = 0.0
    if common_slots:
        matching_values = sum(
            1 for slot in common_slots if sem1[slot] == sem2[slot]
        )
        value_agreement = matching_values / len(common_slots)

    # Weighted combination
    similarity = 0.4 * slot_coverage + 0.6 * value_agreement

    return similarity


def compare_semantics(sem1: Dict[str, str], sem2: Dict[str, str]) -> float:
    """
    Compare two semantic dictionaries for consistency.

    This is an alias for semantic_similarity for backwards compatibility.

    Parameters
    ----------
    sem1 : Dict[str, str]
        First semantics
    sem2 : Dict[str, str]
        Second semantics

    Returns
    -------
    float
        Consistency score (0-1)
    """
    return semantic_similarity(sem1, sem2)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    int
        Edit distance (number of operations needed to transform s1 to s2)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Compute normalized edit distance (0=identical, 1=completely different).

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    float
        Normalized edit distance (0-1)
    """
    if not s1 and not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    return distance / max_len


def token_overlap(s1: str, s2: str, delimiter: str = "|") -> float:
    """
    Compute token overlap between two strings.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string
    delimiter : str
        Token delimiter

    Returns
    -------
    float
        Token overlap score (0-1)
    """
    tokens1 = set(s1.split(delimiter))
    tokens2 = set(s2.split(delimiter))

    if not tokens1 or not tokens2:
        return 0.0 if tokens1 != tokens2 else 1.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def validate_consistency_threshold(
    metrics: ConsistencyMetrics,
    threshold: float = 0.95
) -> Tuple[bool, List[str]]:
    """
    Validate that consistency metrics meet the required threshold.

    Parameters
    ----------
    metrics : ConsistencyMetrics
        Computed consistency metrics
    threshold : float
        Required consistency threshold (default 0.95 for interpretability)

    Returns
    -------
    Tuple[bool, List[str]]
        (passes_threshold, warning_messages)
    """
    warnings = []

    if metrics.bidirectional_score < threshold:
        warnings.append(
            f"Bidirectional consistency {metrics.bidirectional_score:.2%} "
            f"below threshold {threshold:.2%}"
        )

    if metrics.c_to_e_accuracy < threshold:
        warnings.append(
            f"C→E accuracy {metrics.c_to_e_accuracy:.2%} below threshold"
        )

    if metrics.e_to_c_accuracy < threshold:
        warnings.append(
            f"E→C accuracy {metrics.e_to_c_accuracy:.2%} below threshold"
        )

    if metrics.semantic_agreement < threshold:
        warnings.append(
            f"Semantic agreement {metrics.semantic_agreement:.2%} below threshold"
        )

    passes = len(warnings) == 0
    return passes, warnings


def consistency_loss(
    c_channel: str,
    e_channel: str,
    c_from_e: str,
    e_from_c: str
) -> float:
    """
    Compute consistency loss for training.

    Loss is 0 when consistency is perfect, increases as consistency degrades.

    Parameters
    ----------
    c_channel : str
        Original C-channel
    e_channel : str
        Original E-channel
    c_from_e : str
        C reconstructed from E
    e_from_c : str
        E reconstructed from C

    Returns
    -------
    float
        Consistency loss (lower is better)
    """
    metrics = compute_consistency_metrics(
        c_channel=c_channel,
        e_channel=e_channel,
        c_from_e=c_from_e,
        e_from_c=e_from_c
    )

    # Loss is 1 - consistency (perfect consistency = 0 loss)
    loss = 1.0 - metrics.bidirectional_score

    return loss


def batch_consistency_score(
    c_channels: List[str],
    e_channels: List[str],
    c_from_e_list: List[str],
    e_from_c_list: List[str]
) -> Tuple[float, List[ConsistencyMetrics]]:
    """
    Compute consistency scores for a batch of channel pairs.

    Parameters
    ----------
    c_channels : List[str]
        List of C-channels
    e_channels : List[str]
        List of E-channels
    c_from_e_list : List[str]
        List of reconstructed C-channels
    e_from_c_list : List[str]
        List of reconstructed E-channels

    Returns
    -------
    Tuple[float, List[ConsistencyMetrics]]
        (average_score, individual_metrics)
    """
    if len(c_channels) != len(e_channels):
        raise ValueError("C and E channel lists must have same length")

    metrics_list = []
    for c, e, c_from_e, e_from_c in zip(c_channels, e_channels, c_from_e_list, e_from_c_list):
        metrics = compute_consistency_metrics(
            c_channel=c,
            e_channel=e,
            c_from_e=c_from_e,
            e_from_c=e_from_c
        )
        metrics_list.append(metrics)

    # Compute average
    avg_score = np.mean([m.bidirectional_score for m in metrics_list])

    return float(avg_score), metrics_list
