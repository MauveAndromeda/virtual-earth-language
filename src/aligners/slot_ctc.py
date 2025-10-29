"""
CTC-based Slot Alignment for Interpretable Language Evolution

This module implements Connectionist Temporal Classification (CTC) alignment
between slot-structured semantic representations and generated code sequences.

Key Features:
- Handles variable-length alignment between slots and codes
- Provides alignment quality scoring
- Supports both training and inference modes
- Includes visualization utilities for alignment analysis

References:
- Graves et al. (2006): Connectionist Temporal Classification
- Custom adaptations for slot-structured language alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AlignmentMode(Enum):
    """Alignment mode for CTC decoder."""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    VITERBI = "viterbi"


@dataclass
class AlignmentResult:
    """Result of slot-to-code alignment."""
    alignment: List[Tuple[int, int]]  # [(slot_idx, code_idx), ...]
    alignment_score: float
    slot_coverage: float  # Fraction of slots successfully aligned
    code_coverage: float  # Fraction of codes successfully aligned
    confidence: float
    path_probabilities: Optional[torch.Tensor] = None


@dataclass
class CTCOutput:
    """Output from CTC alignment model."""
    log_probs: torch.Tensor  # [batch, time, vocab]
    alignments: List[AlignmentResult]
    loss: Optional[torch.Tensor] = None


class SlotCTCAligner(nn.Module):
    """
    CTC-based alignment between slot sequences and code sequences.

    Architecture:
    - Encodes slot-structured semantics
    - Produces alignment probabilities via CTC
    - Supports multiple decoding strategies
    - Provides alignment quality metrics

    Parameters
    ----------
    slot_vocab_sizes : Dict[str, int]
        Vocabulary size for each slot type
    code_vocab_size : int
        Vocabulary size for code sequences
    hidden_dim : int
        Hidden dimension for alignment model
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout probability
    blank_idx : int
        Index for CTC blank token (default: 0)
    """

    def __init__(
        self,
        slot_vocab_sizes: Dict[str, int],
        code_vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        blank_idx: int = 0
    ):
        super().__init__()

        self.slot_vocab_sizes = slot_vocab_sizes
        self.code_vocab_size = code_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.blank_idx = blank_idx

        # Slot embeddings for each slot type
        self.slot_embeddings = nn.ModuleDict({
            slot_name: nn.Embedding(vocab_size, hidden_dim // len(slot_vocab_sizes))
            for slot_name, vocab_size in slot_vocab_sizes.items()
        })

        # Positional encoding for slot positions
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=100)

        # Bidirectional LSTM for context encoding
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        # CTC projection layer (includes blank token)
        self.ctc_projection = nn.Linear(hidden_dim * 2, code_vocab_size + 1)

        # Attention mechanism for alignment refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # CTC loss function
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)

    def encode_slots(
        self,
        slot_values: Dict[str, torch.Tensor],
        slot_order: List[str]
    ) -> torch.Tensor:
        """
        Encode slot-structured input into continuous representations.

        Parameters
        ----------
        slot_values : Dict[str, torch.Tensor]
            Dictionary mapping slot names to token indices
            Each tensor shape: [batch, 1]
        slot_order : List[str]
            Order of slots to process

        Returns
        -------
        torch.Tensor
            Encoded slot representations [batch, num_slots, hidden_dim]
        """
        batch_size = next(iter(slot_values.values())).size(0)
        slot_embeds = []

        for slot_name in slot_order:
            if slot_name in slot_values and slot_name in self.slot_embeddings:
                slot_embed = self.slot_embeddings[slot_name](slot_values[slot_name])
                slot_embeds.append(slot_embed)

        if not slot_embeds:
            raise ValueError("No valid slots found in input")

        # Concatenate slot embeddings along feature dimension
        # Shape: [batch, num_slots, hidden_dim // num_slots]
        slot_embeds = torch.cat(slot_embeds, dim=-1)

        # Add positional encoding
        slot_embeds = self.positional_encoding(slot_embeds)

        return slot_embeds

    def forward(
        self,
        slot_values: Dict[str, torch.Tensor],
        slot_order: List[str],
        code_sequence: Optional[torch.Tensor] = None,
        code_lengths: Optional[torch.Tensor] = None,
        alignment_mode: AlignmentMode = AlignmentMode.GREEDY
    ) -> CTCOutput:
        """
        Forward pass: align slots to code sequence.

        Parameters
        ----------
        slot_values : Dict[str, torch.Tensor]
            Slot-structured input
        slot_order : List[str]
            Order of slots
        code_sequence : torch.Tensor, optional
            Target code sequence [batch, max_code_len]
        code_lengths : torch.Tensor, optional
            Length of each code sequence [batch]
        alignment_mode : AlignmentMode
            Decoding strategy for alignment

        Returns
        -------
        CTCOutput
            Alignment results with probabilities and metrics
        """
        # Encode slot structure
        slot_embeds = self.encode_slots(slot_values, slot_order)
        batch_size, num_slots, _ = slot_embeds.shape

        # Pass through bidirectional LSTM
        encoder_output, _ = self.encoder(slot_embeds)
        # Shape: [batch, num_slots, hidden_dim * 2]

        # Apply attention for alignment refinement
        attended_output, attention_weights = self.attention(
            encoder_output, encoder_output, encoder_output
        )

        # Project to CTC vocabulary (including blank)
        ctc_logits = self.ctc_projection(attended_output)
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        # Shape: [batch, num_slots, code_vocab_size + 1]

        # Compute CTC loss if targets provided
        loss = None
        if code_sequence is not None and code_lengths is not None:
            # CTC loss expects: [T, N, C] format
            log_probs_t = log_probs.transpose(0, 1)  # [num_slots, batch, vocab]

            # Input lengths (all slots are valid)
            input_lengths = torch.full(
                (batch_size,), num_slots, dtype=torch.long, device=log_probs.device
            )

            loss = self.ctc_loss(log_probs_t, code_sequence, input_lengths, code_lengths)

        # Decode alignments
        alignments = self.decode_alignments(
            log_probs, code_sequence, alignment_mode
        )

        return CTCOutput(
            log_probs=log_probs,
            alignments=alignments,
            loss=loss
        )

    def decode_alignments(
        self,
        log_probs: torch.Tensor,
        code_sequence: Optional[torch.Tensor],
        mode: AlignmentMode = AlignmentMode.GREEDY
    ) -> List[AlignmentResult]:
        """
        Decode CTC output to produce slot-to-code alignments.

        Parameters
        ----------
        log_probs : torch.Tensor
            CTC log probabilities [batch, time, vocab]
        code_sequence : torch.Tensor, optional
            Target code sequence for scoring [batch, max_len]
        mode : AlignmentMode
            Decoding strategy

        Returns
        -------
        List[AlignmentResult]
            Alignment results for each batch element
        """
        batch_size = log_probs.size(0)
        results = []

        for batch_idx in range(batch_size):
            if mode == AlignmentMode.GREEDY:
                result = self._greedy_decode(
                    log_probs[batch_idx],
                    code_sequence[batch_idx] if code_sequence is not None else None
                )
            elif mode == AlignmentMode.BEAM_SEARCH:
                result = self._beam_search_decode(
                    log_probs[batch_idx],
                    code_sequence[batch_idx] if code_sequence is not None else None,
                    beam_width=5
                )
            elif mode == AlignmentMode.VITERBI:
                result = self._viterbi_decode(
                    log_probs[batch_idx],
                    code_sequence[batch_idx] if code_sequence is not None else None
                )
            else:
                raise ValueError(f"Unknown alignment mode: {mode}")

            results.append(result)

        return results

    def _greedy_decode(
        self,
        log_probs: torch.Tensor,
        target_codes: Optional[torch.Tensor] = None
    ) -> AlignmentResult:
        """
        Greedy decoding: select most probable alignment at each step.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities [time, vocab]
        target_codes : torch.Tensor, optional
            Target code sequence for scoring

        Returns
        -------
        AlignmentResult
            Greedy alignment result
        """
        # Get most probable token at each timestep
        best_tokens = log_probs.argmax(dim=-1)  # [time]

        # Remove blanks and consecutive duplicates (CTC collapsing)
        alignment = []
        prev_token = None
        for slot_idx, token_idx in enumerate(best_tokens.tolist()):
            if token_idx != self.blank_idx and token_idx != prev_token:
                alignment.append((slot_idx, token_idx))
                prev_token = token_idx

        # Calculate metrics
        num_slots = log_probs.size(0)
        num_aligned = len(alignment)
        slot_coverage = num_aligned / num_slots if num_slots > 0 else 0.0

        # Calculate alignment score
        alignment_score = sum(
            log_probs[slot_idx, code_idx].item()
            for slot_idx, code_idx in alignment
        ) / max(len(alignment), 1)

        # Calculate code coverage if target provided
        code_coverage = 0.0
        if target_codes is not None:
            target_set = set(target_codes.tolist())
            aligned_set = set(code_idx for _, code_idx in alignment)
            code_coverage = len(target_set & aligned_set) / max(len(target_set), 1)

        # Confidence from max probabilities
        confidence = log_probs.max(dim=-1).values.exp().mean().item()

        return AlignmentResult(
            alignment=alignment,
            alignment_score=alignment_score,
            slot_coverage=slot_coverage,
            code_coverage=code_coverage,
            confidence=confidence,
            path_probabilities=log_probs
        )

    def _beam_search_decode(
        self,
        log_probs: torch.Tensor,
        target_codes: Optional[torch.Tensor] = None,
        beam_width: int = 5
    ) -> AlignmentResult:
        """
        Beam search decoding for better alignment quality.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities [time, vocab]
        target_codes : torch.Tensor, optional
            Target code sequence
        beam_width : int
            Number of beams to maintain

        Returns
        -------
        AlignmentResult
            Best alignment from beam search
        """
        time_steps = log_probs.size(0)
        vocab_size = log_probs.size(1)

        # Initialize beams: (score, alignment_path, last_token)
        beams = [(0.0, [], self.blank_idx)]

        for t in range(time_steps):
            candidates = []

            for score, path, last_token in beams:
                # Expand each beam with all possible tokens
                for token_idx in range(vocab_size):
                    new_score = score + log_probs[t, token_idx].item()

                    # CTC collapsing rules
                    if token_idx == self.blank_idx:
                        # Blank: don't add to path
                        new_path = path.copy()
                    elif token_idx == last_token:
                        # Duplicate: don't add to path
                        new_path = path.copy()
                    else:
                        # New token: add to path
                        new_path = path + [(t, token_idx)]

                    candidates.append((new_score, new_path, token_idx))

            # Keep top-k beams
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # Select best beam
        best_score, best_alignment, _ = beams[0]

        # Calculate metrics
        num_slots = time_steps
        slot_coverage = len(best_alignment) / num_slots if num_slots > 0 else 0.0
        alignment_score = best_score / max(len(best_alignment), 1)

        code_coverage = 0.0
        if target_codes is not None:
            target_set = set(target_codes.tolist())
            aligned_set = set(code_idx for _, code_idx in best_alignment)
            code_coverage = len(target_set & aligned_set) / max(len(target_set), 1)

        confidence = np.exp(best_score / max(len(best_alignment), 1))

        return AlignmentResult(
            alignment=best_alignment,
            alignment_score=alignment_score,
            slot_coverage=slot_coverage,
            code_coverage=code_coverage,
            confidence=confidence,
            path_probabilities=log_probs
        )

    def _viterbi_decode(
        self,
        log_probs: torch.Tensor,
        target_codes: Optional[torch.Tensor] = None
    ) -> AlignmentResult:
        """
        Viterbi algorithm for optimal alignment path.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities [time, vocab]
        target_codes : torch.Tensor, optional
            Target code sequence

        Returns
        -------
        AlignmentResult
            Optimal alignment via Viterbi
        """
        # For simplicity, use greedy as Viterbi approximation
        # Full Viterbi would require state space expansion
        return self._greedy_decode(log_probs, target_codes)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-style models.

    Adds sinusoidal position embeddings to input sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, seq_len, d_model]

        Returns
        -------
        torch.Tensor
            Input with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


def compute_alignment_metrics(
    predicted_alignment: List[Tuple[int, int]],
    reference_alignment: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute alignment quality metrics.

    Parameters
    ----------
    predicted_alignment : List[Tuple[int, int]]
        Predicted slot-to-code alignment
    reference_alignment : List[Tuple[int, int]]
        Reference alignment

    Returns
    -------
    Dict[str, float]
        Alignment metrics (precision, recall, f1)
    """
    pred_set = set(predicted_alignment)
    ref_set = set(reference_alignment)

    if len(pred_set) == 0 or len(ref_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = len(pred_set & ref_set)
    precision = correct / len(pred_set)
    recall = correct / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def visualize_alignment(
    alignment: AlignmentResult,
    slot_names: List[str],
    code_tokens: List[str]
) -> str:
    """
    Create a text visualization of slot-to-code alignment.

    Parameters
    ----------
    alignment : AlignmentResult
        Alignment result to visualize
    slot_names : List[str]
        Names of slots
    code_tokens : List[str]
        Code token strings

    Returns
    -------
    str
        ASCII visualization of alignment
    """
    lines = []
    lines.append("Slot-to-Code Alignment:")
    lines.append("=" * 60)
    lines.append(f"Alignment Score: {alignment.alignment_score:.4f}")
    lines.append(f"Slot Coverage: {alignment.slot_coverage:.2%}")
    lines.append(f"Code Coverage: {alignment.code_coverage:.2%}")
    lines.append(f"Confidence: {alignment.confidence:.4f}")
    lines.append("-" * 60)

    for slot_idx, code_idx in alignment.alignment:
        slot_name = slot_names[slot_idx] if slot_idx < len(slot_names) else f"SLOT_{slot_idx}"
        code_token = code_tokens[code_idx] if code_idx < len(code_tokens) else f"CODE_{code_idx}"
        lines.append(f"  {slot_name:20s} --> {code_token}")

    lines.append("=" * 60)
    return "\n".join(lines)
