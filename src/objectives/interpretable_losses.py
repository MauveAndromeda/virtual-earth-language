# src/objectives/interpretable_losses.py
"""
Interpretable Multi-Objective Loss Functions

Core contribution: Extended loss function that enforces interpretability:

J = α·Success + β·MI + γ·Topology 
    - λ₁·Length - λ₂·Entropy 
    + δ₁·Consistency + δ₂·Alignment + δ₃·Learnability
    + ε·AntiEncryption

Where interpretability terms (δ, ε) are the key innovation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import math
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import networkx as nx
from sklearn.metrics import mutual_info_score
from collections import Counter

@dataclass
class LossComponents:
    """Individual components of the interpretable loss function."""
    success: float = 0.0
    mutual_info: float = 0.0
    topology: float = 0.0
    length_penalty: float = 0.0
    entropy_penalty: float = 0.0
    consistency: float = 0.0
    alignment: float = 0.0
    learnability: float = 0.0
    anti_encryption: float = 0.0
    total: float = 0.0

class InterpretableLossFunction:
    """
    Multi-objective loss function optimizing for both communication success
    and interpretability constraints.
    """
    
    def __init__(
        self,
        # Standard EC weights
        alpha: float = 1.0,      # Task success
        beta: float = 0.6,       # Mutual information
        gamma: float = 0.4,      # Topological similarity
        lambda1: float = 0.15,   # Length penalty
        lambda2: float = 0.08,   # Entropy penalty
        
        # Interpretability weights (key innovation)
        delta1: float = 0.5,     # C↔E consistency
        delta2: float = 0.3,     # Slot alignment (CTC)
        delta3: float = 0.4,     # Few-shot learnability
        
        # Anti-encryption weights
        epsilon: float = 0.2,    # Anti-encryption
        public_listener_weight: float = 0.2,
        noise_robustness_weight: float = 0.1,
        minimal_edit_weight: float = 0.15,
        
        # Normalization parameters
        temperature: float = 1.0,
        moving_average_decay: float = 0.99
    ):
        
        # Store weights
        self.weights = {
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'lambda1': lambda1, 'lambda2': lambda2,
            'delta1': delta1, 'delta2': delta2, 'delta3': delta3,
            'epsilon': epsilon,
            'public_listener': public_listener_weight,
            'noise_robustness': noise_robustness_weight,
            'minimal_edit': minimal_edit_weight
        }
        
        self.temperature = temperature
        self.ma_decay = moving_average_decay
        
        # Initialize normalization statistics
        self.running_stats = {
            'success_mean': 0.5, 'success_std': 0.3,
            'mi_mean': 1.0, 'mi_std': 0.5,
            'topology_mean': 0.7, 'topology_std': 0.2,
            'consistency_mean': 0.8, 'consistency_std': 0.2
        }
        
        # Import dependencies
        try:
            from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM
            from explain.dual_channel import DUAL_CHANNEL_SYSTEM
            self.slot_system = ENHANCED_SLOT_SYSTEM
            self.dual_channel = DUAL_CHANNEL_SYSTEM
        except ImportError:
            self.slot_system = None
            self.dual_channel = None
    
    def compute_loss(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        listener_outputs: Dict[str, torch.Tensor], 
        semantics_batch: List[Dict[str, str]],
        messages_batch: List[str],
        targets_batch: List[int],
        predictions_batch: List[int],
        dual_channel_messages: Optional[List] = None
    ) -> Tuple[torch.Tensor, LossComponents]:
        """
        Compute complete interpretable loss with all components.
        
        Args:
            speaker_outputs: Dict with 'c_channel_logits', 'e_channel_text', etc.
            listener_outputs: Dict with 'choice_logits', 'attention_weights', etc.  
            semantics_batch: List of semantic dictionaries
            messages_batch: List of generated messages
            targets_batch: List of target indices
            predictions_batch: List of predicted indices
            dual_channel_messages: Optional list of DualChannelMessage objects
            
        Returns:
            total_loss: Combined loss tensor
            components: Individual loss components for analysis
        """
        
        batch_size = len(semantics_batch)
        components = LossComponents()
        
        # 1. SUCCESS RATE (α term)
        success_rate = self._compute_success_rate(targets_batch, predictions_batch)
        components.success = success_rate
        
        # 2. MUTUAL INFORMATION (β term) 
        mutual_info = self._compute_mutual_information(
            semantics_batch, messages_batch
        )
        components.mutual_info = mutual_info
        
        # 3. TOPOLOGICAL SIMILARITY (γ term)
        topology_score = self._compute_topological_similarity(
            semantics_batch, messages_batch
        )
        components.topology = topology_score
        
        # 4. LENGTH PENALTY (λ₁ term)
        length_penalty = self._compute_length_penalty(messages_batch)
        components.length_penalty = length_penalty
        
        # 5. ENTROPY PENALTY (λ₂ term) 
        entropy_penalty = self._compute_entropy_penalty(speaker_outputs)
        components.entropy_penalty = entropy_penalty
        
        # === INTERPRETABILITY TERMS (Key Innovation) ===
        
        # 6. C↔E CONSISTENCY (δ₁ term)
        if dual_channel_messages is not None:
            consistency_score = self._compute_consistency_score(dual_channel_messages)
        else:
            consistency_score = self._compute_consistency_fallback(
                speaker_outputs, semantics_batch
            )
        components.consistency = consistency_score
        
        # 7. SLOT ALIGNMENT (δ₂ term)
        alignment_score = self._compute_slot_alignment_score(
            speaker_outputs, semantics_batch
        )
        components.alignment = alignment_score
        
        # 8. LEARNABILITY (δ₃ term)
        learnability_score = self._compute_learnability_score(
            speaker_outputs, listener_outputs
        )
        components.learnability = learnability_score
        
        # 9. ANTI-ENCRYPTION (ε term)
        anti_encryption_score = self._compute_anti_encryption_score(
            speaker_outputs, semantics_batch
        )
        components.anti_encryption = anti_encryption_score
        
        # Combine all terms with weights
        total_loss = (
            self.weights['alpha'] * (1.0 - components.success) +           # Minimize failure
            -self.weights['beta'] * components.mutual_info +               # Maximize MI 
            -self.weights['gamma'] * components.topology +                 # Maximize topology
            self.weights['lambda1'] * components.length_penalty +          # Minimize length
            self.weights['lambda2'] * components.entropy_penalty +         # Minimize entropy
            -self.weights['delta1'] * components.consistency +             # Maximize consistency
            -self.weights['delta2'] * components.alignment +               # Maximize alignment  
            -self.weights['delta3'] * components.learnability +            # Maximize learnability
            -self.weights['epsilon'] * components.anti_encryption          # Maximize interpretability
        )
        
        components.total = float(total_loss) if isinstance(total_loss, torch.Tensor) else total_loss
        
        # Update running statistics
        self._update_running_stats(components)
        
        return torch.tensor(total_loss, requires_grad=True), components
    
    def _compute_success_rate(self, targets: List[int], predictions: List[int]) -> float:
        """Compute communication success rate."""
        if not targets or not predictions:
            return 0.0
        
        correct = sum(1 for t, p in zip(targets, predictions) if t == p)
        return correct / len(targets)
    
    def _compute_mutual_information(
        self, 
        semantics_batch: List[Dict[str, str]], 
        messages_batch: List[str]
    ) -> float:
        """Compute mutual information between semantics and messages."""
        if not semantics_batch or not messages_batch:
            return 0.0
        
        # Convert semantics to discrete codes for MI computation
        semantic_codes = []
        for sem in semantics_batch:
            # Create hashable representation
            code = tuple(sorted(sem.items()))
            semantic_codes.append(hash(code) % 1000)  # Discretize
        
        # Convert messages to discrete codes
        message_codes = [hash(msg) % 1000 for msg in messages_batch]
        
        # Compute mutual information
        try:
            mi = mutual_info_score(semantic_codes, message_codes)
            return float(mi)
        except:
            return 0.0
    
    def _compute_topological_similarity(
        self,
        semantics_batch: List[Dict[str, str]],
        messages_batch: List[str]
    ) -> float:
        """
        Compute topological similarity: semantic neighbors should have similar messages.
        
        Key insight: If two semantics are similar, their messages should be similar too.
        """
        if len(semantics_batch) < 2:
            return 1.0
        
        n = len(semantics_batch)
        topology_score = 0.0
        comparisons = 0
        
        for i in range(n):
            for j in range(i+1, min(i+5, n)):  # Compare with nearby examples
                # Semantic distance
                sem_dist = self._semantic_distance(semantics_batch[i], semantics_batch[j])
                
                # Message distance  
                msg_dist = self._message_distance(messages_batch[i], messages_batch[j])
                
                # Good topology: similar semantics → similar messages
                # Correlation coefficient between distances
                topology_score += 1.0 - abs(sem_dist - msg_dist)
                comparisons += 1
        
        return topology_score / max(comparisons, 1)
    
    def _compute_length_penalty(self, messages_batch: List[str]) -> float:
        """Penalize overly long messages."""
        if not messages_batch:
            return 0.0
        
        avg_length = np.mean([len(msg.split()) for msg in messages_batch])
        # Penalty grows quadratically for very long messages
        return (avg_length / 10.0) ** 2
    
    def _compute_entropy_penalty(self, speaker_outputs: Dict[str, torch.Tensor]) -> float:
        """Penalize high entropy (encourages decisive communication)."""
        if 'c_channel_logits' not in speaker_outputs:
            return 0.0
        
        logits = speaker_outputs['c_channel_logits']
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy per token, average over sequence and batch
        entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = torch.mean(entropy_per_token)
        
        return float(avg_entropy)
    
    def _compute_consistency_score(self, dual_channel_messages: List) -> float:
        """
        Compute C↔E consistency score (key interpretability metric).
        
        Measures how well the efficient code and explanation convey the same meaning.
        """
        if not dual_channel_messages:
            return 0.0
        
        consistency_scores = [msg.consistency_score for msg in dual_channel_messages]
        return float(np.mean(consistency_scores))
    
    def _compute_consistency_fallback(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        semantics_batch: List[Dict[str, str]]
    ) -> float:
        """Fallback consistency computation when dual channel messages unavailable."""
        # Simplified consistency check - can be enhanced
        if 'interpretability_metrics' in speaker_outputs:
            metrics = speaker_outputs['interpretability_metrics']
            if 'consistency_score' in metrics:
                return float(metrics['consistency_score'])
        
        return 0.8  # Default reasonable consistency
    
    def _compute_slot_alignment_score(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        semantics_batch: List[Dict[str, str]]
    ) -> float:
        """
        Compute slot alignment score using CTC-style monotonic alignment.
        
        Ensures that message positions correspond to semantic slots in order.
        """
        if 'attention_weights' not in speaker_outputs:
            return 0.8  # Default score if no attention available
        
        attention = speaker_outputs['attention_weights']  # [batch, seq_len, num_slots]
        
        # Check monotonic alignment property
        batch_size, seq_len, num_slots = attention.shape
        alignment_scores = []
        
        for b in range(min(batch_size, len(semantics_batch))):
            # Get attention weights for this example
            attn = attention[b]  # [seq_len, num_slots]
            
            # Compute alignment score (simplified CTC-style)
            monotonic_score = 0.0
            for t in range(seq_len - 1):
                for s in range(num_slots - 1):
                    # Check if attention progresses monotonically
                    current = attn[t, s]
                    next_time = attn[t + 1, s + 1] if s + 1 < num_slots else attn[t + 1, s]
                    
                    if next_time >= current - 0.1:  # Allow small decreases
                        monotonic_score += 1.0
            
            # Normalize by number of comparisons
            total_comparisons = (seq_len - 1) * (num_slots - 1)
            if total_comparisons > 0:
                alignment_scores.append(monotonic_score / total_comparisons)
        
        return float(np.mean(alignment_scores)) if alignment_scores else 0.8
    
    def _compute_learnability_score(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        listener_outputs: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute few-shot learnability score.
        
        Measures how easily new agents can learn the communication protocol.
        """
        # Use confidence and clarity metrics as proxies for learnability
        learnability = 0.0
        components = 0
        
        # Speaker clarity (low entropy = clear messages)
        if 'interpretability_metrics' in speaker_outputs:
            metrics = speaker_outputs['interpretability_metrics']
            if 'template_confidence' in metrics:
                learnability += metrics['template_confidence']
                components += 1
        
        # Listener confidence  
        if 'choice_confidence' in listener_outputs:
            learnability += float(listener_outputs['choice_confidence'])
            components += 1
        
        # Attention consistency (focused attention = interpretable)
        if 'attention_weights' in speaker_outputs:
            attention = speaker_outputs['attention_weights']
            # Measure attention concentration (1 - entropy)
            attn_probs = F.softmax(attention, dim=-1)
            attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
            concentration = 1.0 - torch.mean(attn_entropy) / math.log(attention.shape[-1])
            learnability += float(concentration)
            components += 1
        
        return learnability / max(components, 1)
    
    def _compute_anti_encryption_score(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        semantics_batch: List[Dict[str, str]]
    ) -> float:
        """
        Compute anti-encryption score (prevents private codes).
        
        Combines multiple measures:
        - Public listener decodability
        - Noise robustness  
        - Minimal edit distance properties
        """
        anti_encryption = 0.0
        components = 0
        
        # 1. Public listener decodability
        if 'interpretability_metrics' in speaker_outputs:
            metrics = speaker_outputs['interpretability_metrics']
            if 'public_decodability' in metrics:
                anti_encryption += metrics['public_decodability']
                components += 1
        
        # 2. Noise robustness (implicit - similar messages for similar semantics)
        if len(semantics_batch) >= 2:
            noise_robustness = self._estimate_noise_robustness(
                speaker_outputs, semantics_batch
            )
            anti_encryption += noise_robustness
            components += 1
        
        # 3. Anchor word usage (prevents arbitrary drift)
        if self.slot_system is not None:
            anchor_usage = self._compute_anchor_word_usage(semantics_batch)
            anti_encryption += anchor_usage
            components += 1
        
        return anti_encryption / max(components, 1)
    
    def _estimate_noise_robustness(
        self,
        speaker_outputs: Dict[str, torch.Tensor],
        semantics_batch: List[Dict[str, str]]
    ) -> float:
        """Estimate robustness to noise by checking similar inputs → similar outputs."""
        if 'c_channel_logits' not in speaker_outputs:
            return 0.5
        
        logits = speaker_outputs['c_channel_logits']
        
        # Find pairs of similar semantics
        robustness_scores = []
        for i, sem1 in enumerate(semantics_batch):
            for j, sem2 in enumerate(semantics_batch[i+1:], i+1):
                if i < logits.shape[0] and j < logits.shape[0]:
                    sem_similarity = 1.0 - self._semantic_distance(sem1, sem2)
                    
                    if sem_similarity > 0.7:  # Similar semantics
                        # Check if messages are also similar
                        msg_sim = F.cosine_similarity(
                            logits[i].flatten(), logits[j].flatten(), dim=0
                        )
                        robustness_scores.append(float(msg_sim))
        
        return float(np.mean(robustness_scores)) if robustness_scores else 0.5
    
    def _compute_anchor_word_usage(self, semantics_batch: List[Dict[str, str]]) -> float:
        """Compute usage rate of anchor words (prevents semantic drift)."""
        if self.slot_system is None:
            return 0.5
        
        anchor_usage = 0.0
        total_words = 0
        
        for semantics in semantics_batch:
            for slot_name, value in semantics.items():
                total_words += 1
                if (slot_name in self.slot_system.anchor_words and
                    value in self.slot_system.anchor_words[slot_name]):
                    anchor_usage += 1.0
        
        return anchor_usage / max(total_words, 1)
    
    def _semantic_distance(self, sem1: Dict[str, str], sem2: Dict[str, str]) -> float:
        """Compute normalized semantic distance between two semantic representations."""
        if self.slot_system is not None:
            return self.slot_system.semantic_distance(sem1, sem2)
        else:
            # Fallback implementation
            all_slots = set(sem1.keys()) | set(sem2.keys())
            if not all_slots:
                return 0.0
            
            differences = sum(1 for slot in all_slots if sem1.get(slot) != sem2.get(slot))
            return differences / len(all_slots)
    
    def _message_distance(self, msg1: str, msg2: str) -> float:
        """Compute normalized message distance.""" 
        if msg1 == msg2:
            return 0.0
        
        # Simple token-based distance
        tokens1 = set(msg1.split())
        tokens2 = set(msg2.split())
        
        if not tokens1 and not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    def _update_running_stats(self, components: LossComponents):
        """Update running statistics for normalization."""
        decay = self.ma_decay
        
        # Update means and stds with exponential moving average
        self.running_stats['success_mean'] = (
            decay * self.running_stats['success_mean'] + 
            (1 - decay) * components.success
        )
        
        self.running_stats['mi_mean'] = (
            decay * self.running_stats['mi_mean'] + 
            (1 - decay) * components.mutual_info
        )
        
        self.running_stats['topology_mean'] = (
            decay * self.running_stats['topology_mean'] + 
            (1 - decay) * components.topology
        )
        
        self.running_stats['consistency_mean'] = (
            decay * self.running_stats['consistency_mean'] + 
            (1 - decay) * components.consistency
        )
    
    def get_loss_breakdown(self, components: LossComponents) -> Dict[str, float]:
        """Get detailed breakdown of loss components for analysis."""
        return {
            'total_loss': components.total,
            'success_rate': components.success,
            'mutual_information': components.mutual_info,
            'topological_similarity': components.topology,
            'length_penalty': components.length_penalty,
            'entropy_penalty': components.entropy_penalty,
            'consistency_score': components.consistency,
            'alignment_score': components.alignment,
            'learnability_score': components.learnability,
            'anti_encryption_score': components.anti_encryption,
            
            # Weighted contributions
            'weighted_success': self.weights['alpha'] * (1.0 - components.success),
            'weighted_consistency': self.weights['delta1'] * components.consistency,
            'weighted_alignment': self.weights['delta2'] * components.alignment,
            'weighted_learnability': self.weights['delta3'] * components.learnability,
            'weighted_anti_encryption': self.weights['epsilon'] * components.anti_encryption
        }

# Global instance for easy access
INTERPRETABLE_LOSS_FUNCTION = InterpretableLossFunction()
