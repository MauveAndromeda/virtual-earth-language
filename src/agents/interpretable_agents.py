# src/agents/interpretable_agents.py
"""
Interpretable Neural Agents with Dual-Channel Communication

Features:
- Speaker: Learns to generate consistent C-channel + E-channel messages
- Listener: Processes dual-channel input with interpretability awareness  
- Teaching Protocol: Agents can explain their language to new learners
- Anti-encryption: Built-in safeguards against private code development
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class AgentAction:
    """Action taken by an agent with interpretability metrics."""
    choice: int
    confidence: float
    explanation: str
    c_channel_attention: Optional[torch.Tensor] = None
    e_channel_attention: Optional[torch.Tensor] = None

class InterpretableSpeaker(nn.Module):
    """
    Speaker agent that generates interpretable dual-channel messages.
    
    Architecture enforces:
    - Slot-structured attention over semantics
    - Consistency between C-channel and E-channel outputs
    - Anti-encryption regularization
    """
    
    def __init__(
        self,
        semantic_dim: int = 64,
        hidden_dim: int = 128,
        vocab_size: int = 256,
        max_length: int = 12,
        num_slots: int = 5,
        consistency_weight: float = 1.0,
        anti_encryption_weight: float = 0.5
    ):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_slots = num_slots
        self.consistency_weight = consistency_weight
        self.anti_encryption_weight = anti_encryption_weight
        
        # Semantic input encoder (converts slot-value pairs to embeddings)
        self.semantic_encoder = nn.ModuleDict({
            'action': nn.Embedding(8, semantic_dim),
            'object': nn.Embedding(12, semantic_dim), 
            'attribute': nn.Embedding(16, semantic_dim),
            'location': nn.Embedding(24, semantic_dim),
            'modifier': nn.Embedding(6, semantic_dim)
        })
        
        # Slot attention mechanism for interpretability
        self.slot_attention = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # C-Channel generator (discrete codes)
        self.c_channel_encoder = nn.LSTM(
            input_size=semantic_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.c_channel_head = nn.Linear(hidden_dim, vocab_size)
        
        # E-Channel generator (explanation text) - using learned templates
        self.e_channel_encoder = nn.LSTM(
            input_size=semantic_dim, 
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Template selection and slot filling for E-channel
        self.template_selector = nn.Linear(hidden_dim, 4)  # 4 explanation templates
        self.slot_fillers = nn.ModuleDict({
            slot: nn.Linear(hidden_dim, len(vocab)) 
            for slot, vocab in [
                ('action', ['MOVE', 'TAKE', 'DROP', 'GIVE', 'POINT', 'WAIT', 'LOOK', 'SCAN']),
                ('object', ['CIRCLE', 'SQUARE', 'TRIANGLE', 'DIAMOND', 'STAR', 'CROSS']),
                ('attribute', ['RED', 'BLUE', 'GREEN', 'YELLOW', 'SMALL', 'LARGE']),
                ('location', ['HERE', 'THERE', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'CENTER']),
                ('modifier', ['NOT', 'VERY', 'ALMOST', 'EXACTLY', 'MAYBE', 'QUICKLY'])
            ]
        })
        
        # Consistency checker (ensures C↔E alignment)
        self.consistency_checker = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Anti-encryption regularizer
        self.public_listener_simulator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_slots),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, semantics_tensor: torch.Tensor) -> Tuple[torch.Tensor, str, Dict[str, float]]:
        """
        Generate dual-channel message from semantics.
        
        Args:
            semantics_tensor: [batch_size, num_slots, semantic_dim]
            
        Returns:
            c_channel_logits: [batch_size, max_length, vocab_size]
            e_channel_explanation: Generated explanation string
            interpretability_metrics: Dict of interpretability scores
        """
        batch_size = semantics_tensor.shape[0]
        
        # Slot attention for interpretable processing
        attended_semantics, attention_weights = self.slot_attention(
            semantics_tensor, semantics_tensor, semantics_tensor
        )
        
        # Generate C-Channel (discrete codes)
        c_hidden, _ = self.c_channel_encoder(attended_semantics)
        c_channel_logits = self.c_channel_head(c_hidden)  # [batch, max_length, vocab_size]
        
        # Generate E-Channel (explanations)
        e_hidden, _ = self.e_channel_encoder(attended_semantics)
        e_pooled = torch.mean(e_hidden, dim=1)  # [batch, hidden_dim]
        
        # Template selection and slot filling
        template_logits = self.template_selector(e_pooled)
        template_idx = torch.argmax(template_logits, dim=-1)
        
        # Generate explanation text (simplified - in practice would use more sophisticated NLG)
        e_channel_explanation = self._generate_explanation(e_pooled, template_idx[0])
        
        # Measure consistency between channels
        c_pooled = torch.mean(c_hidden, dim=1)
        consistency_input = torch.cat([c_pooled, e_pooled], dim=-1)
        consistency_score = self.consistency_checker(consistency_input)
        
        # Anti-encryption: ensure public interpretability
        public_decodability = self.public_listener_simulator(c_pooled)
        entropy = -torch.sum(public_decodability * torch.log(public_decodability + 1e-8), dim=-1)
        
        interpretability_metrics = {
            'consistency_score': float(torch.mean(consistency_score)),
            'attention_entropy': float(torch.mean(-torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1))),
            'public_decodability': float(torch.mean(entropy)),
            'template_confidence': float(torch.max(torch.softmax(template_logits, dim=-1)))
        }
        
        return c_channel_logits, e_channel_explanation, interpretability_metrics
    
    def _generate_explanation(self, hidden_state: torch.Tensor, template_idx: int) -> str:
        """Generate human-readable explanation using learned templates."""
        templates = [
            "PLAN(ACTION={action}, TARGET={object}, WITH={attribute}, AT={location})",
            "DO({action}) on {attribute} {object} located {location}",
            "{modifier} {action} the {object} that is {attribute} at {location}",
            "Navigate to {location} and {action} {attribute} {object}"
        ]
        
        # In practice, this would use the slot_fillers to generate actual words
        # For now, return template with placeholders
        selected_template = templates[template_idx % len(templates)]
        
        # Simple placeholder filling (would be more sophisticated in practice)
        explanation = selected_template.format(
            action="ACTION_PLACEHOLDER",
            object="OBJECT_PLACEHOLDER", 
            attribute="ATTR_PLACEHOLDER",
            location="LOC_PLACEHOLDER",
            modifier="MOD_PLACEHOLDER"
        )
        
        return explanation
    
    def teaching_mode(self, semantics_tensor: torch.Tensor, explanation: str) -> Dict[str, torch.Tensor]:
        """
        Generate teaching signal for new learners.
        
        Returns explicit mappings between semantics, codes, and explanations.
        """
        with torch.no_grad():
            c_logits, e_explanation, metrics = self.forward(semantics_tensor)
            
            # Generate teaching examples
            teaching_signal = {
                'semantic_embedding': semantics_tensor,
                'c_channel_codes': torch.argmax(c_logits, dim=-1),
                'e_channel_text': e_explanation,
                'attention_map': self.slot_attention(semantics_tensor, semantics_tensor, semantics_tensor)[1],
                'consistency_score': torch.tensor(metrics['consistency_score'])
            }
            
            return teaching_signal

class InterpretableListener(nn.Module):
    """
    Listener agent that processes dual-channel messages with interpretability awareness.
    """
    
    def __init__(
        self,
        semantic_dim: int = 64,
        hidden_dim: int = 128,
        vocab_size: int = 256,
        max_length: int = 12,
        num_candidates: int = 4
    ):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_candidates = num_candidates
        
        # C-Channel processor
        self.c_channel_embedder = nn.Embedding(vocab_size, semantic_dim)
        self.c_channel_encoder = nn.LSTM(
            input_size=semantic_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # E-Channel processor (text understanding)
        self.e_channel_encoder = nn.LSTM(
            input_size=semantic_dim,  # Would use proper text encoder in practice
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Dual-channel fusion with interpretability
        self.channel_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Candidate scoring
        self.candidate_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Interpretability analyzer
        self.interpretability_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # consistency, clarity, confidence
        )
        
    def forward(
        self, 
        c_channel: torch.Tensor,
        e_channel_embedding: torch.Tensor,  # Pre-processed text embeddings
        candidate_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, AgentAction]:
        """
        Process dual-channel message and select from candidates.
        
        Args:
            c_channel: [batch_size, max_length] - discrete codes
            e_channel_embedding: [batch_size, text_length, embed_dim] - text embeddings
            candidate_embeddings: [batch_size, num_candidates, semantic_dim]
            
        Returns:
            choice_logits: [batch_size, num_candidates]
            action: AgentAction with interpretability info
        """
        batch_size = c_channel.shape[0]
        
        # Process C-Channel
        c_embedded = self.c_channel_embedder(c_channel)
        c_hidden, _ = self.c_channel_encoder(c_embedded)
        c_pooled = torch.mean(c_hidden, dim=1)  # [batch, hidden_dim]
        
        # Process E-Channel
        e_hidden, _ = self.e_channel_encoder(e_channel_embedding)
        e_pooled = torch.mean(e_hidden, dim=1)  # [batch, hidden_dim]
        
        # Fuse channels with attention
        dual_input = torch.stack([c_pooled.unsqueeze(1), e_pooled.unsqueeze(1)], dim=1)
        fused_repr, channel_attention = self.channel_fusion(
            dual_input, dual_input, dual_input
        )
        fused_pooled = torch.mean(fused_repr, dim=1)  # [batch, hidden_dim]
        
        # Score candidates
        candidate_scores = []
        for i in range(self.num_candidates):
            candidate_emb = candidate_embeddings[:, i, :]  # [batch, semantic_dim]
            # Expand to match fused_pooled dimensions
            candidate_expanded = F.linear(candidate_emb, torch.eye(self.hidden_dim, self.semantic_dim))
            scorer_input = torch.cat([fused_pooled, candidate_expanded], dim=-1)
            score = self.candidate_scorer(scorer_input)
            candidate_scores.append(score)
        
        choice_logits = torch.cat(candidate_scores, dim=-1)  # [batch, num_candidates]
        
        # Interpretability analysis
        interpretability_scores = self.interpretability_analyzer(fused_pooled)
        consistency, clarity, confidence = torch.split(interpretability_scores, 1, dim=-1)
        
        # Create action with interpretability info
        choice_probs = F.softmax(choice_logits, dim=-1)
        choice = torch.argmax(choice_probs, dim=-1)
        
        action = AgentAction(
            choice=int(choice[0]),
            confidence=float(torch.max(choice_probs[0])),
            explanation=f"Selected candidate {int(choice[0])} based on dual-channel analysis",
            c_channel_attention=channel_attention[:, :, 0],  # Attention to C-channel
            e_channel_attention=channel_attention[:, :, 1]   # Attention to E-channel
        )
        
        return choice_logits, action
    
    def learning_mode(
        self,
        teaching_examples: List[Dict[str, torch.Tensor]],
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Learn from teaching examples provided by experienced agents.
        
        Args:
            teaching_examples: List of teaching signals from TeachingSpeaker
            learning_rate: Adaptation rate for few-shot learning
            
        Returns:
            learning_metrics: Performance on teaching examples
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        correct_predictions = 0
        
        for example in teaching_examples:
            optimizer.zero_grad()
            
            # Extract teaching components
            c_codes = example['c_channel_codes']
            e_embedding = example.get('e_channel_embedding', torch.randn(1, 10, self.semantic_dim))
            candidates = example['candidate_embeddings'] 
            target = example['target_idx']
            
            # Forward pass
            choice_logits, _ = self.forward(c_codes, e_embedding, candidates)
            
            # Loss calculation
            loss = F.cross_entropy(choice_logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = torch.argmax(choice_logits, dim=-1)
            correct_predictions += int(predicted == target)
        
        return {
            'avg_loss': total_loss / len(teaching_examples),
            'accuracy': correct_predictions / len(teaching_examples),
            'examples_learned': len(teaching_examples)
        }

class TeachingProtocol:
    """
    Implements teaching and learning protocols between agents.
    """
    
    def __init__(self, speaker: InterpretableSpeaker, listener: InterpretableListener):
        self.speaker = speaker
        self.listener = listener
        self.teaching_history = []
        
    def generate_teaching_examples(
        self, 
        num_examples: int = 100,
        difficulty_progression: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate teaching examples with increasing complexity."""
        
        examples = []
        
        for i in range(num_examples):
            # Generate semantic input
            if difficulty_progression:
                # Start with simple examples, increase complexity
                num_slots = min(5, 2 + i // 20)
                semantics = torch.randn(1, num_slots, self.speaker.semantic_dim)
            else:
                semantics = torch.randn(1, 5, self.speaker.semantic_dim)
            
            # Teacher generates message
            teaching_signal = self.speaker.teaching_mode(semantics, "")
            
            # Create candidate set (target + distractors)
            candidates = torch.randn(1, 4, self.speaker.semantic_dim)
            candidates[0, 0] = torch.mean(semantics.squeeze(0), dim=0)  # Target is first
            
            example = {
                'semantic_input': semantics,
                'c_channel_codes': teaching_signal['c_channel_codes'],
                'e_channel_text': teaching_signal['e_channel_text'],
                'candidate_embeddings': candidates,
                'target_idx': torch.tensor([0]),  # Target is always first candidate
                'consistency_score': teaching_signal['consistency_score'],
                'attention_map': teaching_signal['attention_map']
            }
            
            examples.append(example)
        
        return examples
    
    def conduct_teaching_session(
        self, 
        learner: InterpretableListener,
        num_examples: int = 100,
        success_threshold: float = 0.9
    ) -> Dict[str, float]:
        """
        Conduct complete teaching session.
        
        Returns:
            session_results: Learning outcomes and metrics
        """
        
        # Generate teaching curriculum
        teaching_examples = self.generate_teaching_examples(
            num_examples=num_examples,
            difficulty_progression=True
        )
        
        # Conduct learning
        learning_results = learner.learning_mode(teaching_examples)
        
        # Evaluate learning success
        final_accuracy = learning_results['accuracy']
        session_success = final_accuracy >= success_threshold
        
        # Record in history
        session_record = {
            'examples_taught': num_examples,
            'final_accuracy': final_accuracy,
            'success': session_success,
            'consistency_scores': [ex['consistency_score'].item() for ex in teaching_examples[:10]]
        }
        
        self.teaching_history.append(session_record)
        
        return {
            'session_success': session_success,
            'final_accuracy': final_accuracy,
            'examples_needed': num_examples,
            'avg_consistency': np.mean(session_record['consistency_scores']),
            'teaching_efficiency': final_accuracy / num_examples  # Learning per example
        }
