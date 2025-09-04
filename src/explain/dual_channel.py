# src/explain/dual_channel.py
"""
Dual-Channel Communication System

Core innovation: Every message has both:
- C-Channel: Efficient discrete codes for fast transmission  
- E-Channel: Human-readable explanations for interpretability
- Consistency constraint: C↔E bidirectional translation accuracy >95%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DualChannelMessage:
    """A message containing both efficient code and human explanation."""
    c_channel: List[int]          # Efficient discrete codes
    e_channel: str                # Human-readable explanation
    semantics: Dict[str, str]     # Original semantic content
    consistency_score: float = 0.0  # C↔E consistency measure

class ChannelEncoder(ABC):
    """Abstract base class for channel encoders."""
    
    @abstractmethod
    def encode(self, semantics: Dict[str, str]) -> Union[List[int], str]:
        pass
    
    @abstractmethod
    def decode(self, encoded: Union[List[int], str]) -> Dict[str, str]:
        pass

class CChannelEncoder(ChannelEncoder):
    """
    C-Channel: Efficient discrete code encoder
    Maps semantics to integer sequences optimized for transmission.
    """
    
    def __init__(self, vocab_size: int = 256, max_length: int = 12):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Import enhanced slot system
        from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM
        self.slot_system = ENHANCED_SLOT_SYSTEM
        
        # Create slot-to-token mapping
        self.slot_to_tokens = {}
        self.token_to_slot = {}
        token_id = 1  # Reserve 0 for PAD
        
        for slot_def in self.slot_system.slots:
            slot_tokens = {}
            vocab = self.slot_system.get_slot_vocabulary(slot_def.name)
            
            for word in vocab:
                if token_id < vocab_size - 1:  # Reserve last token for UNK
                    slot_tokens[word] = token_id
                    self.token_to_slot[token_id] = (slot_def.name, word)
                    token_id += 1
            
            self.slot_to_tokens[slot_def.name] = slot_tokens
        
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = vocab_size - 1
    
    def encode(self, semantics: Dict[str, str]) -> List[int]:
        """Encode semantics to integer sequence."""
        tokens = []
        
        # Follow slot order for consistency
        for slot_name in self.slot_system.slot_order:
            if slot_name in semantics:
                word = semantics[slot_name]
                token = self.slot_to_tokens.get(slot_name, {}).get(word, self.UNK_TOKEN)
                tokens.append(token)
            else:
                tokens.append(self.PAD_TOKEN)
        
        # Pad or truncate to max_length
        tokens = tokens[:self.max_length]
        tokens.extend([self.PAD_TOKEN] * (self.max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> Dict[str, str]:
        """Decode integer sequence back to semantics."""
        semantics = {}
        
        for i, token in enumerate(tokens):
            if token in self.token_to_slot:
                slot_name, word = self.token_to_slot[token]
                semantics[slot_name] = word
        
        return semantics

class EChannelEncoder(ChannelEncoder):
    """
    E-Channel: Human-readable explanation encoder
    Generates structured natural language descriptions.
    """
    
    def __init__(self, use_structured_format: bool = True):
        self.use_structured_format = use_structured_format
        
        from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM
        self.slot_system = ENHANCED_SLOT_SYSTEM
        
        # Natural language templates for different slots
        self.slot_templates = {
            "ACTION": "DO({action})",
            "OBJECT": "TARGET({object})", 
            "ATTRIBUTE": "WITH({attribute})",
            "LOCATION": "AT({location})",
            "MODIFIER": "MOD({modifier})"
        }
        
        # Alternative natural language patterns
        self.natural_templates = [
            "{ACTION} the {ATTRIBUTE} {OBJECT} {LOCATION}",
            "{ACTION} {MODIFIER} to {OBJECT} at {LOCATION}", 
            "Go {LOCATION} and {ACTION} {ATTRIBUTE} {OBJECT}",
            "{MODIFIER} {ACTION} {OBJECT} with {ATTRIBUTE} property"
        ]
    
    def encode(self, semantics: Dict[str, str]) -> str:
        """Encode semantics to human-readable explanation."""
        if self.use_structured_format:
            return self._encode_structured(semantics)
        else:
            return self._encode_natural(semantics)
    
    def _encode_structured(self, semantics: Dict[str, str]) -> str:
        """Generate structured format like: PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))"""
        parts = []
        
        for slot_name in self.slot_system.slot_order:
            if slot_name in semantics:
                template = self.slot_templates.get(slot_name, f"{slot_name}({{{slot_name.lower()}}})")
                part = template.format(**{slot_name.lower(): semantics[slot_name]})
                parts.append(part)
        
        return f"PLAN({', '.join(parts)})"
    
    def _encode_natural(self, semantics: Dict[str, str]) -> str:
        """Generate natural language description."""
        # Fill available template with semantics
        template = np.random.choice(self.natural_templates)
        
        # Replace placeholders, handle missing slots gracefully
        result = template
        for slot_name in self.slot_system.slot_order:
            placeholder = f"{{{slot_name}}}"
            if slot_name in semantics:
                result = result.replace(placeholder, semantics[slot_name].lower())
            else:
                # Remove unfilled placeholders and adjust grammar
                result = result.replace(placeholder + " ", "")
                result = result.replace(" " + placeholder, "")
                result = result.replace(placeholder, "")
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        return result.capitalize()
    
    def decode(self, explanation: str) -> Dict[str, str]:
        """Parse explanation back to semantics."""
        if explanation.startswith("PLAN("):
            return self._decode_structured(explanation)
        else:
            return self._decode_natural(explanation)
    
    def _decode_structured(self, explanation: str) -> Dict[str, str]:
        """Parse structured format back to semantics."""
        semantics = {}
        
        # Extract content between PLAN( and )
        match = re.match(r'PLAN\((.*)\)', explanation)
        if not match:
            return semantics
        
        content = match.group(1)
        
        # Parse each component
        for slot_name, template in self.slot_templates.items():
            # Create regex pattern from template
            pattern = template.replace(f"{{{slot_name.lower()}}}", r'([^),]+)')
            pattern = re.escape(pattern).replace(r'\([^),]+\)', r'([^),]+)')
            
            match = re.search(pattern, content)
            if match:
                semantics[slot_name] = match.group(1)
        
        return semantics
    
    def _decode_natural(self, explanation: str) -> Dict[str, str]:
        """Parse natural language back to semantics (heuristic-based)."""
        semantics = {}
        explanation = explanation.upper()
        
        # Look for slot vocabulary words
        for slot_def in self.slot_system.slots:
            vocab = self.slot_system.get_slot_vocabulary(slot_def.name)
            for word in vocab:
                if word in explanation:
                    semantics[slot_def.name] = word
                    break
        
        return semantics

class DualChannelSystem:
    """
    Complete dual-channel communication system with consistency enforcement.
    """
    
    def __init__(
        self, 
        vocab_size: int = 256,
        max_length: int = 12,
        consistency_threshold: float = 0.95,
        structured_explanations: bool = True
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.consistency_threshold = consistency_threshold
        
        # Initialize encoders
        self.c_encoder = CChannelEncoder(vocab_size, max_length)
        self.e_encoder = EChannelEncoder(structured_explanations)
        
        from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM
        self.slot_system = ENHANCED_SLOT_SYSTEM
        
        # Consistency tracking
        self.consistency_history = []
    
    def encode_message(self, semantics: Dict[str, str]) -> DualChannelMessage:
        """Encode semantics into dual-channel message."""
        # Validate input semantics
        valid, errors = self.slot_system.validate_semantics(semantics)
        if not valid:
            raise ValueError(f"Invalid semantics: {errors}")
        
        # Generate both channels
        c_channel = self.c_encoder.encode(semantics)
        e_channel = self.e_encoder.encode(semantics)
        
        # Test consistency
        consistency_score = self._measure_consistency(semantics, c_channel, e_channel)
        
        message = DualChannelMessage(
            c_channel=c_channel,
            e_channel=e_channel,
            semantics=semantics,
            consistency_score=consistency_score
        )
        
        return message
    
    def decode_c_channel(self, c_channel: List[int]) -> Dict[str, str]:
        """Decode C-channel to semantics."""
        return self.c_encoder.decode(c_channel)
    
    def decode_e_channel(self, e_channel: str) -> Dict[str, str]:
        """Decode E-channel to semantics.""" 
        return self.e_encoder.decode(e_channel)
    
    def _measure_consistency(
        self, 
        original_semantics: Dict[str, str], 
        c_channel: List[int], 
        e_channel: str
    ) -> float:
        """Measure C↔E consistency score."""
        
        # C → semantics → E → semantics roundtrip
        c_to_sem = self.c_encoder.decode(c_channel)
        e_from_c_sem = self.e_encoder.encode(c_to_sem)
        final_sem_from_e = self.e_encoder.decode(e_from_c_sem)
        
        # E → semantics → C → semantics roundtrip  
        e_to_sem = self.e_encoder.decode(e_channel)
        c_from_e_sem = self.c_encoder.encode(e_to_sem)
        final_sem_from_c = self.c_encoder.decode(c_from_e_sem)
        
        # Calculate semantic similarity scores
        c_consistency = 1.0 - self.slot_system.semantic_distance(original_semantics, final_sem_from_e)
        e_consistency = 1.0 - self.slot_system.semantic_distance(original_semantics, final_sem_from_c)
        
        # Combined consistency score
        consistency = (c_consistency + e_consistency) / 2.0
        
        # Track history
        self.consistency_history.append(consistency)
        
        return consistency
    
    def get_consistency_stats(self) -> Dict[str, float]:
        """Get consistency statistics."""
        if not self.consistency_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        history = np.array(self.consistency_history)
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
            "above_threshold": float(np.mean(history >= self.consistency_threshold))
        }
    
    def repair_inconsistency(
        self, 
        message: DualChannelMessage,
        prioritize_channel: str = "c"
    ) -> DualChannelMessage:
        """
        Repair inconsistent messages by regenerating the lower-priority channel.
        """
        if message.consistency_score >= self.consistency_threshold:
            return message  # Already consistent
        
        if prioritize_channel == "c":
            # Trust C-channel, regenerate E-channel
            semantics = self.c_encoder.decode(message.c_channel)
            new_e_channel = self.e_encoder.encode(semantics)
            
            return DualChannelMessage(
                c_channel=message.c_channel,
                e_channel=new_e_channel,
                semantics=semantics,
                consistency_score=self._measure_consistency(
                    semantics, message.c_channel, new_e_channel
                )
            )
        else:
            # Trust E-channel, regenerate C-channel  
            semantics = self.e_encoder.decode(message.e_channel)
            new_c_channel = self.c_encoder.encode(semantics)
            
            return DualChannelMessage(
                c_channel=new_c_channel,
                e_channel=message.e_channel,
                semantics=semantics,
                consistency_score=self._measure_consistency(
                    semantics, new_c_channel, message.e_channel
                )
            )
    
    def translate_between_populations(
        self, 
        message: DualChannelMessage,
        target_system: 'DualChannelSystem'
    ) -> DualChannelMessage:
        """
        Translate message between different population communication systems.
        Uses E-channel as universal intermediate representation.
        """
        
        # Extract semantics via E-channel (more robust to vocabulary differences)
        bridge_semantics = self.e_encoder.decode(message.e_channel)
        
        # Generate message in target system
        try:
            target_message = target_system.encode_message(bridge_semantics)
            return target_message
        except ValueError:
            # Fallback: use available semantic components
            valid_semantics = {}
            for slot_name, value in bridge_semantics.items():
                target_vocab = target_system.slot_system.get_slot_vocabulary(slot_name)
                if value in target_vocab:
                    valid_semantics[slot_name] = value
            
            if valid_semantics:
                return target_system.encode_message(valid_semantics)
            else:
                raise ValueError("Cannot translate message - no compatible semantics")

# Global instance
DUAL_CHANNEL_SYSTEM = DualChannelSystem()

# Legacy compatibility functions
def enhanced_code_from_sem(sem: Dict[str, str]) -> List[int]:
    """Enhanced version returning C-channel codes."""
    message = DUAL_CHANNEL_SYSTEM.encode_message(sem)
    return message.c_channel

def enhanced_explain_from_sem(sem: Dict[str, str]) -> str:
    """Enhanced version returning E-channel explanation."""
    message = DUAL_CHANNEL_SYSTEM.encode_message(sem)
    return message.e_channel

def enhanced_sem_from_code(code: List[int]) -> Dict[str, str]:
    """Enhanced version decoding from C-channel."""
    return DUAL_CHANNEL_SYSTEM.decode_c_channel(code)
