from typing import Dict, List

class EnhancedSlotSystem:
    """
    Enhanced Slot System that defines a vocabulary for each semantic slot (e.g. shape, color, position)
    and provides basic encoding and decoding utilities.
    """

    def __init__(self) -> None:
        # Define slot vocabularies
        self.slots: Dict[str, List[str]] = {
            "shape": ["CIRCLE", "SQUARE", "TRIANGLE", "PENTAGON"],
            "color": ["RED", "GREEN", "BLUE", "YELLOW"],
            "position": ["LEFT", "RIGHT", "TOP", "BOTTOM", "CENTER"],
        }

    def get_slot_vocab(self, slot_name: str) -> List[str]:
        """
        Return the vocabulary list for a given slot name.
        """
        return self.slots.get(slot_name, [])

    def encode_message(self, message: Dict[str, str]) -> List[int]:
        """
        Encode a message dictionary into a list of integer indices based on the slot vocabularies.
        """
        codes: List[int] = []
        for slot_name, value in message.items():
            vocab = self.get_slot_vocab(slot_name)
            try:
                index = vocab.index(value)
            except ValueError:
                index = -1
            codes.append(index)
        return codes

    def decode_message(self, codes: List[int]) -> Dict[str, str]:
        """
        Decode a list of integer indices back into a message dictionary using the slot vocabularies.
        """
        result: Dict[str, str] = {}
        for slot_name, index in zip(self.slots.keys(), codes):
            vocab = self.get_slot_vocab(slot_name)
            if 0 <= index < len(vocab):
                result[slot_name] = vocab[index]
            else:
                result[slot_name] = None
        return result

# Global instance of the enhanced slot system
ENHANCED_SLOT_SYSTEM = EnhancedSlotSystem()
