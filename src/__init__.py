"""Virtual Earth: Interpretable Language Evolution

A breakthrough framework for emergent communication that produces 
human-readable languages instead of private codes.

Key Innovation: Interpretability-First Design
- Slot-based structured grammar
- Dual-channel (Code ↔ Explanation) system  
- Teaching and learning protocols
- Cross-population translation bridges

Ubuntu-optimized with CUDA support.
"""

__version__ = "2.0.0-interpretable"
__author__ = "MauveAndromeda"

import os
import torch

# System info with interpretability focus
if torch.cuda.is_available():
    print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    print("🧠 Ready for interpretable communication at scale!")
else:
    print("💻 Running on CPU")
    print("🧠 Interpretability framework ready!")

# Optimal threading for Ubuntu
if hasattr(torch, 'set_num_threads'):
    torch.set_num_threads(min(8, os.cpu_count()))

print(f"🌍 Virtual Earth v{__version__} - Interpretable Language Evolution")
print("📚 Documentation: https://github.com/MauveAndromeda/virtual-earth-language")
