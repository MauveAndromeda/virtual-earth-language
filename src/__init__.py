"""Virtual Earth Language Evolution

A framework for studying emergent language evolution in multi-agent systems
with geographic and demographic constraints.

Optimized for Ubuntu/Linux systems with CUDA support.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

import os
import torch

# Auto-detect GPU on Ubuntu
if torch.cuda.is_available():
    print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
else:
    print("💻 Running on CPU")

# Set optimal threading for Ubuntu
if hasattr(torch, 'set_num_threads'):
    torch.set_num_threads(min(8, os.cpu_count()))
