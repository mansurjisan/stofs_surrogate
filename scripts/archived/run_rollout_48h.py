#!/usr/bin/env python3
"""
Run 48h rollout using saved model - minimal standalone script.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate/scripts')
os.chdir('/mnt/d/AI_4_STOFS/stofs_surrogate')

# Run the training script but skip to just the rollout portion
# by setting a flag

if __name__ == '__main__':
    # Just run the main training script - it will use the cached data and model
    # and only regenerate the rollout since we updated the plot function
    print("Running training script to regenerate 48h rollout...")
    print("(This will reload data and generate new rollout plot)")
    print()

    exec(open('scripts/train_midatlantic_with_forcing.py').read())
