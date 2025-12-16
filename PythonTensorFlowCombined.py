#!/usr/bin/env python3
"""
PythonTensorFlowCombined.py

Implements the MLP architecture:
2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2

Supports 3 output modes:
A) regression   : 2 continuous outputs
B) softmax      : 2-class classification (mutually exclusive)
C) multilabel   : 2 independent probabilities
"""

from __future__ import annotations
import argparse
import numpy as np
import tensorflow as tf


def build_model(mode: str, lr: float = 1e-3) -> tf.keras.Model:
    """
    mode:
      - "regression"
      - "softmax"
      - "multilabel"
    """
    mode = mode.lower()
    if mode not in {"regression", "softmax", "multilabel"}:
        raise ValueError("mode must be one of: regression, s

