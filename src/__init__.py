"""
MS Experimentation & Causal Inference

A comprehensive framework for A/B testing, causal inference, and uplift modeling.
"""

__version__ = "1.0.0"
__author__ = "Amitabh"

from . import ab_test
from . import causal
from . import uplift_model
from . import utils

__all__ = ["ab_test", "causal", "uplift_model", "utils"]
