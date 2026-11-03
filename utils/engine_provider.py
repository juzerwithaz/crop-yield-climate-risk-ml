"""
engine_provider.py
Cached singleton provider for the V2 Crop Intelligence Inference Engine.
"""

import os
import sys

# Cross-platform compatibility shim for unpickling Cython loss modules
if "_loss" not in sys.modules:
    try:
        import sklearn._loss._loss as _loss_mod
        sys.modules["_loss"] = _loss_mod
    except (ImportError, AttributeError):
        try:
            import sklearn._loss as _loss_mod
            sys.modules["_loss"] = _loss_mod
        except (ImportError, AttributeError):
            pass

import streamlit as st
from inference_v2 import CropYieldInferenceEngineV2, ValidationError, InferenceError

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_artifacts_v2")

@st.cache_resource(show_spinner="Loading V2 Crop Intelligence Models & Artifacts...")
def get_inference_engine(artifacts_dir: str = DEFAULT_ARTIFACTS_DIR) -> CropYieldInferenceEngineV2:
    """
    Instantiates and returns the cached CropYieldInferenceEngineV2.
    Ensures that ML models and metadata are loaded once into memory.
    """
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"V2 Model artifacts directory not found: {artifacts_dir}")
    return CropYieldInferenceEngineV2(artifacts_dir=artifacts_dir)

