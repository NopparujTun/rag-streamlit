"""Configuration loader for the RAG application."""

import logging
from typing import Any, Dict

import yaml
import streamlit as st

logger = logging.getLogger(__name__)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration or halt the Streamlit app on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:
        st.error(f"🚨 Config Error: {exc}")
        st.stop()
        return {}
