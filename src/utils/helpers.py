"""Shared text-processing and formatting utilities.

Functions here are pure (no side effects) and safe to call from any module.
"""

import re


def clean_markdown(text: str) -> str:
    """Remove horizontal-rule markers that break Streamlit markdown rendering.

    Strips standalone lines of ``---``, ``===``, or ``___`` (three or more
    repetitions) which PDF extraction sometimes produces and which Streamlit
    renders as unwanted ``<h1>`` / ``<h2>`` elements.

    Args:
        text: Raw markdown text to clean.

    Returns:
        Cleaned text with horizontal-rule artefacts removed.
    """
    if not text:
        return ""

    cleaned = re.sub(r"^[-=_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return cleaned.strip()


def format_time(seconds: float) -> str:
    """Format a duration in seconds into a human-readable Thai string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string like ``"1.23 วินาที"`` or ``"< 0.01 วินาที"`` for very
        small values.
    """
    if seconds < 0.01:
        return "< 0.01 วินาที"
    return f"{seconds:.2f} วินาที"