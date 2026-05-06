"""Text normalization for Prompt Guard input."""

from __future__ import annotations

import re
import unicodedata


def normalize_for_detection(text: str) -> str:
    """Canonicalize text before passing to ML-based detectors.

    Removes common obfuscation techniques and normalizes unicode
    so that Prompt Guard sees a clean signal.
    """
    text = _normalize_unicode(text)
    text = _strip_zero_width_chars(text)
    text = _normalize_whitespace(text)
    return text.strip()


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _strip_zero_width_chars(text: str) -> str:
    zero_width = [
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\ufeff",  # zero width no-break space (BOM)
        "\u00ad",  # soft hyphen
    ]
    for char in zero_width:
        text = text.replace(char, "")
    return text


def _normalize_whitespace(text: str) -> str:
    # Collapse multiple spaces/newlines into single spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text