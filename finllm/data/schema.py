"""Canonical label set shared by every model and dataset loader."""

from __future__ import annotations

from enum import Enum


class Label(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


LABELS: list[str] = [label.value for label in Label]
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

# Aliases seen across public financial sentiment datasets.
_ALIASES: dict[str, str] = {
    "neg": "negative",
    "bearish": "negative",
    "0": "negative",
    "neu": "neutral",
    "none": "neutral",
    "1": "neutral",
    "pos": "positive",
    "bullish": "positive",
    "2": "positive",
}


def normalize_label(value: object) -> str:
    """Map dataset-specific label spellings onto the canonical label set."""
    text = str(value).strip().lower()
    text = _ALIASES.get(text, text)
    if text not in LABEL_TO_ID:
        raise ValueError(f"Unknown sentiment label {value!r}; expected one of {LABELS}")
    return text
