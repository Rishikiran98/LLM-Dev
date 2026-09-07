"""Sentence segmentation and chunking used by risk scoring and retrieval."""

from __future__ import annotations

import re

# Split on sentence-final punctuation followed by whitespace and an uppercase/digit/quote.
# Guards common financial abbreviations so "Inc. reported" does not split.
_ABBREVIATIONS = ("inc", "ltd", "corp", "co", "llc", "plc", "u.s", "u.k", "no", "vs", "mr", "ms")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a small abbreviation guard."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    pieces = _SENTENCE_END.split(text)
    merged: list[str] = []
    for piece in pieces:
        if merged:
            last_word = merged[-1].rstrip(".").rsplit(" ", 1)[-1].lower()
            if last_word in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
        merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def chunk_text(text: str, *, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into word-bounded chunks of roughly ``chunk_size`` characters.

    Chunks end on sentence boundaries where possible and overlap by about
    ``overlap`` characters so retrieval does not lose context at the seams.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Carry over trailing sentences to form the overlap.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current = carried
            current_len = carried_len
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
