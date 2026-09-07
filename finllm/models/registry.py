"""Save/load any sentiment model from a directory using its ``model.json`` metadata."""

from __future__ import annotations

import json
from pathlib import Path

from finllm.models.base import SentimentModel
from finllm.models.baseline import BaselineSentimentModel

META_NAME = "model.json"


def save_model(model: SentimentModel, directory: str | Path) -> Path:
    directory = Path(directory)
    model.save(str(directory))
    return directory


def load_model(directory: str | Path, **kwargs) -> SentimentModel:
    """Load a model saved by :func:`save_model`, dispatching on ``model.json``."""
    directory = Path(directory)
    meta_path = directory / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"No {META_NAME} in {directory}; is this a saved model?")
    kind = json.loads(meta_path.read_text(encoding="utf-8")).get("kind")
    if kind == "baseline":
        return BaselineSentimentModel.load(directory)
    if kind == "transformer":
        from finllm.models.transformer import TransformerSentimentModel

        return TransformerSentimentModel.load(directory, **kwargs)
    raise ValueError(f"Unknown model kind {kind!r} in {meta_path}")
