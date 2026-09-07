"""Typed configuration loaded from YAML with environment-variable overrides.

Precedence (highest first): explicit keyword overrides, ``FINLLM_*`` environment
variables, the YAML file, then the dataclass defaults below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "FINLLM_"


class DataSettings(BaseModel):
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = Field(0.2, gt=0, lt=1)
    random_seed: int = 42


class BaselineSettings(BaseModel):
    word_ngram_max: int = Field(2, ge=1)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_features: int = Field(50_000, ge=100)
    C: float = Field(4.0, gt=0)


class TransformerSettings(BaseModel):
    base_model: str = "ProsusAI/finbert"
    max_length: int = Field(128, ge=8)
    epochs: int = Field(3, ge=1)
    batch_size: int = Field(16, ge=1)
    learning_rate: float = Field(2e-5, gt=0)


class RiskSettings(BaseModel):
    sentiment_weight: float = Field(0.6, ge=0, le=1)
    lexicon_weight: float = Field(0.4, ge=0, le=1)


class RetrievalSettings(BaseModel):
    chunk_size: int = Field(400, ge=50)
    chunk_overlap: int = Field(60, ge=0)
    top_k: int = Field(5, ge=1)


class LLMSettings(BaseModel):
    model: str = "claude-opus-5"
    max_tokens: int = Field(4096, ge=256)
    effort: str = "medium"


class PathSettings(BaseModel):
    artifacts_dir: Path = Path("artifacts")


class Settings(BaseModel):
    data: DataSettings = DataSettings()
    baseline: BaselineSettings = BaselineSettings()
    transformer: TransformerSettings = TransformerSettings()
    risk: RiskSettings = RiskSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    paths: PathSettings = PathSettings()


def _coerce(value: str) -> Any:
    """Parse an environment-variable string with YAML rules so numbers stay numbers."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Turn ``FINLLM_SECTION__KEY=value`` variables into a nested dict."""
    result: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(raw)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    **overrides: dict[str, Any],
) -> Settings:
    """Load settings from ``path`` (or ``configs/default.yaml`` if present)."""
    data: dict[str, Any] = {}
    candidate = Path(path) if path else Path("configs/default.yaml")
    if candidate.exists():
        with open(candidate, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {candidate}")
    data = _deep_merge(data, _env_overrides(dict(os.environ if environ is None else environ)))
    data = _deep_merge(data, overrides)
    return Settings.model_validate(data)
