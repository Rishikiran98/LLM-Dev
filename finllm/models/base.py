"""Common interface every sentiment model implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from finllm.data.schema import LABELS


class Prediction(BaseModel):
    """One sentiment prediction with a full probability distribution."""

    text: str
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

    @property
    def polarity(self) -> float:
        """Signed score in [-1, 1]: P(positive) - P(negative)."""
        return self.probabilities.get("positive", 0.0) - self.probabilities.get("negative", 0.0)

    @classmethod
    def from_probabilities(cls, text: str, probs: Sequence[float]) -> Prediction:
        if len(probs) != len(LABELS):
            raise ValueError(f"Expected {len(LABELS)} probabilities, got {len(probs)}")
        mapping = {label: float(p) for label, p in zip(LABELS, probs, strict=True)}
        best = max(mapping, key=mapping.__getitem__)
        return cls(text=text, label=best, confidence=mapping[best], probabilities=mapping)


@runtime_checkable
class SentimentModel(Protocol):
    """Anything that can score financial text.

    ``predict_proba`` returns one row per input in ``LABELS`` order
    (negative, neutral, positive).
    """

    name: str

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]: ...

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

    def save(self, directory: str) -> None: ...


class PredictMixin:
    """Default ``predict`` built on top of ``predict_proba``."""

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        rows = self.predict_proba(texts)  # type: ignore[attr-defined]
        return [Prediction.from_probabilities(t, row) for t, row in zip(texts, rows, strict=True)]
