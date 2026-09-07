"""Document-level risk assessment.

The score blends two signals:

* the share of negative sentiment across sentences, from any ``SentimentModel``;
* weighted hits from a categorised risk lexicon (credit, liquidity, legal, ...).

The result is a 0-100 score plus the evidence behind it, so an analyst can see
which sentences and which terms drove the number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

import yaml
from pydantic import BaseModel, Field

from finllm.config import RiskSettings
from finllm.models.base import Prediction, SentimentModel
from finllm.text import split_sentences


@dataclass(frozen=True)
class LexiconCategory:
    name: str
    weight: float
    terms: tuple[str, ...]


class RiskLexicon:
    """Whole-word, case-insensitive matcher over categorised risk terms."""

    def __init__(self, categories: Sequence[LexiconCategory]):
        self.categories = list(categories)
        self._patterns: dict[str, re.Pattern[str]] = {}
        for category in self.categories:
            escaped = sorted((re.escape(t) for t in category.terms), key=len, reverse=True)
            self._patterns[category.name] = re.compile(
                r"(?<![\w-])(?:" + "|".join(escaped) + r")(?![\w-])", flags=re.IGNORECASE
            )

    @classmethod
    def default(cls) -> RiskLexicon:
        with resources.as_file(resources.files("finllm.resources") / "risk_lexicon.yaml") as p:
            return cls.from_yaml(p)

    @classmethod
    def from_yaml(cls, path) -> RiskLexicon:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        categories = [
            LexiconCategory(
                name=name, weight=float(spec.get("weight", 1.0)), terms=tuple(spec["terms"])
            )
            for name, spec in raw.items()
        ]
        return cls(categories)

    def weight(self, category: str) -> float:
        return next(c.weight for c in self.categories if c.name == category)

    def find(self, text: str) -> dict[str, list[str]]:
        """Return ``{category: [matched terms...]}`` for one text."""
        hits: dict[str, list[str]] = {}
        for name, pattern in self._patterns.items():
            found = [m.group(0).lower() for m in pattern.finditer(text)]
            if found:
                hits[name] = found
        return hits


class SentenceRisk(BaseModel):
    text: str
    sentiment: str
    polarity: float
    risk_terms: dict[str, list[str]] = Field(default_factory=dict)
    score: float = Field(ge=0, le=1)


class RiskReport(BaseModel):
    score: float = Field(ge=0, le=100, description="0 = no risk signal, 100 = maximal")
    level: str
    sentiment_component: float = Field(ge=0, le=1)
    lexicon_component: float = Field(ge=0, le=1)
    category_scores: dict[str, float]
    sentence_count: int
    negative_share: float = Field(ge=0, le=1)
    flagged: list[SentenceRisk]

    @staticmethod
    def level_for(score: float) -> str:
        if score >= 70:
            return "high"
        if score >= 40:
            return "elevated"
        if score >= 15:
            return "moderate"
        return "low"


class RiskScorer:
    def __init__(
        self,
        model: SentimentModel,
        lexicon: RiskLexicon | None = None,
        settings: RiskSettings | None = None,
        *,
        flag_threshold: float = 0.35,
        top_flagged: int = 10,
    ) -> None:
        self.model = model
        self.lexicon = lexicon or RiskLexicon.default()
        self.settings = settings or RiskSettings()
        self.flag_threshold = flag_threshold
        self.top_flagged = top_flagged

    def score_sentences(self, sentences: Sequence[str]) -> list[SentenceRisk]:
        predictions: list[Prediction] = self.model.predict(list(sentences))
        rows: list[SentenceRisk] = []
        for pred in predictions:
            hits = self.lexicon.find(pred.text)
            lexicon_signal = min(
                1.0, sum(self.lexicon.weight(c) * len(t) for c, t in hits.items()) / 2.0
            )
            negative_prob = pred.probabilities.get("negative", 0.0)
            score = (
                self.settings.sentiment_weight * negative_prob
                + self.settings.lexicon_weight * lexicon_signal
            )
            rows.append(
                SentenceRisk(
                    text=pred.text,
                    sentiment=pred.label,
                    polarity=pred.polarity,
                    risk_terms=hits,
                    score=round(min(1.0, score), 4),
                )
            )
        return rows

    def assess(self, text: str) -> RiskReport:
        sentences = split_sentences(text)
        if not sentences:
            return RiskReport(
                score=0.0,
                level="low",
                sentiment_component=0.0,
                lexicon_component=0.0,
                category_scores={},
                sentence_count=0,
                negative_share=0.0,
                flagged=[],
            )
        rows = self.score_sentences(sentences)
        n = len(rows)
        negative_share = sum(r.sentiment == "negative" for r in rows) / n
        # Sentiment component: mean negativity derived from polarity, which is
        # smoother than the raw label share. Neutral sentences contribute nothing.
        sentiment_component = 0.0
        category_totals: dict[str, float] = {c.name: 0.0 for c in self.lexicon.categories}
        for row in rows:
            neg = (1 - row.polarity) / 2 if row.sentiment != "neutral" else 0.0
            sentiment_component += neg
            for category, terms in row.risk_terms.items():
                category_totals[category] += self.lexicon.weight(category) * len(terms)
        sentiment_component /= n
        # Lexicon component: weighted hits per sentence, saturating at one hit per sentence.
        lexicon_component = min(1.0, sum(category_totals.values()) / n)
        category_scores = {
            k: round(min(1.0, v / n), 4) for k, v in category_totals.items() if v > 0
        }
        blended = (
            self.settings.sentiment_weight * sentiment_component
            + self.settings.lexicon_weight * lexicon_component
        )
        score = round(100 * min(1.0, blended), 2)
        flagged = sorted(
            (r for r in rows if r.score >= self.flag_threshold),
            key=lambda r: r.score,
            reverse=True,
        )[: self.top_flagged]
        return RiskReport(
            score=score,
            level=RiskReport.level_for(score),
            sentiment_component=round(sentiment_component, 4),
            lexicon_component=round(lexicon_component, 4),
            category_scores=category_scores,
            sentence_count=n,
            negative_share=round(negative_share, 4),
            flagged=flagged,
        )
