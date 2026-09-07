"""TF-IDF + logistic regression baseline.

All feature transformers live inside one scikit-learn ``Pipeline`` so they are
fit on the training split only, avoiding the leakage of fitting vectorizers on
the full dataset before splitting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from finllm.config import BaselineSettings
from finllm.data.schema import LABEL_TO_ID, LABELS
from finllm.models.base import PredictMixin

ARTIFACT_NAME = "baseline.joblib"
META_NAME = "model.json"


def build_pipeline(settings: BaselineSettings | None = None) -> Pipeline:
    cfg = settings or BaselineSettings()
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, cfg.word_ngram_max),
                    sublinear_tf=True,
                    min_df=1,
                    max_features=cfg.max_features,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(cfg.char_ngram_range),
                    sublinear_tf=True,
                    min_df=1,
                    max_features=cfg.max_features,
                    lowercase=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=cfg.C, max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("features", features), ("clf", classifier)])


class BaselineSentimentModel(PredictMixin):
    """Fast, dependency-light classifier; the reference point for the transformer."""

    name = "baseline"

    def __init__(self, settings: BaselineSettings | None = None) -> None:
        self.settings = settings or BaselineSettings()
        self.pipeline = build_pipeline(self.settings)
        self._fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> BaselineSentimentModel:
        y = np.array([LABEL_TO_ID[label] for label in labels])
        self.pipeline.fit(list(texts), y)
        self._fitted = True
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted; call fit() or load() first")
        raw = self.pipeline.predict_proba(list(texts))
        # Some classes may be missing from tiny training sets: map back to LABELS order.
        full = np.zeros((raw.shape[0], len(LABELS)))
        for col, class_id in enumerate(self.pipeline.classes_):
            full[:, int(class_id)] = raw[:, col]
        return full.tolist()

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / ARTIFACT_NAME)
        meta = {"kind": self.name, "labels": LABELS, "settings": self.settings.model_dump()}
        (directory / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> BaselineSentimentModel:
        directory = Path(directory)
        meta = json.loads((directory / META_NAME).read_text(encoding="utf-8"))
        model = cls(BaselineSettings.model_validate(meta.get("settings", {})))
        model.pipeline = joblib.load(directory / ARTIFACT_NAME)
        model._fitted = True
        return model
