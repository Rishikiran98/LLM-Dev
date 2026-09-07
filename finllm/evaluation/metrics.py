"""Classification metrics and a small benchmark harness."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from finllm.data.schema import LABELS
from finllm.models.base import SentimentModel


class EvaluationResult(BaseModel):
    model: str
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]  # rows = true label, cols = predicted, LABELS order

    def summary(self) -> str:
        lines = [
            f"model: {self.model}  n={self.n}",
            f"accuracy: {self.accuracy:.4f}  macro-F1: {self.macro_f1:.4f}  "
            f"weighted-F1: {self.weighted_f1:.4f}",
            "per-class (precision / recall / f1 / support):",
        ]
        for label in LABELS:
            row = self.per_class[label]
            lines.append(
                f"  {label:<9} {row['precision']:.3f} / {row['recall']:.3f} / "
                f"{row['f1']:.3f} / {int(row['support'])}"
            )
        lines.append("confusion (rows=true, cols=pred, order=" + ",".join(LABELS) + "):")
        for label, row in zip(LABELS, self.confusion, strict=True):
            lines.append(f"  {label:<9} {row}")
        return "\n".join(lines)


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str], *, model_name: str = "model"
) -> EvaluationResult:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": float(s),
        }
        for label, p, r, f, s in zip(LABELS, precision, recall, f1, support, strict=True)
    }
    return EvaluationResult(
        model=model_name,
        n=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    )


def evaluate_model(
    model: SentimentModel, texts: Sequence[str], labels: Sequence[str]
) -> EvaluationResult:
    predictions = model.predict(list(texts))
    return evaluate_predictions(labels, [p.label for p in predictions], model_name=model.name)


class MajorityClassModel:
    """Sanity baseline: always predicts the most frequent training label."""

    name = "majority"

    def __init__(self, labels: Sequence[str]):
        counts = {label: 0 for label in LABELS}
        for label in labels:
            counts[label] += 1
        self.label = max(counts, key=counts.__getitem__)

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        row = [1.0 if label == self.label else 0.0 for label in LABELS]
        return [row for _ in texts]

    def predict(self, texts: Sequence[str]):
        from finllm.models.base import Prediction

        return [Prediction.from_probabilities(t, self.predict_proba([t])[0]) for t in texts]

    def save(self, directory: str) -> None:  # pragma: no cover - never persisted
        raise NotImplementedError
