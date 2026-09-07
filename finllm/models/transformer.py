"""Fine-tuning and inference with a Hugging Face transformer (optional extra).

Install with ``pip install "finllm[transformer]"``. The default base model is
FinBERT, already pre-trained on financial text, so a few epochs on a labelled
dataset is enough to adapt it to a new domain or label scheme.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from finllm.config import TransformerSettings
from finllm.data.schema import ID_TO_LABEL, LABEL_TO_ID, LABELS
from finllm.models.base import PredictMixin

META_NAME = "model.json"


def _require_transformers():
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'Transformer models need torch and transformers: pip install "finllm[transformer]"'
        ) from exc
    return torch, transformers


class TransformerSentimentModel(PredictMixin):
    """Sequence-classification transformer aligned to the canonical label set."""

    name = "transformer"

    def __init__(self, model, tokenizer, settings: TransformerSettings, device: str | None = None):
        torch, _ = _require_transformers()
        self.settings = settings
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ inference
    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        torch, _ = _require_transformers()
        out: list[list[float]] = []
        batch_size = self.settings.batch_size
        # Map the model's own label order onto LABELS order.
        order = self._label_order()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.settings.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.extend(probs[:, order].tolist())
        return out

    def _label_order(self) -> list[int]:
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        missing = [label for label in LABELS if label not in label2id]
        if missing:
            raise ValueError(
                f"Model labels {sorted(label2id)} do not cover the canonical set {LABELS}"
            )
        return [label2id[label] for label in LABELS]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        meta = {"kind": self.name, "labels": LABELS, "settings": self.settings.model_dump()}
        (directory / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> TransformerSentimentModel:
        _, transformers = _require_transformers()
        directory = Path(directory)
        meta_path = directory / META_NAME
        settings = TransformerSettings()
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            settings = TransformerSettings.model_validate(meta.get("settings", {}))
        tokenizer = transformers.AutoTokenizer.from_pretrained(directory)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(directory)
        return cls(model, tokenizer, settings, device=device)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        settings: TransformerSettings | None = None,
        device: str | None = None,
    ) -> TransformerSentimentModel:
        """Load a Hub checkpoint (e.g. ``ProsusAI/finbert``) without fine-tuning."""
        _, transformers = _require_transformers()
        settings = settings or TransformerSettings(base_model=name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(name_or_path)
        return cls(model, tokenizer, settings, device=device)


def fine_tune(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    *,
    settings: TransformerSettings | None = None,
    eval_texts: Sequence[str] | None = None,
    eval_labels: Sequence[str] | None = None,
    output_dir: str | Path = "artifacts/transformer",
    seed: int = 42,
) -> TransformerSentimentModel:
    """Fine-tune ``settings.base_model`` on labelled text and return the wrapped model."""
    torch, transformers = _require_transformers()
    from datasets import Dataset  # type: ignore[import-not-found]

    cfg = settings or TransformerSettings()
    transformers.set_seed(seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    def to_dataset(texts: Sequence[str], labels: Sequence[str]) -> Dataset:
        ds = Dataset.from_dict(
            {"text": list(texts), "label": [LABEL_TO_ID[label] for label in labels]}
        )
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=cfg.max_length),
            batched=True,
        )

    train_ds = to_dataset(train_texts, train_labels)
    eval_ds = (
        to_dataset(eval_texts, eval_labels)
        if eval_texts is not None and eval_labels is not None
        else None
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = float((preds == labels).mean())
        f1s = []
        for class_id in range(len(LABELS)):
            tp = int(((preds == class_id) & (labels == class_id)).sum())
            fp = int(((preds == class_id) & (labels != class_id)).sum())
            fn = int(((preds != class_id) & (labels == class_id)).sum())
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 0.0)
        return {"accuracy": accuracy, "macro_f1": float(np.mean(f1s))}

    args = transformers.TrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=seed,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    trainer.train()
    wrapped = TransformerSentimentModel(model, tokenizer, cfg)
    wrapped.save(output_dir)
    return wrapped
