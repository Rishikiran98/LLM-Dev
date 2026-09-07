"""Dataset loaders that all return a DataFrame with ``text`` and ``label`` columns."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from finllm.data.schema import normalize_label

TEXT_COL = "text"
LABEL_COL = "label"


@dataclass(frozen=True)
class HFDatasetSpec:
    """How to pull a Hugging Face Hub dataset into the canonical schema."""

    path: str
    config: str | None
    split: str
    text_field: str
    label_field: str
    label_names: tuple[str, ...] | None = None  # for integer-labelled datasets


# Public financial sentiment datasets. Check each dataset's license before
# using it outside research: financial_phrasebank is CC BY-NC-SA 4.0.
HF_DATASETS: dict[str, HFDatasetSpec] = {
    "financial_phrasebank": HFDatasetSpec(
        path="takala/financial_phrasebank",
        config="sentences_allagree",
        split="train",
        text_field="sentence",
        label_field="label",
        label_names=("negative", "neutral", "positive"),
    ),
    "twitter_financial_news": HFDatasetSpec(
        path="zeroshot/twitter-financial-news-sentiment",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_names=("bearish", "bullish", "neutral"),
    ),
}


def _to_canonical(frame: pd.DataFrame, text_field: str, label_field: str) -> pd.DataFrame:
    if text_field not in frame.columns or label_field not in frame.columns:
        raise KeyError(
            f"Expected columns {text_field!r} and {label_field!r}; found {list(frame.columns)}"
        )
    out = pd.DataFrame(
        {
            TEXT_COL: frame[text_field].astype("string").fillna("").astype(str),
            LABEL_COL: frame[label_field].map(normalize_label),
        }
    )
    return out.reset_index(drop=True)


def load_csv(
    path: str | Path, *, text_column: str = TEXT_COL, label_column: str = LABEL_COL
) -> pd.DataFrame:
    """Load a labelled CSV. Labels are normalized to negative/neutral/positive."""
    frame = pd.read_csv(path)
    return _to_canonical(frame, text_column, label_column)


def load_sample() -> pd.DataFrame:
    """Load the small synthetic dataset bundled with the package (120 sentences)."""
    with resources.as_file(
        resources.files("finllm.resources") / "sample_financial_sentiment.csv"
    ) as csv_path:
        return load_csv(csv_path)


def load_hf_dataset(name: str, *, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download a registered Hugging Face dataset. Requires ``pip install datasets``."""
    if name not in HF_DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; registered: {sorted(HF_DATASETS)}")
    spec = HF_DATASETS[name]
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Loading Hub datasets needs the 'datasets' package: pip install 'finllm[transformer]'"
        ) from exc

    ds = load_dataset(spec.path, spec.config, split=spec.split, cache_dir=cache_dir)
    frame = ds.to_pandas()
    if spec.label_names is not None and pd.api.types.is_integer_dtype(frame[spec.label_field]):
        frame[spec.label_field] = frame[spec.label_field].map(lambda i: spec.label_names[int(i)])
    return _to_canonical(frame, spec.text_field, spec.label_field)


def load_any(source: str | Path, **kwargs: str) -> pd.DataFrame:
    """Resolve ``sample``, a registered Hub dataset name, or a CSV path."""
    source_str = str(source)
    if source_str == "sample":
        return load_sample()
    if source_str in HF_DATASETS:
        return load_hf_dataset(source_str)
    return load_csv(source, **kwargs)
