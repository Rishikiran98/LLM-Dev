"""Cleaning, de-duplication, and stratified splitting."""

from __future__ import annotations

import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from finllm.data.loaders import LABEL_COL, TEXT_COL

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TICKER_CASHTAG = re.compile(r"\$([A-Za-z]{1,6})\b")


def clean_text(text: str) -> str:
    """Light normalisation that keeps financial signal (numbers, %, $) intact."""
    text = html.unescape(str(text))
    text = _URL.sub(" ", text)
    text = _TICKER_CASHTAG.sub(r"\1", text)  # $AAPL -> AAPL
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip()


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate texts (case-insensitive), keeping the first label seen."""
    key = frame[TEXT_COL].str.lower().str.strip()
    return frame.loc[~key.duplicated()].reset_index(drop=True)


def prepare_dataset(frame: pd.DataFrame, *, min_chars: int = 5) -> pd.DataFrame:
    """Clean, drop empty/short rows, and de-duplicate."""
    out = frame.copy()
    out[TEXT_COL] = out[TEXT_COL].map(clean_text)
    out = out[out[TEXT_COL].str.len() >= min_chars]
    out = out.dropna(subset=[LABEL_COL])
    return deduplicate(out)


def split_dataset(
    frame: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. Returns ``(train, test)``."""
    train, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame[LABEL_COL]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
