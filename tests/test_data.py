import pandas as pd
import pytest

from finllm.data.loaders import load_csv, load_hf_dataset, load_sample
from finllm.data.preprocess import clean_text, deduplicate, prepare_dataset, split_dataset
from finllm.data.schema import LABELS, normalize_label


def test_sample_is_balanced_and_canonical(sample_frame):
    assert list(sample_frame.columns) == ["text", "label"]
    counts = sample_frame["label"].value_counts()
    assert set(counts.index) == set(LABELS)
    assert counts.min() == counts.max() == 40


@pytest.mark.parametrize(
    "raw,expected",
    [("Positive", "positive"), ("bearish", "negative"), (2, "positive"), ("neu", "neutral")],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_label("meh")


def test_load_csv_with_custom_columns(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"sentence": ["Profit up", "Profit down"], "y": ["pos", "neg"]}).to_csv(
        path, index=False
    )
    frame = load_csv(path, text_column="sentence", label_column="y")
    assert frame["label"].tolist() == ["positive", "negative"]


def test_load_csv_missing_column(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_csv(path)


def test_load_hf_unknown_name():
    with pytest.raises(KeyError):
        load_hf_dataset("not-a-dataset")


def test_clean_text_keeps_financial_tokens():
    out = clean_text("  $AAPL  rose 5% to $150 &amp; more  http://x.y/z ")
    assert out == "AAPL rose 5% to $150 & more"


def test_deduplicate_case_insensitive():
    frame = pd.DataFrame({"text": ["Up", "up ", "Down"], "label": ["positive"] * 3})
    assert len(deduplicate(frame)) == 2


def test_prepare_drops_short_rows():
    frame = pd.DataFrame({"text": ["ok", "long enough text"], "label": ["neutral", "neutral"]})
    assert prepare_dataset(frame)["text"].tolist() == ["long enough text"]


def test_split_is_stratified_and_deterministic(sample_frame):
    a_train, a_test = split_dataset(sample_frame, test_size=0.25, seed=1)
    b_train, b_test = split_dataset(sample_frame, test_size=0.25, seed=1)
    assert a_test["text"].tolist() == b_test["text"].tolist()
    assert len(a_test) == 30
    assert a_test["label"].value_counts().min() == 10
    assert not set(a_train["text"]) & set(a_test["text"])


def test_load_sample_matches_fixture(sample_frame):
    assert len(load_sample()) == len(sample_frame)
