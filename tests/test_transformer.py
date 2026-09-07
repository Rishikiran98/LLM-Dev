"""Offline check of the transformer path using a tiny randomly initialised BERT.

Runs only when torch and transformers are installed (``pip install "finllm[transformer]"``).
No Hub download is needed, so it works in air-gapped CI.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")

from finllm.config import TransformerSettings  # noqa: E402
from finllm.data.schema import ID_TO_LABEL, LABEL_TO_ID, LABELS  # noqa: E402
from finllm.models.registry import load_model  # noqa: E402
from finllm.models.transformer import TransformerSentimentModel, fine_tune  # noqa: E402

pytestmark = pytest.mark.transformer


@pytest.fixture(scope="module")
def tiny_bert_dir(tmp_path_factory, sample_frame):
    """Build a tiny BERT checkpoint from scratch with a vocab drawn from the sample."""
    directory = tmp_path_factory.mktemp("tiny-bert")
    words = sorted({w.lower() for t in sample_frame["text"] for w in t.replace(".", "").split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (directory / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(str(directory / "vocab.txt"), do_lower_case=True)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    transformers.BertForSequenceClassification(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def test_from_pretrained_predicts_in_canonical_order(tiny_bert_dir):
    model = TransformerSentimentModel.from_pretrained(
        str(tiny_bert_dir), TransformerSettings(base_model=str(tiny_bert_dir), max_length=32)
    )
    preds = model.predict(["Revenue rose sharply.", "The meeting is on Monday."])
    assert [set(p.probabilities) for p in preds] == [set(LABELS)] * 2
    assert all(abs(sum(p.probabilities.values()) - 1) < 1e-5 for p in preds)


def test_fine_tune_and_reload(tiny_bert_dir, sample_frame, tmp_path):
    settings = TransformerSettings(
        base_model=str(tiny_bert_dir), max_length=32, epochs=1, batch_size=8, learning_rate=1e-3
    )
    texts, labels = sample_frame["text"].tolist()[:32], sample_frame["label"].tolist()[:32]
    out = tmp_path / "ft"
    model = fine_tune(
        texts,
        labels,
        settings=settings,
        eval_texts=texts[:8],
        eval_labels=labels[:8],
        output_dir=out,
    )
    assert (out / "model.json").exists()
    reloaded = load_model(out)
    assert reloaded.name == "transformer"
    a, b = model.predict_proba(texts[:2]), reloaded.predict_proba(texts[:2])
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a == pytest.approx(row_b, abs=1e-5)
