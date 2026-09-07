import pytest

from finllm.text import chunk_text, split_sentences


def test_split_sentences_basic():
    text = "Revenue rose 5%. Margins fell! Will guidance change? Yes."
    assert split_sentences(text) == [
        "Revenue rose 5%.",
        "Margins fell!",
        "Will guidance change?",
        "Yes.",
    ]


def test_split_sentences_abbreviation_guard():
    assert split_sentences("Acme Inc. reported results. Shares rose.") == [
        "Acme Inc. reported results.",
        "Shares rose.",
    ]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_chunk_text_respects_size_and_overlaps():
    sentence = "The company reported quarterly results today. "
    text = sentence * 20
    chunks = chunk_text(text, chunk_size=120, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + len(sentence) for c in chunks)
    # Overlap: the first sentence of chunk 2 appears at the end of chunk 1.
    assert chunks[1].split(". ")[0] in chunks[0]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a. b.", chunk_size=50, overlap=50)
