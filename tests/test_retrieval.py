import pytest

from finllm.config import RetrievalSettings
from finllm.retrieval.index import DocumentIndex

DOCS = [
    (
        "acme_q2",
        "Acme raised its full-year guidance after strong cloud demand. "
        "Operating margin expanded to 24.5 percent. "
        "The board approved a dividend increase.",
    ),
    (
        "beta_q2",
        "Beta Corp cut its revenue forecast on weak demand for personal computers. "
        "The company announced 4,000 layoffs. "
        "Management suspended the dividend to preserve cash.",
    ),
]


def test_search_returns_relevant_document():
    index = DocumentIndex().add_documents(DOCS).build()
    hits = index.search("Which company cut its forecast?", top_k=1)
    assert hits[0].chunk.doc_id == "beta_q2"
    hits = index.search("guidance raised margin", top_k=1)
    assert hits[0].chunk.doc_id == "acme_q2"


def test_search_drops_zero_scores():
    index = DocumentIndex().add_documents(DOCS).build()
    assert index.search("zebra quantum", top_k=5) == []


def test_chunking_follows_settings():
    settings = RetrievalSettings(chunk_size=80, chunk_overlap=20, top_k=3)
    index = DocumentIndex(settings).add_documents(DOCS).build()
    assert len(index.chunks) > len(DOCS)
    assert len(index.search("dividend")) <= 3


def test_build_without_documents():
    with pytest.raises(ValueError):
        DocumentIndex().build()


def test_unknown_backend():
    with pytest.raises(ValueError):
        DocumentIndex(backend="magic")


def test_save_and_load(tmp_path):
    index = DocumentIndex().add_documents(DOCS).build()
    index.save(tmp_path / "idx")
    loaded = DocumentIndex.load(tmp_path / "idx")
    assert [c.model_dump() for c in loaded.chunks] == [c.model_dump() for c in index.chunks]
    assert loaded.search("layoffs", top_k=1)[0].chunk.doc_id == "beta_q2"
