"""A small, persistent document index for retrieval-augmented analysis.

Two embedding backends share one interface:

* ``tfidf`` (default): no extra dependencies, good lexical recall for filings.
* ``sentence-transformer``: dense embeddings via ``sentence-transformers``
  (``pip install "finllm[embeddings]"``), better for paraphrased questions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from finllm.config import RetrievalSettings
from finllm.text import chunk_text


class IndexedChunk(BaseModel):
    chunk_id: int
    doc_id: str
    text: str
    position: int


class Hit(BaseModel):
    chunk: IndexedChunk
    score: float


class _TfidfBackend:
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode"
        )
        self.matrix = None

    def fit(self, texts: Sequence[str]) -> None:
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str) -> np.ndarray:
        q = self.vectorizer.transform([text])
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def dump(self) -> dict:
        return {"vectorizer": self.vectorizer, "matrix": self.matrix}

    @classmethod
    def load(cls, state: dict) -> _TfidfBackend:
        backend = cls()
        backend.vectorizer = state["vectorizer"]
        backend.matrix = state["matrix"]
        return backend


class _SentenceTransformerBackend:
    name = "sentence-transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Dense retrieval needs sentence-transformers: pip install "finllm[embeddings]"'
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.matrix: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> None:
        self.matrix = self.model.encode(list(texts), normalize_embeddings=True)

    def query(self, text: str) -> np.ndarray:
        q = self.model.encode([text], normalize_embeddings=True)[0]
        return self.matrix @ q

    def dump(self) -> dict:
        return {"model_name": self.model_name, "matrix": self.matrix}

    @classmethod
    def load(cls, state: dict) -> _SentenceTransformerBackend:
        backend = cls(state["model_name"])
        backend.matrix = state["matrix"]
        return backend


_BACKENDS = {"tfidf": _TfidfBackend, "sentence-transformer": _SentenceTransformerBackend}


class DocumentIndex:
    def __init__(
        self, settings: RetrievalSettings | None = None, *, backend: str = "tfidf", **backend_kwargs
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(_BACKENDS)}")
        self.settings = settings or RetrievalSettings()
        self.backend_name = backend
        self._backend = _BACKENDS[backend](**backend_kwargs)
        self.chunks: list[IndexedChunk] = []

    # ------------------------------------------------------------------ building
    def add_documents(self, documents: Iterable[tuple[str, str]]) -> DocumentIndex:
        """Add ``(doc_id, text)`` pairs. Call :meth:`build` afterwards."""
        for doc_id, text in documents:
            pieces = chunk_text(
                text,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            for position, piece in enumerate(pieces):
                self.chunks.append(
                    IndexedChunk(
                        chunk_id=len(self.chunks), doc_id=doc_id, text=piece, position=position
                    )
                )
        return self

    def build(self) -> DocumentIndex:
        if not self.chunks:
            raise ValueError("No documents added; nothing to index")
        self._backend.fit([c.text for c in self.chunks])
        return self

    # ------------------------------------------------------------------ querying
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.settings.top_k
        scores = self._backend.query(query)
        order = np.argsort(-scores)[:k]
        return [
            Hit(chunk=self.chunks[int(i)], score=float(scores[int(i)]))
            for i in order
            if scores[int(i)] > 0
        ]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._backend.dump(), directory / "backend.joblib")
        payload = {
            "backend": self.backend_name,
            "settings": self.settings.model_dump(),
            "chunks": [c.model_dump() for c in self.chunks],
        }
        (directory / "index.json").write_text(json.dumps(payload), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> DocumentIndex:
        directory = Path(directory)
        payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        index = cls.__new__(cls)
        index.settings = RetrievalSettings.model_validate(payload["settings"])
        index.backend_name = payload["backend"]
        index.chunks = [IndexedChunk.model_validate(c) for c in payload["chunks"]]
        state = joblib.load(directory / "backend.joblib")
        index._backend = _BACKENDS[index.backend_name].load(state)
        return index
