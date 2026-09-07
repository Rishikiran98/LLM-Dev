"""FastAPI service exposing sentiment, risk, and (optionally) LLM analysis.

Run with ``finllm serve`` or ``uvicorn finllm.api.app:create_app --factory``.
Set ``FINLLM_MODEL_DIR`` to the saved model directory (default artifacts/baseline)
and ``FINLLM_INDEX_DIR`` to enable ``/ask``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finllm import __version__
from finllm.config import Settings, load_settings
from finllm.models.base import Prediction
from finllm.models.registry import load_model
from finllm.risk import RiskReport, RiskScorer
from finllm.text import split_sentences


class SentimentRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class SentimentResponse(BaseModel):
    predictions: list[Prediction]


class DocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class _Resources:
    def __init__(self, settings: Settings, model_dir: Path, index_dir: Path | None) -> None:
        self.settings = settings
        self.model = load_model(model_dir)
        self.scorer = RiskScorer(self.model, settings=settings.risk)
        self.index = None
        if index_dir is not None and index_dir.exists():
            from finllm.retrieval.index import DocumentIndex

            self.index = DocumentIndex.load(index_dir)
        self._analyst = None

    @property
    def analyst(self):
        if self._analyst is None:
            from finllm.llm.analyst import FinancialAnalyst
            from finllm.llm.client import ClaudeClient

            self._analyst = FinancialAnalyst(ClaudeClient(self.settings.llm), scorer=self.scorer)
        return self._analyst


def create_app(
    model_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    settings: Settings | None = None,
    analyst=None,
) -> FastAPI:
    model_dir = Path(model_dir or os.environ.get("FINLLM_MODEL_DIR", "artifacts/baseline"))
    index_env = os.environ.get("FINLLM_INDEX_DIR")
    index_dir = Path(index_dir) if index_dir else (Path(index_env) if index_env else None)
    settings = settings or load_settings()

    app = FastAPI(title="Fin-LLM", version=__version__)

    @lru_cache(maxsize=1)
    def resources() -> _Resources:
        res = _Resources(settings, model_dir, index_dir)
        if analyst is not None:
            res._analyst = analyst
        return res

    @app.get("/health")
    def health() -> dict:
        res = resources()
        return {
            "status": "ok",
            "version": __version__,
            "model": res.model.name,
            "index_loaded": res.index is not None,
        }

    @app.post("/sentiment", response_model=SentimentResponse)
    def sentiment(req: SentimentRequest) -> SentimentResponse:
        return SentimentResponse(predictions=resources().model.predict(req.texts))

    @app.post("/sentiment/document", response_model=SentimentResponse)
    def sentiment_document(req: DocumentRequest) -> SentimentResponse:
        sentences = split_sentences(req.text)
        return SentimentResponse(predictions=resources().model.predict(sentences))

    @app.post("/risk", response_model=RiskReport)
    def risk(req: DocumentRequest) -> RiskReport:
        return resources().scorer.assess(req.text)

    @app.post("/analyze")
    def analyze(req: DocumentRequest) -> dict:
        try:
            result = resources().analyst.analyze(req.text, title=req.title)
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return result.model_dump()

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        res = resources()
        if res.index is None:
            raise HTTPException(status_code=404, detail="No index loaded; set FINLLM_INDEX_DIR")
        try:
            answer = res.analyst.ask(req.question, res.index, top_k=req.top_k)
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return answer.model_dump()

    return app
