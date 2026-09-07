import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finllm.api.app import create_app  # noqa: E402
from finllm.config import Settings  # noqa: E402
from finllm.llm.analyst import FinancialAnalyst  # noqa: E402
from finllm.llm.client import FakeLLMClient  # noqa: E402
from finllm.retrieval.index import DocumentIndex  # noqa: E402
from finllm.risk import RiskScorer  # noqa: E402


@pytest.fixture
def client(saved_model_dir, baseline_model, tmp_path):
    index_dir = tmp_path / "index"
    DocumentIndex().add_documents([("d", "Acme raised guidance. Margins grew.")]).build().save(
        index_dir
    )
    analyst = FinancialAnalyst(
        FakeLLMClient(text="Guidance was raised [1]."), scorer=RiskScorer(baseline_model)
    )
    app = create_app(saved_model_dir, index_dir, settings=Settings(), analyst=analyst)
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "baseline" and body["index_loaded"]


def test_sentiment_endpoint(client):
    resp = client.post("/sentiment", json={"texts": ["Profit soared 40%."]})
    assert resp.status_code == 200
    pred = resp.json()["predictions"][0]
    assert set(pred["probabilities"]) == {"negative", "neutral", "positive"}


def test_sentiment_document_splits_sentences(client):
    resp = client.post("/sentiment/document", json={"text": "Profit rose. Costs fell."})
    assert len(resp.json()["predictions"]) == 2


def test_sentiment_validation(client):
    assert client.post("/sentiment", json={"texts": []}).status_code == 422


def test_risk_endpoint(client):
    resp = client.post("/risk", json={"text": "The company breached its debt covenants."})
    body = resp.json()
    assert resp.status_code == 200 and "credit" in body["category_scores"]


def test_analyze_and_ask_use_injected_analyst(client):
    analyze = client.post("/analyze", json={"text": "Revenue rose 18%.", "title": "t"})
    assert analyze.status_code == 200 and analyze.json()["llm"]["overall_sentiment"] == "positive"
    ask = client.post("/ask", json={"question": "Was guidance raised?"})
    assert ask.status_code == 200 and ask.json()["answer"].startswith("Guidance")


def test_ask_without_index(saved_model_dir):
    app = create_app(saved_model_dir, None, settings=Settings())
    resp = TestClient(app).post("/ask", json={"question": "x"})
    assert resp.status_code == 404
