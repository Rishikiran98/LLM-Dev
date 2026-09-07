from types import SimpleNamespace

import pytest

from finllm.config import LLMSettings
from finllm.llm.analyst import DocumentAnalysis, FinancialAnalyst
from finllm.llm.client import ClaudeClient, FakeLLMClient
from finllm.retrieval.index import DocumentIndex
from finllm.risk import RiskScorer


def test_fake_client_builds_from_schema_example():
    client = FakeLLMClient()
    result = client.complete_structured(system="s", user="u", schema=DocumentAnalysis)
    assert isinstance(result, DocumentAnalysis)
    assert client.calls[0]["user"] == "u"


def test_analyze_grounds_prompt_in_model_outputs(baseline_model):
    client = FakeLLMClient()
    analyst = FinancialAnalyst(client, scorer=RiskScorer(baseline_model))
    text = "Regulators fined the lender $800 million. The company will report on Friday."
    result = analyst.analyze(text, title="Lender note")
    prompt = client.calls[-1]["user"]
    assert "Lender note" in prompt and text in prompt
    assert "Local sentiment model" in prompt and "Local risk scorer" in prompt
    assert result.model_sentiment in {"negative", "neutral", "positive"}
    assert result.risk.sentence_count == 2
    assert result.llm.overall_sentiment == "positive"  # from the schema example


def test_analyze_requires_scorer():
    with pytest.raises(ValueError):
        FinancialAnalyst(FakeLLMClient()).analyze("text")


def test_ask_cites_retrieved_passages():
    index = (
        DocumentIndex()
        .add_documents([("doc1", "Acme raised its guidance. Margins expanded.")])
        .build()
    )
    client = FakeLLMClient(text="Acme raised guidance [1].")
    answer = FinancialAnalyst(client).ask("Did Acme raise guidance?", index)
    assert answer.answer.endswith("[1].")
    assert answer.sources[0].chunk.doc_id == "doc1"
    assert "[1]" in client.calls[-1]["user"]


def test_ask_without_hits_does_not_call_llm():
    index = DocumentIndex().add_documents([("doc1", "Nothing relevant here.")]).build()
    client = FakeLLMClient()
    answer = FinancialAnalyst(client).ask("zebra quantum", index)
    assert answer.sources == [] and client.calls == []


class _StubMessages:
    """Mimics ``anthropic.Anthropic().messages`` closely enough to test the wrapper."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_claude_client_sends_adaptive_thinking_and_effort():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
    )
    stub = SimpleNamespace(messages=_StubMessages(response))
    client = ClaudeClient(LLMSettings(model="claude-opus-5", effort="low"), client=stub)
    assert client.complete(system="s", user="u") == "hello"
    sent = stub.messages.kwargs
    assert sent["model"] == "claude-opus-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["system"] == "s"


def test_claude_client_structured_output():
    example = DocumentAnalysis.model_config["json_schema_extra"]["example"]
    response = SimpleNamespace(
        stop_reason="end_turn", parsed_output=DocumentAnalysis.model_validate(example)
    )
    stub = SimpleNamespace(messages=_StubMessages(response))
    client = ClaudeClient(client=stub)
    out = client.complete_structured(system="s", user="u", schema=DocumentAnalysis)
    assert out.overall_sentiment == "positive"
    assert stub.messages.kwargs["output_format"] is DocumentAnalysis


def test_claude_client_surfaces_refusal():
    response = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="other"),
        content=[],
    )
    stub = SimpleNamespace(messages=_StubMessages(response))
    with pytest.raises(RuntimeError, match="declined"):
        ClaudeClient(client=stub).complete(system="s", user="u")
