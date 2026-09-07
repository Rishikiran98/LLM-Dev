"""LLM-assisted analysis that grounds Claude in retrieved text and model outputs.

Two entry points:

* :meth:`FinancialAnalyst.analyze` reads a whole document, runs the local
  sentiment model and risk scorer, and asks Claude for a structured analysis
  that must cite the evidence it used.
* :meth:`FinancialAnalyst.ask` answers a question over a :class:`DocumentIndex`
  with retrieved chunks as the only allowed source.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from finllm.llm.client import LLMClient
from finllm.retrieval.index import DocumentIndex, Hit
from finllm.risk import RiskReport, RiskScorer

SYSTEM_ANALYST = """You are a financial analyst assistant. You read earnings reports, \
financial news, analyst notes, and market commentary and produce careful, sceptical analysis.

Rules:
- Use only the material provided in the user message. Do not bring in outside facts \
about the company or the market.
- Quote or closely paraphrase specific sentences as evidence for every claim.
- Distinguish facts stated in the text from your interpretation.
- Where the local sentiment model and your own reading disagree, say so and explain why.
- Never give investment advice; describe risks and signals, not recommendations to trade."""

SYSTEM_QA = """You answer questions about financial documents using only the retrieved \
passages supplied in the user message. Each passage has an id like [3]. Cite passage ids \
inline after the sentences they support. If the passages do not contain the answer, say \
that plainly instead of guessing. Never give investment advice."""


class RiskItem(BaseModel):
    category: str = Field(
        description="credit, liquidity, legal_regulatory, operational, market, guidance, or other"
    )
    description: str
    evidence: str = Field(description="A quoted or closely paraphrased sentence from the text")
    severity: str = Field(description="low, moderate, elevated, or high")


class DocumentAnalysis(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": "The company beat estimates but flagged supply chain risk.",
                "overall_sentiment": "positive",
                "sentiment_rationale": "Revenue and margin beats outweigh the cautionary note.",
                "agrees_with_model": True,
                "key_positives": ["Revenue rose 18%"],
                "key_negatives": ["Supply chain disruptions"],
                "risks": [
                    {
                        "category": "operational",
                        "description": "Production dependent on a constrained supplier",
                        "evidence": "Supply chain disruptions forced the automaker to halt "
                        "production",
                        "severity": "moderate",
                    }
                ],
                "confidence": 0.8,
            }
        }
    )

    summary: str = Field(description="Two to four sentences summarising the document")
    overall_sentiment: str = Field(description="negative, neutral, or positive")
    sentiment_rationale: str
    agrees_with_model: bool = Field(
        description="Whether the analyst agrees with the local model's sentiment reading"
    )
    key_positives: list[str]
    key_negatives: list[str]
    risks: list[RiskItem]
    confidence: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    llm: DocumentAnalysis
    model_sentiment: str
    model_polarity: float
    risk: RiskReport


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Hit]


def _format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (doc={h.chunk.doc_id}, chunk={h.chunk.position}, score={h.score:.3f})\n"
        f"{h.chunk.text}"
        for i, h in enumerate(hits)
    )


class FinancialAnalyst:
    def __init__(self, client: LLMClient, scorer: RiskScorer | None = None) -> None:
        self.client = client
        self.scorer = scorer

    def analyze(self, text: str, *, title: str | None = None) -> AnalysisResult:
        if self.scorer is None:
            raise ValueError("analyze() needs a RiskScorer so the LLM can see model outputs")
        risk = self.scorer.assess(text)
        preds = self.scorer.model.predict([text])
        model_pred = preds[0]

        flagged = (
            "\n".join(
                f"- ({r.sentiment}, risk={r.score:.2f}, terms={r.risk_terms}) {r.text}"
                for r in risk.flagged
            )
            or "- none"
        )
        user = (
            f"Document title: {title or 'untitled'}\n\n"
            f"=== Document ===\n{text}\n\n"
            f"=== Local sentiment model ===\n"
            f"label={model_pred.label} polarity={model_pred.polarity:+.2f} "
            f"probabilities={model_pred.probabilities}\n\n"
            f"=== Local risk scorer ===\n"
            f"score={risk.score}/100 level={risk.level} negative_share={risk.negative_share}\n"
            f"category_scores={risk.category_scores}\n"
            f"flagged sentences:\n{flagged}\n\n"
            "Produce the structured analysis."
        )
        llm = self.client.complete_structured(
            system=SYSTEM_ANALYST, user=user, schema=DocumentAnalysis
        )
        return AnalysisResult(
            llm=llm,
            model_sentiment=model_pred.label,
            model_polarity=round(model_pred.polarity, 4),
            risk=risk,
        )

    def ask(self, question: str, index: DocumentIndex, *, top_k: int | None = None) -> Answer:
        hits = index.search(question, top_k=top_k)
        if not hits:
            return Answer(
                question=question,
                answer="No relevant passages were found in the index for this question.",
                sources=[],
            )
        user = (
            f"Question: {question}\n\n=== Retrieved passages ===\n{_format_hits(hits)}\n\n"
            "Answer the question using only these passages, citing passage ids."
        )
        answer = self.client.complete(system=SYSTEM_QA, user=user)
        return Answer(question=question, answer=answer, sources=hits)
