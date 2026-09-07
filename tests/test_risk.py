import pytest

from finllm.risk import LexiconCategory, RiskLexicon, RiskReport, RiskScorer


def test_default_lexicon_loads_categories():
    lexicon = RiskLexicon.default()
    names = {c.name for c in lexicon.categories}
    assert {"credit", "liquidity", "legal_regulatory", "operational", "market", "guidance"} <= names


def test_lexicon_matches_whole_words_only():
    lexicon = RiskLexicon([LexiconCategory("credit", 1.0, ("default", "loan losses"))])
    assert lexicon.find("The company may DEFAULT on its notes.") == {"credit": ["default"]}
    assert lexicon.find("Defaults rose") == {}  # 'defaults' is not 'default'
    assert lexicon.find("higher loan losses") == {"credit": ["loan losses"]}


def test_level_thresholds():
    assert RiskReport.level_for(5) == "low"
    assert RiskReport.level_for(20) == "moderate"
    assert RiskReport.level_for(50) == "elevated"
    assert RiskReport.level_for(90) == "high"


def test_empty_text_scores_zero(baseline_model):
    report = RiskScorer(baseline_model).assess("")
    assert report.score == 0 and report.level == "low" and report.sentence_count == 0


def test_risky_text_scores_higher_than_benign(baseline_model):
    scorer = RiskScorer(baseline_model)
    risky = (
        "The auditor raised substantial doubt about the company's ability to continue as a "
        "going concern. Regulators fined the lender $800 million for compliance failures. "
        "The company breached its debt covenants."
    )
    benign = (
        "The company will report second-quarter results on August 4. "
        "The board will hold its annual meeting in Chicago. "
        "Shares trade on the Nasdaq under the ticker FNLM."
    )
    risky_report, benign_report = scorer.assess(risky), scorer.assess(benign)
    assert risky_report.score > benign_report.score
    assert risky_report.level in {"elevated", "high"}
    assert {"liquidity", "legal_regulatory", "credit"} <= set(risky_report.category_scores)
    assert risky_report.flagged and risky_report.flagged[0].score >= risky_report.flagged[-1].score
    assert benign_report.category_scores == {}


def test_report_is_json_serialisable(baseline_model):
    report = RiskScorer(baseline_model).assess("Shares plunged after a profit warning.")
    payload = report.model_dump_json()
    assert RiskReport.model_validate_json(payload) == report


def test_weights_change_score(baseline_model):
    from finllm.config import RiskSettings

    text = "The company disclosed a data breach affecting 30 million accounts."
    lexicon_heavy = RiskScorer(
        baseline_model, settings=RiskSettings(sentiment_weight=0.0, lexicon_weight=1.0)
    )
    sentiment_only = RiskScorer(
        baseline_model, settings=RiskSettings(sentiment_weight=1.0, lexicon_weight=0.0)
    )
    assert lexicon_heavy.assess(text).lexicon_component == pytest.approx(
        sentiment_only.assess(text).lexicon_component
    )
    assert lexicon_heavy.assess(text).score != sentiment_only.assess(text).score
