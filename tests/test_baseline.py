import pytest

from finllm.data.preprocess import split_dataset
from finllm.data.schema import LABELS
from finllm.evaluation.metrics import MajorityClassModel, evaluate_model, evaluate_predictions
from finllm.models.base import Prediction, SentimentModel
from finllm.models.baseline import BaselineSentimentModel
from finllm.models.registry import load_model


def test_prediction_from_probabilities():
    pred = Prediction.from_probabilities("x", [0.1, 0.2, 0.7])
    assert pred.label == "positive"
    assert pred.confidence == pytest.approx(0.7)
    assert pred.polarity == pytest.approx(0.6)


def test_prediction_rejects_wrong_length():
    with pytest.raises(ValueError):
        Prediction.from_probabilities("x", [0.5, 0.5])


def test_baseline_conforms_to_protocol(baseline_model):
    assert isinstance(baseline_model, SentimentModel)


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        BaselineSentimentModel().predict(["anything"])


def test_probabilities_sum_to_one(baseline_model):
    rows = baseline_model.predict_proba(["Profit soared.", "The meeting is on Monday."])
    for row in rows:
        assert len(row) == len(LABELS)
        assert sum(row) == pytest.approx(1.0)


def test_baseline_beats_majority_on_training_data(sample_frame, baseline_model):
    texts, labels = sample_frame["text"].tolist(), sample_frame["label"].tolist()
    fitted = evaluate_model(baseline_model, texts, labels)
    majority = evaluate_model(MajorityClassModel(labels), texts, labels)
    assert fitted.accuracy > majority.accuracy
    assert fitted.macro_f1 > 0.9  # it has seen these rows; this checks the wiring


def test_baseline_learns_generalisable_signal(sample_frame):
    train, test = split_dataset(sample_frame, test_size=0.2, seed=3)
    model = BaselineSentimentModel().fit(train["text"].tolist(), train["label"].tolist())
    result = evaluate_model(model, test["text"].tolist(), test["label"].tolist())
    assert result.accuracy > 1 / 3  # better than chance on unseen synthetic rows


def test_save_and_load_round_trip(saved_model_dir, baseline_model):
    loaded = load_model(saved_model_dir)
    texts = ["Shares plunged after the profit warning.", "The company will report on Friday."]
    assert loaded.predict_proba(texts) == baseline_model.predict_proba(texts)


def test_load_model_without_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_evaluation_summary_and_confusion():
    y_true = ["negative", "neutral", "positive", "positive"]
    y_pred = ["negative", "neutral", "positive", "negative"]
    result = evaluate_predictions(y_true, y_pred, model_name="t")
    assert result.accuracy == pytest.approx(0.75)
    assert result.confusion == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert "macro-F1" in result.summary()
