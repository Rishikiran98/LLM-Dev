import json

from typer.testing import CliRunner

from finllm.cli import app

runner = CliRunner()


def test_data_show_and_list():
    result = runner.invoke(app, ["data", "show", "sample", "--n", "2"])
    assert result.exit_code == 0, result.output
    assert "rows: 120" in result.output
    result = runner.invoke(app, ["data", "list"])
    assert "financial_phrasebank" in result.output


def test_train_predict_risk_round_trip(tmp_path):
    out = tmp_path / "model"
    result = runner.invoke(app, ["train", "sample", "--model", "baseline", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "metrics.json").exists() and (out / "model.json").exists()

    result = runner.invoke(
        app,
        ["predict", "--model-dir", str(out), "--text", "Shares plunged. Board meets.", "--json"],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.output.strip().splitlines()]
    assert len(rows) == 2 and {"label", "confidence"} <= set(rows[0])

    result = runner.invoke(
        app, ["risk", "--model-dir", str(out), "--text", "The lender defaulted.", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["category_scores"].get("credit")

    result = runner.invoke(app, ["evaluate", str(out), "--source", "sample"])
    assert result.exit_code == 0, result.output
    assert "model: majority" in result.output and "model: baseline" in result.output


def test_train_rejects_unknown_model(tmp_path):
    result = runner.invoke(app, ["train", "sample", "--model", "nope", "--output", str(tmp_path)])
    assert result.exit_code != 0


def test_index_command(tmp_path):
    (tmp_path / "a.txt").write_text("Acme raised guidance. Margins expanded.")
    out = tmp_path / "idx"
    result = runner.invoke(app, ["index", str(tmp_path), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "index.json").exists()


def test_predict_without_input_fails():
    result = runner.invoke(app, ["predict", "--model-dir", "/nonexistent"], input="")
    assert result.exit_code != 0
