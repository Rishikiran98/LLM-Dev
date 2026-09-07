from pathlib import Path

import pytest

from finllm.config import Settings, load_settings


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(environ={})
    assert settings == Settings()


def test_yaml_and_env_override(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("llm:\n  model: claude-sonnet-5\nrisk:\n  sentiment_weight: 0.3\n")
    settings = load_settings(cfg, environ={"FINLLM_RISK__LEXICON_WEIGHT": "0.7"})
    assert settings.llm.model == "claude-sonnet-5"
    assert settings.risk.sentiment_weight == 0.3
    assert settings.risk.lexicon_weight == 0.7  # env var parsed as a number


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(Path(tmp_path) / "nope.yaml")


def test_repo_default_config_is_valid():
    repo_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    settings = load_settings(repo_cfg, environ={})
    assert settings.retrieval.chunk_overlap < settings.retrieval.chunk_size
