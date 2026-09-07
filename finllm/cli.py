"""Command-line interface: ``finllm --help``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from finllm.config import Settings, load_settings

app = typer.Typer(help="Fin-LLM: financial sentiment, risk assessment, and LLM-assisted analysis.")
data_app = typer.Typer(help="Inspect and prepare datasets.")
app.add_typer(data_app, name="data")

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to a YAML config file.")
_SOURCE_HELP = "'sample', a registered Hub dataset name, or a CSV path"


def _settings(config: Path | None) -> Settings:
    return load_settings(config)


def _read_text(path: Path | None, text: str | None) -> str:
    if text:
        return text
    if path:
        return Path(path).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("Provide --text, --file, or pipe text on stdin")


# --------------------------------------------------------------------------- data
@data_app.command("show")
def data_show(
    source: str = typer.Argument("sample", help=_SOURCE_HELP),
    n: int = typer.Option(5, help="Rows to print"),
) -> None:
    """Print dataset size, label balance, and a few rows."""
    from finllm.data.loaders import load_any
    from finllm.data.preprocess import prepare_dataset

    frame = prepare_dataset(load_any(source))
    typer.echo(f"rows: {len(frame)}")
    typer.echo(f"labels: {frame['label'].value_counts().to_dict()}")
    typer.echo(frame.head(n).to_string(index=False))


@data_app.command("list")
def data_list() -> None:
    """List registered Hugging Face datasets."""
    from finllm.data.loaders import HF_DATASETS

    for name, spec in HF_DATASETS.items():
        typer.echo(f"{name:<24} {spec.path} ({spec.config or 'default'})")


# --------------------------------------------------------------------------- train
@app.command()
def train(
    source: str = typer.Argument("sample", help=_SOURCE_HELP),
    model: str = typer.Option("baseline", help="baseline or transformer"),
    output: Path = typer.Option(None, help="Save directory (default artifacts/<model>)"),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Train a sentiment model and report held-out metrics."""
    from finllm.data.loaders import load_any
    from finllm.data.preprocess import prepare_dataset, split_dataset
    from finllm.evaluation.metrics import evaluate_model

    settings = _settings(config)
    frame = prepare_dataset(load_any(source))
    train_df, test_df = split_dataset(
        frame, test_size=settings.data.test_size, seed=settings.data.random_seed
    )
    output = output or settings.paths.artifacts_dir / model
    typer.echo(f"training {model} on {len(train_df)} rows, evaluating on {len(test_df)}")

    if model == "baseline":
        from finllm.models.baseline import BaselineSentimentModel

        fitted = BaselineSentimentModel(settings.baseline).fit(
            train_df["text"].tolist(), train_df["label"].tolist()
        )
    elif model == "transformer":
        from finllm.models.transformer import fine_tune

        fitted = fine_tune(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            settings=settings.transformer,
            eval_texts=test_df["text"].tolist(),
            eval_labels=test_df["label"].tolist(),
            output_dir=output,
            seed=settings.data.random_seed,
        )
    else:
        raise typer.BadParameter("model must be 'baseline' or 'transformer'")

    fitted.save(str(output))
    result = evaluate_model(fitted, test_df["text"].tolist(), test_df["label"].tolist())
    typer.echo(result.summary())
    (Path(output) / "metrics.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"saved model and metrics to {output}")


# --------------------------------------------------------------------------- evaluate
@app.command()
def evaluate(
    model_dir: Path = typer.Argument(..., help="Directory produced by `finllm train`"),
    source: str = typer.Option("sample", help="Dataset to evaluate on"),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Evaluate a saved model against a dataset and the majority-class baseline."""
    from finllm.data.loaders import load_any
    from finllm.data.preprocess import prepare_dataset, split_dataset
    from finllm.evaluation.metrics import MajorityClassModel, evaluate_model
    from finllm.models.registry import load_model

    settings = _settings(config)
    frame = prepare_dataset(load_any(source))
    train_df, test_df = split_dataset(
        frame, test_size=settings.data.test_size, seed=settings.data.random_seed
    )
    texts, labels = test_df["text"].tolist(), test_df["label"].tolist()
    majority = evaluate_model(MajorityClassModel(train_df["label"].tolist()), texts, labels)
    typer.echo(majority.summary())
    typer.echo("")
    typer.echo(evaluate_model(load_model(model_dir), texts, labels).summary())


# --------------------------------------------------------------------------- predict
@app.command()
def predict(
    model_dir: Path = typer.Option(Path("artifacts/baseline"), help="Saved model directory"),
    text: str | None = typer.Option(None, help="Text to classify"),
    file: Path | None = typer.Option(None, help="File whose sentences are classified"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON lines"),
) -> None:
    """Classify sentiment of a sentence, or every sentence in a file."""
    from finllm.models.registry import load_model
    from finllm.text import split_sentences

    model = load_model(model_dir)
    sentences = split_sentences(_read_text(file, text))
    for pred in model.predict(sentences):
        if as_json:
            typer.echo(pred.model_dump_json())
        else:
            typer.echo(f"{pred.label:<9} {pred.confidence:.2f}  {pred.text}")


# --------------------------------------------------------------------------- risk
@app.command()
def risk(
    model_dir: Path = typer.Option(Path("artifacts/baseline"), help="Saved model directory"),
    text: str | None = typer.Option(None),
    file: Path | None = typer.Option(None),
    as_json: bool = typer.Option(False, "--json"),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Score a document for financial risk signals."""
    from finllm.models.registry import load_model
    from finllm.risk import RiskScorer

    settings = _settings(config)
    scorer = RiskScorer(load_model(model_dir), settings=settings.risk)
    report = scorer.assess(_read_text(file, text))
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"risk score: {report.score}/100 ({report.level})")
    typer.echo(
        f"sentences: {report.sentence_count}  negative share: {report.negative_share:.2f}  "
        f"sentiment component: {report.sentiment_component:.2f}  "
        f"lexicon component: {report.lexicon_component:.2f}"
    )
    if report.category_scores:
        typer.echo("categories: " + json.dumps(report.category_scores))
    for row in report.flagged:
        terms = ", ".join(t for ts in row.risk_terms.values() for t in ts) or "-"
        typer.echo(f"  [{row.score:.2f}] ({row.sentiment}; {terms}) {row.text}")


# --------------------------------------------------------------------------- index
@app.command()
def index(
    paths: list[Path] = typer.Argument(..., help="Text files or directories of .txt/.md to index"),
    output: Path = typer.Option(Path("artifacts/index"), help="Where to save the index"),
    backend: str = typer.Option("tfidf", help="tfidf or sentence-transformer"),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Build a retrieval index over documents for `finllm ask`."""
    from finllm.retrieval.index import DocumentIndex

    settings = _settings(config)
    docs: list[tuple[str, str]] = []
    for path in paths:
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ""}:
                docs.append((str(f), f.read_text(encoding="utf-8", errors="ignore")))
    if not docs:
        raise typer.BadParameter("No text files found")
    idx = DocumentIndex(settings.retrieval, backend=backend).add_documents(docs).build()
    idx.save(output)
    typer.echo(f"indexed {len(docs)} documents into {len(idx.chunks)} chunks at {output}")


# --------------------------------------------------------------------------- ask / analyze
def _analyst(settings: Settings, model_dir: Path | None):
    from finllm.llm.analyst import FinancialAnalyst
    from finllm.llm.client import ClaudeClient

    scorer = None
    if model_dir is not None:
        from finllm.models.registry import load_model
        from finllm.risk import RiskScorer

        scorer = RiskScorer(load_model(model_dir), settings=settings.risk)
    return FinancialAnalyst(ClaudeClient(settings.llm), scorer=scorer)


@app.command()
def ask(
    question: str = typer.Argument(...),
    index_dir: Path = typer.Option(Path("artifacts/index")),
    top_k: int | None = typer.Option(None),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Answer a question over an index using Claude, with passage citations."""
    from finllm.retrieval.index import DocumentIndex

    settings = _settings(config)
    answer = _analyst(settings, None).ask(question, DocumentIndex.load(index_dir), top_k=top_k)
    typer.echo(answer.answer)
    typer.echo("\nsources:")
    for i, hit in enumerate(answer.sources, start=1):
        typer.echo(f"  [{i}] {hit.chunk.doc_id} #{hit.chunk.position} (score {hit.score:.3f})")


@app.command()
def analyze(
    model_dir: Path = typer.Option(Path("artifacts/baseline")),
    text: str | None = typer.Option(None),
    file: Path | None = typer.Option(None),
    title: str | None = typer.Option(None),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Full analysis: local sentiment + risk scoring + Claude's structured read."""
    settings = _settings(config)
    result = _analyst(settings, model_dir).analyze(_read_text(file, text), title=title)
    typer.echo(result.model_dump_json(indent=2))


# --------------------------------------------------------------------------- serve
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    model_dir: Path = typer.Option(Path("artifacts/baseline")),
) -> None:
    """Run the HTTP API (needs `pip install "finllm[api]"`)."""
    import os

    import uvicorn

    os.environ["FINLLM_MODEL_DIR"] = str(model_dir)
    uvicorn.run("finllm.api.app:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":  # pragma: no cover
    app()
