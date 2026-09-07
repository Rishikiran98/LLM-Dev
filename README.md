# Fin-LLM

Financial-language sentiment, risk assessment, and LLM-assisted document analysis, packaged as an installable Python project with a CLI and an HTTP API.

Fin-LLM reads earnings commentary, financial news, analyst notes, and filings and produces three things:

1. **Sentence-level sentiment** (negative / neutral / positive) from a local model: a fast scikit-learn baseline or a fine-tuned transformer such as FinBERT.
2. **A document risk score** (0–100) that blends model sentiment with a categorised risk lexicon (credit, liquidity, legal/regulatory, operational, market, guidance) and shows the sentences that drove it.
3. **Structured LLM analysis and question answering** with Claude, grounded in the document, the local model's outputs, and retrieved passages, with citations.

The LLM never sees the document alone: it is handed the local model's sentiment reading and the risk scorer's flagged sentences, and is asked to agree or disagree with evidence. That keeps the expensive model honest and the cheap models useful.

## Quick start

```bash
git clone https://github.com/Rishikiran98/LLM-Dev.git
cd LLM-Dev
pip install -e ".[dev,api]"        # add ,llm for Claude, ,transformer for FinBERT fine-tuning

finllm data show sample            # 120 bundled, synthetic labelled sentences
finllm train sample                # trains the baseline into artifacts/baseline
finllm predict --text "Shares plunged 22% after the company cut its full-year outlook."
finllm risk --file examples/acme_q2_commentary.txt
```

The bundled sample exists so the pipeline runs anywhere with no downloads. It is far too small to measure model quality. For real numbers, train on a public dataset:

```bash
pip install datasets
finllm data list                                   # registered Hugging Face datasets
finllm train financial_phrasebank --model baseline
finllm train financial_phrasebank --model transformer   # needs the [transformer] extra
finllm evaluate artifacts/transformer --source financial_phrasebank
```

Financial PhraseBank is licensed CC BY-NC-SA 4.0; check the terms of any dataset before commercial use.

## Using Claude

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...       # or `ant auth login`

# Full analysis: local sentiment + risk score + Claude's structured read, as JSON
finllm analyze --file examples/acme_q2_commentary.txt --title "Acme Q2"

# Retrieval-augmented Q&A over a folder of documents
finllm index examples/ --output artifacts/index
finllm ask "Did Acme change its guidance, and what risks did it flag?"
```

`analyze` returns a `DocumentAnalysis` object (summary, overall sentiment with rationale, whether the LLM agrees with the local model, key positives and negatives, a list of risks each with quoted evidence and severity, and a confidence score) alongside the local model's reading and the full risk report. `ask` answers only from retrieved passages and cites them by id.

The default model is `claude-opus-5` with adaptive thinking; change it in `configs/default.yaml` or with `FINLLM_LLM__MODEL=claude-sonnet-5`.

## HTTP API

```bash
pip install -e ".[api]"
finllm serve --model-dir artifacts/baseline          # http://127.0.0.1:8000/docs
```

| Method | Path                  | Body                          | Returns                          |
|--------|-----------------------|-------------------------------|----------------------------------|
| GET    | `/health`             |                               | model name, index status         |
| POST   | `/sentiment`          | `{"texts": [...]}`            | one prediction per text          |
| POST   | `/sentiment/document` | `{"text": "..."}`             | one prediction per sentence      |
| POST   | `/risk`               | `{"text": "..."}`             | `RiskReport`                     |
| POST   | `/analyze`            | `{"text": "...", "title": ""}`| local reading + Claude analysis  |
| POST   | `/ask`                | `{"question": "..."}`         | answer + cited passages          |

Set `FINLLM_INDEX_DIR` to a saved index to enable `/ask`.

## Project layout

```
finllm/
  config.py            typed settings: YAML + FINLLM_* environment overrides
  data/                loaders (CSV, Hugging Face), cleaning, stratified split, label schema
  models/
    base.py            Prediction + SentimentModel interface
    baseline.py        TF-IDF (word + char) + logistic regression in one Pipeline
    transformer.py     FinBERT fine-tuning and inference (optional extra)
    registry.py        save/load any model by its model.json
  risk.py              risk lexicon + RiskScorer -> RiskReport
  retrieval/index.py   chunked document index (TF-IDF or sentence-transformers)
  llm/
    client.py          ClaudeClient (anthropic SDK) and FakeLLMClient for tests
    analyst.py         FinancialAnalyst.analyze() and .ask()
  evaluation/          accuracy, macro-F1, confusion matrix, majority baseline
  cli.py               `finllm` command
  api/app.py           FastAPI service
  resources/           bundled sample data and risk lexicon
configs/default.yaml   all tunable settings
tests/                 68 tests; the transformer test builds a tiny BERT offline
notebooks/archive/     the original book-recommender notebook this project grew out of
```

## Design notes

- **No leakage.** Every feature transformer lives inside the model pipeline and is fit only on the training split.
- **One label schema.** Datasets with `bearish/bullish` or integer labels are normalised to `negative/neutral/positive` on load, and transformer checkpoints are re-mapped to that order at inference.
- **Testable without a network.** `FakeLLMClient` stands in for Claude, and the transformer test trains a randomly initialised tiny BERT from a local vocabulary.
- **Risk is explainable.** `RiskReport` carries the sentiment and lexicon components, per-category scores, and the flagged sentences with their matched terms.
- **The lexicon is original.** It is a project-maintained list, not the Loughran-McDonald dictionary, so it can be redistributed with the code. Extend it in `finllm/resources/risk_lexicon.yaml`.

## Development

```bash
make dev        # install with dev, api, and llm extras
make lint       # ruff check + format check
make test       # offline tests (excludes transformer and network markers)
make test-all   # everything, needs torch/transformers
```

CI runs lint and the offline suite on Python 3.10, 3.11, and 3.12, plus the transformer suite on CPU torch.

## Roadmap

- Benchmark table (baseline vs FinBERT vs Claude zero-shot) on Financial PhraseBank and the Twitter financial news set.
- Aspect-level sentiment (revenue, margins, guidance, liquidity) instead of one label per sentence.
- Batch document ingestion (PDF and HTML filings) feeding the index.
- Time-series aggregation of sentiment and risk per company for trend views.
- Evaluation harness for the LLM layer using held-out analyst-written summaries.

## Contributing

Fork, branch from `main`, run `make lint test`, and open a pull request with a description of the change and any new tests. Model or lexicon changes should include before/after metrics.

## License

MIT (see `pyproject.toml`). Add a `LICENSE` file before publishing a release.
