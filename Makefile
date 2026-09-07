.PHONY: install dev lint format test test-all train serve clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,api,llm]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -m "not transformer and not network"

test-all:
	pytest

train:
	finllm train sample --model baseline --output artifacts/baseline

serve:
	finllm serve --model-dir artifacts/baseline

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info
