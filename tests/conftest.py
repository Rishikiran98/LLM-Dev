from __future__ import annotations

import pytest

from finllm.data.loaders import load_sample
from finllm.data.preprocess import prepare_dataset
from finllm.models.baseline import BaselineSentimentModel


@pytest.fixture(scope="session")
def sample_frame():
    return prepare_dataset(load_sample())


@pytest.fixture(scope="session")
def baseline_model(sample_frame):
    """A baseline fitted on the whole sample; used where accuracy is irrelevant."""
    return BaselineSentimentModel().fit(
        sample_frame["text"].tolist(), sample_frame["label"].tolist()
    )


@pytest.fixture(scope="session")
def saved_model_dir(tmp_path_factory, baseline_model):
    directory = tmp_path_factory.mktemp("model")
    baseline_model.save(directory)
    return directory
