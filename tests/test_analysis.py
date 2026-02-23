"""Tests for analysis functions."""

import pytest
import pandas as pd
import numpy as np
from report_lib.analysis import (
    analyze_dataset_characteristics,
    analyze_training_dynamics,
)


@pytest.fixture
def sample_metadata():
    return pd.DataFrame(
        {
            "speaker": ["spk1", "spk1", "spk2", "spk2", "spk3"],
            "text": ["hello", "world", "foo bar", "baz qux", "test"],
            "duration": [1.5, 2.0, 3.1, 2.5, 1.8],
        }
    )


@pytest.fixture
def sample_splits():
    return {
        "train": pd.DataFrame({"speaker": ["spk1", "spk1", "spk2"]}),
        "val": pd.DataFrame({"speaker": ["spk2", "spk3"]}),
        "test_holdout": pd.DataFrame({"speaker": ["spk3", "spk3"]}),
        "test_holdout_speakers": {"spk3"},
    }


def test_analyze_dataset_characteristics_returns_dict(sample_metadata, sample_splits):
    """Test dataset analysis returns structured results."""
    result = analyze_dataset_characteristics(sample_metadata, sample_splits)

    assert isinstance(result, dict)
    assert "speaker_distribution" in result
    assert "text_statistics" in result
    assert "audio_properties" in result


def test_dataset_analysis_identifies_holdout_speakers(sample_metadata, sample_splits):
    """Test holdout speaker identification in analysis."""
    result = analyze_dataset_characteristics(sample_metadata, sample_splits)

    assert result["holdout_speakers"] == {"spk3"}


def test_analyze_training_dynamics_returns_dict():
    """Test training dynamics analysis."""
    trainer_state = {
        "epoch": 100,
        "global_step": 5000,
        "training_loss": [1.5, 1.2, 1.0, 0.8, 0.7],
    }

    result = analyze_training_dynamics(trainer_state)

    assert isinstance(result, dict)
    assert "metadata" in result
    assert "convergence" in result
