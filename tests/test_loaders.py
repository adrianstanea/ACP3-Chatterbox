"""Tests for data loaders module."""

import pytest
from pathlib import Path
import pandas as pd
import json
from report_lib.loaders import (
    load_metrics_csvs,
    load_trainer_state,
    load_dataset_metadata,
    load_dataset_splits,
    validate_loaded_data,
)


def test_load_metrics_csvs_returns_dict(tmp_path):
    """Test that metrics loading returns dict keyed by condition."""
    # Create mock CSV files
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    test_data = pd.DataFrame(
        {
            "CER": [0.1, 0.15],
            "WER": [0.25, 0.3],
            "UTMOS_Synthesized": [3.2, 3.4],
            "Speaker_Cosine_Similarity": [0.8, 0.82],
        }
    )
    test_data.to_csv(metrics_dir / "bas_exag0.3_cfg0.3.csv", index=False)

    result = load_metrics_csvs(metrics_dir)

    assert isinstance(result, dict)
    assert "bas_exag0.3_cfg0.3" in result
    assert isinstance(result["bas_exag0.3_cfg0.3"], pd.DataFrame)


def test_load_trainer_state_parses_json(tmp_path):
    """Test trainer state JSON loading."""
    state_file = tmp_path / "trainer_state.json"
    state_data = {
        "epoch": 100,
        "global_step": 5000,
        "loss": 0.45,
        "learning_rate": 0.0001,
    }
    state_file.write_text(json.dumps(state_data))

    result = load_trainer_state(state_file)

    assert result["epoch"] == 100
    assert result["global_step"] == 5000


def test_load_dataset_metadata_returns_dataframe(tmp_path):
    """Test metadata loading."""
    metadata_file = tmp_path / "metadata.csv"
    metadata = pd.DataFrame(
        {
            "speaker_id": ["spk1", "spk2"],
            "text": ["text1", "text2"],
            "duration": [2.5, 3.1],
        }
    )
    metadata.to_csv(metadata_file, index=False)

    result = load_dataset_metadata(metadata_file)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_load_dataset_splits_identifies_holdout_speakers(tmp_path):
    """Test split loading with holdout speaker identification."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train = pd.DataFrame(
        {"speaker": ["spk1", "spk1", "spk2"], "utterance": ["u1", "u2", "u3"]}
    )
    val = pd.DataFrame({"speaker": ["spk1", "spk2"], "utterance": ["u4", "u5"]})
    test = pd.DataFrame({"speaker": ["spk3", "spk4"], "utterance": ["u6", "u7"]})

    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test_holdout.csv", index=False)

    result = load_dataset_splits(splits_dir)

    assert "train" in result
    assert "test_holdout_speakers" in result
    assert result["test_holdout_speakers"] == {"spk3", "spk4"}
