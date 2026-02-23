"""Integration tests for the full academic report pipeline."""

import pytest
from pathlib import Path
import json
import pandas as pd
import tempfile
import sys


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a minimal test data directory."""
    # Create metrics directory with sample CSV
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    test_metrics = pd.DataFrame(
        {
            "CER": [0.10, 0.15, 0.12, 0.11],
            "WER": [0.25, 0.30, 0.28, 0.27],
            "UTMOS_Synthesized": [3.2, 3.1, 3.3, 3.2],
            "Speaker_Cosine_Similarity": [0.80, 0.82, 0.81, 0.79],
        }
    )
    test_metrics.to_csv(metrics_dir / "bas_exag0.3_cfg0.3.csv", index=False)

    # Create trainer state
    trainer_state = {
        "epoch": 100,
        "global_step": 5000,
        "training_loss": [1.5, 1.2, 1.0, 0.8, 0.7],
        "validation_loss": [1.4, 1.1, 0.9, 0.8, 0.75],
    }
    trainer_file = tmp_path / "trainer_state.json"
    with open(trainer_file, "w") as f:
        json.dump(trainer_state, f)

    # Create metadata
    metadata = pd.DataFrame(
        {
            "speaker": ["spk1", "spk2", "spk3", "spk1"],
            "text": ["hello world", "foo bar baz", "test text", "another sample"],
            "duration": [2.5, 3.1, 2.8, 2.9],
        }
    )
    metadata_file = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_file, index=False)

    # Create splits
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train = pd.DataFrame(
        {"speaker": ["spk1", "spk1", "spk2"], "utterance": ["u1", "u2", "u3"]}
    )
    val = pd.DataFrame({"speaker": ["spk2", "spk3"], "utterance": ["u4", "u5"]})
    test = pd.DataFrame({"speaker": ["spk3"], "utterance": ["u6"]})

    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test_holdout.csv", index=False)

    # Create filelists
    filelists_dir = tmp_path / "filelists"
    filelists_dir.mkdir()

    filelist = pd.DataFrame({"file": ["file1.wav", "file2.wav", "file3.wav", "file4.wav"]})
    filelist.to_csv(filelists_dir / "bas_exag0.3_cfg0.3.csv", index=False)

    # Create reports dir (empty, just needs to exist)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    return {
        "metrics_dir": metrics_dir,
        "trainer_state": trainer_file,
        "metadata": metadata_file,
        "splits_dir": splits_dir,
        "filelists_dir": filelists_dir,
        "reports_dir": reports_dir,
    }


def test_full_pipeline_executes(temp_data_dir, tmp_path):
    """Test that the full pipeline can execute without errors."""
    from report_lib.loaders import (
        load_metrics_csvs,
        load_trainer_state,
        load_dataset_metadata,
        load_dataset_splits,
        load_filelists,
    )
    from report_lib.analysis import (
        analyze_dataset_characteristics,
        analyze_training_dynamics,
        analyze_performance,
    )

    # Load data
    metrics = load_metrics_csvs(temp_data_dir["metrics_dir"])
    trainer_state = load_trainer_state(temp_data_dir["trainer_state"])
    metadata = load_dataset_metadata(temp_data_dir["metadata"])
    splits = load_dataset_splits(temp_data_dir["splits_dir"])
    filelists = load_filelists(temp_data_dir["filelists_dir"])

    # Verify data loaded
    assert len(metrics) > 0
    assert len(trainer_state) > 0
    assert len(metadata) > 0

    # Run analyses
    dataset_analysis = analyze_dataset_characteristics(metadata, splits)
    training_analysis = analyze_training_dynamics(trainer_state)
    performance_analysis = analyze_performance(metrics)

    # Verify analyses ran
    assert "speaker_distribution" in dataset_analysis
    assert "convergence" in training_analysis
    assert "conditions" in performance_analysis


def test_output_directories_created(temp_data_dir, tmp_path):
    """Test that output directories are properly created."""
    from report_lib.utils import ensure_output_dirs

    output_dir = tmp_path / "report_output"
    dirs = ensure_output_dirs(output_dir, ["html", "pdf", "latex", "markdown", "json"])

    # Verify structure
    assert dirs["figures"].exists()
    assert (dirs["figures"] / "dataset_characterization").exists()
    assert (dirs["figures"] / "training_dynamics").exists()
    assert (dirs["figures"] / "evaluation_results").exists()
    assert (dirs["figures"] / "ablation_analysis").exists()
    assert dirs["tables"].exists()
    assert dirs["data"].exists()


def test_json_export_roundtrip(tmp_path):
    """Test JSON save and load roundtrip."""
    from report_lib.utils import save_json, load_json

    test_data = {
        "metrics": {"mean": 0.5, "std": 0.1},
        "conditions": ["cond1", "cond2"],
        "nested": {"key": "value"},
    }

    output_file = tmp_path / "test.json"
    save_json(test_data, output_file)

    loaded_data = load_json(output_file)

    assert loaded_data == test_data
