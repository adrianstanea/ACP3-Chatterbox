"""Tests for visualization functions."""

import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from report_lib.visualizations import (
    plot_speaker_distribution,
    plot_text_statistics,
    plot_loss_curves,
    plot_performance_comparison,
)


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "figures"


def test_plot_speaker_distribution_creates_file(tmp_output_dir):
    """Test speaker distribution plot generation."""
    tmp_output_dir.mkdir()

    speaker_dist = {"spk1": 100, "spk2": 95, "spk3": 110}

    output_path = plot_speaker_distribution(speaker_dist, tmp_output_dir, dpi=100)

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_plot_text_statistics_creates_file(tmp_output_dir):
    """Test text statistics plot generation."""
    tmp_output_dir.mkdir()

    text_stats = {
        "char_count_mean": 45.2,
        "char_count_std": 12.3,
        "char_count_min": 10,
        "char_count_max": 150,
    }

    output_path = plot_text_statistics(text_stats, tmp_output_dir, dpi=100)

    assert output_path.exists()


def test_plot_loss_curves_creates_file(tmp_output_dir):
    """Test loss curve plot generation."""
    tmp_output_dir.mkdir()

    trainer_state = {
        "training_loss": [1.5, 1.2, 1.0, 0.8, 0.7],
        "epoch": 5,
    }

    output_path = plot_loss_curves(trainer_state, tmp_output_dir, dpi=100)

    assert output_path.exists()


def test_plot_performance_comparison_creates_file(tmp_output_dir):
    """Test performance comparison plot generation."""
    tmp_output_dir.mkdir()

    performance = {
        "bas_exag0.3_cfg0.3": {
            "CER": {"mean": 0.10, "std": 0.14},
            "WER": {"mean": 0.28, "std": 0.23},
            "UTMOS_Synthesized": {"mean": 3.39, "std": 0.57},
            "Speaker_Cosine_Similarity": {"mean": 0.81, "std": 0.07},
        },
    }

    output_path = plot_performance_comparison(performance, tmp_output_dir, dpi=100)

    assert output_path.exists()
