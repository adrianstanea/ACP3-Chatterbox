"""Tests for table generation functions."""

import pytest
from pathlib import Path
import pandas as pd
from report_lib.tables import (
    generate_metrics_table,
    generate_dataset_summary_table,
    export_to_latex,
    export_to_markdown,
    generate_ablation_table,
)


def test_generate_metrics_table_returns_dataframe():
    """Test metrics table generation."""
    performance = {
        "bas_exag0.3_cfg0.3": {
            "CER": {"mean": 0.10, "std": 0.14},
            "WER": {"mean": 0.28, "std": 0.23},
        },
        "sgs_exag0.3_cfg0.3": {
            "CER": {"mean": 0.16, "std": 0.23},
            "WER": {"mean": 0.37, "std": 0.30},
        },
    }

    table = generate_metrics_table(performance)

    assert isinstance(table, pd.DataFrame)
    assert len(table) == 2


def test_generate_dataset_summary_table_returns_dataframe():
    """Test dataset summary table generation."""
    dataset_chars = {
        "metadata": {"total_samples": 1000, "unique_speakers": 50},
        "speaker_distribution": {"mean_samples_per_speaker": 20.0},
        "text_statistics": {"char_count_mean": 45.2},
        "audio_properties": {"duration_mean": 2.5},
    }

    table = generate_dataset_summary_table(dataset_chars)

    assert isinstance(table, pd.DataFrame)
    assert len(table) > 0


def test_export_to_latex_creates_file(tmp_path):
    """Test LaTeX export."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_path = tmp_path / "table.tex"

    export_to_latex(df, output_path)

    assert output_path.exists()
    assert ".tex" in str(output_path)


def test_export_to_markdown_creates_file(tmp_path):
    """Test Markdown export."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_path = tmp_path / "table.md"

    export_to_markdown(df, output_path)

    assert output_path.exists()


def test_generate_ablation_table_returns_dataframe():
    """Test ablation table generation."""
    ablation = {
        "synthesizer_effect": {
            "CER": {
                "bas_mean": 0.10,
                "sgs_mean": 0.15,
                "difference": -0.05,
            },
            "WER": {
                "bas_mean": 0.28,
                "sgs_mean": 0.38,
                "difference": -0.10,
            },
        }
    }

    table = generate_ablation_table(ablation)

    assert isinstance(table, pd.DataFrame)
