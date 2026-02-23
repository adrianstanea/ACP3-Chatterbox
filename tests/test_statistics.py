"""Tests for statistical testing functions."""

import pytest
import numpy as np
import pandas as pd
from report_lib.statistics import (
    compute_effect_size,
    run_ttest,
    compare_conditions,
)


def test_compute_effect_size_returns_float():
    """Test effect size computation."""
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    effect_size = compute_effect_size(group1, group2)

    assert isinstance(effect_size, float)
    assert effect_size < 0  # group1 has lower mean


def test_run_ttest_returns_dict():
    """Test t-test execution."""
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    result = run_ttest(group1, group2)

    assert isinstance(result, dict)
    assert "t_statistic" in result
    assert "p_value" in result
    assert "effect_size" in result
    assert "significant" in result


def test_run_ttest_significant_difference():
    """Test t-test detects significant differences."""
    group1 = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    group2 = np.array([5.0, 5.1, 5.2, 5.3, 5.4])

    result = run_ttest(group1, group2)

    assert result["significant"] == True  # Use == for numpy bool compatibility
    assert result["p_value"] < 0.05


def test_compare_conditions_returns_dict():
    """Test pairwise condition comparison."""
    metrics_dict = {
        "cond1": pd.DataFrame(
            {"CER": [0.1, 0.12, 0.11], "WER": [0.25, 0.27, 0.26]}
        ),
        "cond2": pd.DataFrame(
            {"CER": [0.15, 0.17, 0.16], "WER": [0.30, 0.32, 0.31]}
        ),
    }

    result = compare_conditions(metrics_dict, "CER")

    assert isinstance(result, dict)
    assert "cond1_vs_cond2" in result
