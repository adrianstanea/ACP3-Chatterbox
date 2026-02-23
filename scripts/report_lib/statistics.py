"""Statistical testing functions for academic reports."""

import numpy as np
from scipy import stats
from typing import Dict


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_ttest(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """Run independent t-test."""
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = compute_effect_size(group1, group2)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "significant": p_value < 0.05,
    }


def compare_conditions(metrics_dict: Dict, metric_name: str) -> Dict:
    """Compare all pairs of conditions on a metric."""
    results = {}
    conditions = list(metrics_dict.keys())

    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i + 1 :]:
            key = f"{cond1}_vs_{cond2}"

            if (
                metric_name in metrics_dict[cond1].columns
                and metric_name in metrics_dict[cond2].columns
            ):
                data1 = metrics_dict[cond1][metric_name].dropna().values
                data2 = metrics_dict[cond2][metric_name].dropna().values

                if len(data1) > 0 and len(data2) > 0:
                    results[key] = run_ttest(data1, data2)

    return results
