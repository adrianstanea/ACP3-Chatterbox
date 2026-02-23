"""Table generation functions for academic reports."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any


def generate_metrics_table(performance: Dict[str, Dict]) -> pd.DataFrame:
    """Generate metrics table from performance analysis."""
    rows = []

    for condition, metrics in performance.items():
        row = {"Condition": condition}

        for metric, stats in metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                mean = stats["mean"]
                std = stats.get("std", 0)
                row[f"{metric}"] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[metric] = stats

        rows.append(row)

    return pd.DataFrame(rows)


def generate_dataset_summary_table(dataset_chars: Dict[str, Any]) -> pd.DataFrame:
    """Generate dataset characterization summary."""
    data = {
        "Metric": [
            "Total Samples",
            "Unique Speakers",
            "Mean Samples/Speaker",
            "Mean Text Length (chars)",
            "Mean Duration (sec)",
        ],
        "Value": [
            dataset_chars.get("metadata", {}).get("total_samples", "N/A"),
            dataset_chars.get("metadata", {}).get("unique_speakers", "N/A"),
            f"{dataset_chars.get('speaker_distribution', {}).get('mean_samples_per_speaker', 0):.1f}",
            f"{dataset_chars.get('text_statistics', {}).get('char_count_mean', 0):.1f}",
            f"{dataset_chars.get('audio_properties', {}).get('duration_mean', 0):.2f}",
        ],
    }

    return pd.DataFrame(data)


def export_to_latex(
    df: pd.DataFrame, output_path: Path, caption: str = "", label: str = ""
) -> None:
    """Export DataFrame to LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_str = df.to_latex(index=False, escape=True)

    if caption or label:
        latex_str = latex_str.replace(
            "\\begin{tabular}", "\\begin{table}\n\\centering\n\\begin{tabular}"
        )
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")

        if caption:
            latex_str = latex_str.replace(
                "\\end{table}", f"\\caption{{{caption}}}\n\\end{{table}}"
            )
        if label:
            latex_str = latex_str.replace("\\end{table}", f"\\label{{{label}}}\n\\end{{table}}")

    with open(output_path, "w") as f:
        f.write(latex_str)


def export_to_markdown(df: pd.DataFrame, output_path: Path) -> None:
    """Export DataFrame to Markdown table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_str = df.to_markdown(index=False)

    with open(output_path, "w") as f:
        f.write(markdown_str)


def generate_ablation_table(ablation: Dict[str, Any]) -> pd.DataFrame:
    """Generate ablation study results table."""
    rows = []

    # Synthesizer effect
    if "synthesizer_effect" in ablation:
        for metric, effects in ablation["synthesizer_effect"].items():
            rows.append(
                {
                    "Parameter": "Synthesizer",
                    "Metric": metric,
                    "BAS": f"{effects.get('bas_mean', 0):.3f}",
                    "SGS": f"{effects.get('sgs_mean', 0):.3f}",
                    "Difference": f"{effects.get('difference', 0):.3f}",
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame()
