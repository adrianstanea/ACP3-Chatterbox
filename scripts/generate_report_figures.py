#!/usr/bin/env python3
"""
Generate academic report figures and statistics.

Creates:
1. LaTeX-compatible tables with mean ± std format
2. 2-3 key figures (bar charts, distributions)
3. Markdown tables for documentation

Usage:
    python scripts/generate_report_figures.py \
        --experiment-dir experiments/2026-02-22 \
        --output-dir reports
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures and statistics")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory with metrics/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: experiment-dir/reports)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (useful if matplotlib not available)",
    )
    return parser.parse_args()


def load_all_metrics(metrics_dir: Path) -> Dict[str, List[Dict]]:
    """Load all metrics CSVs into a dict by condition."""
    all_metrics = {}

    for csv_path in sorted(metrics_dir.glob("*.csv")):
        condition = csv_path.stem
        samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                sample = {}
                for key, value in row.items():
                    try:
                        sample[key] = float(value)
                    except (ValueError, TypeError):
                        sample[key] = value
                samples.append(sample)

        all_metrics[condition] = samples

    return all_metrics


def compute_condition_stats(samples: List[Dict], metrics: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for each metric in a condition."""
    stats = {}

    for metric in metrics:
        values = []
        for sample in samples:
            val = sample.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass

        if values:
            stats[metric] = (np.mean(values), np.std(values))
        else:
            stats[metric] = (float("nan"), float("nan"))

    return stats


def parse_condition(condition: str) -> Dict[str, str]:
    """Parse condition name into components."""
    parts = condition.split("_")
    return {
        "speaker": parts[0].upper() if parts else "",
        "exaggeration": parts[1].replace("exag", "") if len(parts) > 1 else "",
        "cfg": parts[2].replace("cfg", "") if len(parts) > 2 else "",
    }


def generate_latex_table(
    all_stats: Dict[str, Dict[str, Tuple[float, float]]],
    metrics: List[str],
    metric_labels: Dict[str, str],
) -> str:
    """Generate LaTeX table with mean ± std format."""

    # Build header
    n_metrics = len(metrics)
    header = r"\begin{table}[htbp]" + "\n"
    header += r"\centering" + "\n"
    header += r"\caption{Objective evaluation metrics for Romanian TTS (mean $\pm$ std)}" + "\n"
    header += r"\label{tab:metrics}" + "\n"
    header += r"\begin{tabular}{l" + "c" * n_metrics + "}\n"
    header += r"\toprule" + "\n"

    # Column headers
    col_headers = ["Condition"]
    for metric in metrics:
        col_headers.append(metric_labels.get(metric, metric))
    header += " & ".join(col_headers) + r" \\" + "\n"
    header += r"\midrule" + "\n"

    # Data rows
    rows = []
    for condition in sorted(all_stats.keys()):
        parsed = parse_condition(condition)
        row_label = f"{parsed['speaker']} (exag={parsed['exaggeration']}, cfg={parsed['cfg']})"

        row_values = [row_label]
        stats = all_stats[condition]

        for metric in metrics:
            mean, std = stats.get(metric, (float("nan"), float("nan")))
            if np.isnan(mean):
                row_values.append("--")
            else:
                row_values.append(f"${mean:.3f} \\pm {std:.3f}$")

        rows.append(" & ".join(row_values) + r" \\")

    # Footer
    footer = r"\bottomrule" + "\n"
    footer += r"\end{tabular}" + "\n"
    footer += r"\end{table}" + "\n"

    return header + "\n".join(rows) + "\n" + footer


def generate_markdown_table(
    all_stats: Dict[str, Dict[str, Tuple[float, float]]],
    metrics: List[str],
    metric_labels: Dict[str, str],
) -> str:
    """Generate Markdown table with mean ± std format."""

    # Header
    headers = ["Condition"]
    for metric in metrics:
        headers.append(metric_labels.get(metric, metric))

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for condition in sorted(all_stats.keys()):
        parsed = parse_condition(condition)
        row_label = f"{parsed['speaker']} (exag={parsed['exaggeration']}, cfg={parsed['cfg']})"

        row_values = [row_label]
        stats = all_stats[condition]

        for metric in metrics:
            mean, std = stats.get(metric, (float("nan"), float("nan")))
            if np.isnan(mean):
                row_values.append("--")
            else:
                row_values.append(f"{mean:.3f} ± {std:.3f}")

        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def generate_speaker_comparison_table(
    all_stats: Dict[str, Dict[str, Tuple[float, float]]],
    metrics: List[str],
    metric_labels: Dict[str, str],
) -> Tuple[str, str]:
    """Generate aggregated stats by speaker (LaTeX and Markdown)."""

    # Aggregate by speaker
    speaker_samples = {"BAS": [], "SGS": []}

    for condition, stats in all_stats.items():
        speaker = condition.split("_")[0].upper()
        if speaker in speaker_samples:
            speaker_samples[speaker].append(stats)

    # Compute aggregated stats per speaker
    speaker_stats = {}
    for speaker, stats_list in speaker_samples.items():
        speaker_stats[speaker] = {}
        for metric in metrics:
            values = []
            for stats in stats_list:
                mean, _ = stats.get(metric, (float("nan"), float("nan")))
                if not np.isnan(mean):
                    values.append(mean)
            if values:
                speaker_stats[speaker][metric] = (np.mean(values), np.std(values))
            else:
                speaker_stats[speaker][metric] = (float("nan"), float("nan"))

    # Generate tables
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Speaker-level metrics comparison}")
    latex_lines.append(r"\label{tab:speaker_comparison}")
    latex_lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex_lines.append(r"\toprule")

    headers = ["Speaker"] + [metric_labels.get(m, m) for m in metrics]
    latex_lines.append(" & ".join(headers) + r" \\")
    latex_lines.append(r"\midrule")

    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for speaker in ["BAS", "SGS"]:
        row = [speaker]
        md_row = [speaker]
        for metric in metrics:
            mean, std = speaker_stats[speaker].get(metric, (float("nan"), float("nan")))
            if np.isnan(mean):
                row.append("--")
                md_row.append("--")
            else:
                row.append(f"${mean:.3f} \\pm {std:.3f}$")
                md_row.append(f"{mean:.3f} ± {std:.3f}")
        latex_lines.append(" & ".join(row) + r" \\")
        md_lines.append("| " + " | ".join(md_row) + " |")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines), "\n".join(md_lines)


def generate_figures(
    all_metrics: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Generate key figures for the report."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")
        return

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
    })

    # =========================================================================
    # Figure 1: UTMOS by Condition (Bar Chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = sorted(all_metrics.keys())
    utmos_means = []
    utmos_stds = []

    for condition in conditions:
        samples = all_metrics[condition]
        values = [s.get("UTMOS_Synthesized", float("nan")) for s in samples]
        values = [v for v in values if not np.isnan(v)]
        utmos_means.append(np.mean(values) if values else 0)
        utmos_stds.append(np.std(values) if values else 0)

    x = np.arange(len(conditions))
    colors = ["#2563eb" if c.startswith("bas") else "#dc2626" for c in conditions]

    bars = ax.bar(x, utmos_means, yerr=utmos_stds, capsize=4, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], rotation=0)
    ax.set_ylabel("UTMOS Score")
    ax.set_xlabel("Condition")
    ax.set_title("Speech Quality (UTMOS) by Experimental Condition")
    ax.set_ylim(0, 5)

    # Add legend for speakers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", label="BAS", edgecolor="black"),
        Patch(facecolor="#dc2626", label="SGS", edgecolor="black"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_utmos_by_condition.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_utmos_by_condition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig_utmos_by_condition.png/pdf")

    # =========================================================================
    # Figure 2: CER vs WER Scatter (shows intelligibility correlation)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    for condition in conditions:
        samples = all_metrics[condition]
        cer_vals = [s.get("CER", float("nan")) for s in samples]
        wer_vals = [s.get("WER", float("nan")) for s in samples]

        # Filter NaN pairs
        pairs = [(c, w) for c, w in zip(cer_vals, wer_vals) if not (np.isnan(c) or np.isnan(w))]
        if not pairs:
            continue

        cer_vals, wer_vals = zip(*pairs)

        color = "#2563eb" if condition.startswith("bas") else "#dc2626"
        marker = "o" if "exag0.3" in condition else "^"

        ax.scatter(cer_vals, wer_vals, c=color, marker=marker, alpha=0.5, s=30, label=condition)

    ax.set_xlabel("Character Error Rate (CER)")
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_title("Intelligibility: CER vs WER Distribution")

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="CER = WER")
    ax.set_xlim(0, max(0.5, ax.get_xlim()[1]))
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))

    plt.tight_layout()
    fig.savefig(output_dir / "fig_cer_wer_scatter.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_cer_wer_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig_cer_wer_scatter.png/pdf")

    # =========================================================================
    # Figure 3: Parameter Effect Heatmap (Exaggeration vs CFG)
    # =========================================================================
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cbar_ax = fig.add_subplot(gs[0, 2])

    for idx, speaker in enumerate(["bas", "sgs"]):
        ax = axes[idx]

        # Build heatmap data
        exag_values = ["0.3", "0.7"]
        cfg_values = ["0.3", "0.5"]

        heatmap_data = np.zeros((len(exag_values), len(cfg_values)))

        for i, exag in enumerate(exag_values):
            for j, cfg in enumerate(cfg_values):
                condition = f"{speaker}_exag{exag}_cfg{cfg}"
                if condition in all_metrics:
                    samples = all_metrics[condition]
                    utmos_vals = [s.get("UTMOS_Synthesized", float("nan")) for s in samples]
                    utmos_vals = [v for v in utmos_vals if not np.isnan(v)]
                    heatmap_data[i, j] = np.mean(utmos_vals) if utmos_vals else 0

        im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=2.5, vmax=4.5, aspect="auto")

        # Add text annotations
        for i in range(len(exag_values)):
            for j in range(len(cfg_values)):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                              ha="center", va="center", color="black", fontsize=12, fontweight="bold")

        ax.set_xticks(np.arange(len(cfg_values)))
        ax.set_yticks(np.arange(len(exag_values)))
        ax.set_xticklabels([f"CFG={v}" for v in cfg_values])
        ax.set_yticklabels([f"Exag={v}" for v in exag_values])
        ax.set_title(f"Speaker: {speaker.upper()}")

    fig.suptitle("Parameter Effect on Speech Quality", fontsize=13)

    # Add colorbar in its own subplot
    fig.colorbar(im, cax=cbar_ax, label="UTMOS Score")

    fig.savefig(str(output_dir / "fig_parameter_heatmap.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(output_dir / "fig_parameter_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig_parameter_heatmap.png/pdf")


def main():
    args = parse_args()

    metrics_dir = args.experiment_dir / "metrics"

    if not metrics_dir.exists():
        print(f"ERROR: Metrics directory not found: {metrics_dir}")
        print("Run compute_experiment_metrics.py first.")
        sys.exit(1)

    output_dir = args.output_dir or (args.experiment_dir / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Metrics directory: {metrics_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load all metrics
    all_metrics = load_all_metrics(metrics_dir)

    if not all_metrics:
        print("ERROR: No metrics found")
        sys.exit(1)

    print(f"Loaded metrics for {len(all_metrics)} conditions")

    # Define metrics to include
    metrics_to_report = [
        "CER",
        "WER",
        "UTMOS_Synthesized",
        "Speaker_Cosine_Similarity",
    ]

    metric_labels = {
        "CER": "CER",
        "WER": "WER",
        "UTMOS_Synthesized": "UTMOS",
        "Speaker_Cosine_Similarity": "Spk. Sim.",
        "Log_F0_RMSE": "F0 RMSE",
        "MCD": "MCD",
    }

    # Compute stats for all conditions
    all_stats = {}
    for condition, samples in all_metrics.items():
        all_stats[condition] = compute_condition_stats(samples, metrics_to_report)

    # Generate tables
    print("\nGenerating tables...")

    # Full condition table
    latex_table = generate_latex_table(all_stats, metrics_to_report, metric_labels)
    md_table = generate_markdown_table(all_stats, metrics_to_report, metric_labels)

    with open(output_dir / "table_full_metrics.tex", "w") as f:
        f.write(latex_table)
    print("Generated: table_full_metrics.tex")

    with open(output_dir / "table_full_metrics.md", "w") as f:
        f.write("# Full Metrics by Condition\n\n")
        f.write(md_table)
    print("Generated: table_full_metrics.md")

    # Speaker comparison table
    latex_speaker, md_speaker = generate_speaker_comparison_table(all_stats, metrics_to_report, metric_labels)

    with open(output_dir / "table_speaker_comparison.tex", "w") as f:
        f.write(latex_speaker)
    print("Generated: table_speaker_comparison.tex")

    with open(output_dir / "table_speaker_comparison.md", "w") as f:
        f.write("# Speaker Comparison\n\n")
        f.write(md_speaker)
    print("Generated: table_speaker_comparison.md")

    # Generate figures
    if not args.no_figures:
        print("\nGenerating figures...")
        generate_figures(all_metrics, output_dir)

    # Summary JSON
    summary = {
        "conditions": list(all_stats.keys()),
        "metrics": metrics_to_report,
        "stats": {
            cond: {
                metric: {"mean": stats[0], "std": stats[1]}
                for metric, stats in condition_stats.items()
            }
            for cond, condition_stats in all_stats.items()
        },
    }

    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Generated: metrics_summary.json")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
