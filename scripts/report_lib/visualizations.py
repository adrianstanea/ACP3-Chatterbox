"""Visualization functions for academic report generation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any
import warnings

# Set academic style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


def plot_speaker_distribution(
    speaker_dist: Dict[str, int], output_dir: Path, dpi: int = 300
) -> Path:
    """Plot speaker sample distribution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    speakers = list(speaker_dist.keys())
    counts = list(speaker_dist.values())

    ax.bar(range(len(speakers)), counts, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Speaker Distribution in Dataset")
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=45, ha="right")

    plt.tight_layout()

    output_path = output_dir / "speaker_distribution.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_text_statistics(
    text_stats: Dict[str, float], output_dir: Path, dpi: int = 300
) -> Path:
    """Plot text length and diversity statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Character counts
    char_data = [
        text_stats.get("char_count_min", 0),
        text_stats.get("char_count_mean", 0),
        text_stats.get("char_count_max", 0),
    ]
    char_labels = ["Min", "Mean", "Max"]
    axes[0].bar(char_labels, char_data, color="#2ca02c", alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Character Count")
    axes[0].set_title("Text Length Statistics")

    # Word counts (if available)
    if "word_count_mean" in text_stats:
        word_data = [
            text_stats.get("word_count_mean", 0),
            text_stats.get("word_count_std", 0),
        ]
        word_labels = ["Mean", "Std Dev"]
        axes[1].bar(word_labels, word_data, color="#ff7f0e", alpha=0.7, edgecolor="black")
        axes[1].set_ylabel("Word Count")
        axes[1].set_title("Word Count Statistics")

    plt.tight_layout()
    output_path = output_dir / "text_statistics.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_loss_curves(
    trainer_state: Dict[str, Any], output_dir: Path, dpi: int = 300
) -> Path:
    """Plot training loss curves from log_history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if "log_history" in trainer_state:
        log_history = trainer_state["log_history"]
        steps = [log["step"] for log in log_history if "step" in log]
        losses = [log["loss"] for log in log_history if "loss" in log]
        
        if steps and losses:
            ax.plot(steps, losses, marker="o", linestyle="-", label="Training Loss", color="#1f77b4", linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_performance_comparison(
    performance: Dict[str, Dict], output_dir: Path, dpi: int = 300
) -> Path:
    """Plot side-by-side metric comparison across conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        conditions = []
        means = []
        stds = []

        for cond, cond_data in performance.items():
            if metric in cond_data:
                conditions.append(cond.replace("_", "\n"))
                means.append(cond_data[metric]["mean"])
                stds.append(cond_data[metric]["std"])

        if means:
            x = range(len(conditions))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by Condition")
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "performance_comparison.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_ablation_all_conditions_grouped(
    ablation: Dict[str, Any], output_dir: Path, dpi: int = 300
) -> Path:
    """Version 1: Plot all 8 conditions with smart y-axis scaling.
    
    Creates 3 separate figures:
    - Figure 1: CER and WER (2 subplots, full scale)
    - Figure 2: UTMOS_Synthesized (1 subplot, zoomed scale)
    - Figure 3: Speaker_Cosine_Similarity (1 subplot, zoomed scale)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_order = [
        "bas_exag0.3_cfg0.3", "bas_exag0.3_cfg0.5", "bas_exag0.7_cfg0.3", "bas_exag0.7_cfg0.5",
        "sgs_exag0.3_cfg0.3", "sgs_exag0.3_cfg0.5", "sgs_exag0.7_cfg0.3", "sgs_exag0.7_cfg0.5",
    ]
    
    # Color scheme: BAS = blue shades, SGS = orange shades
    colors = ["#1f77b4", "#4587d0", "#2ca02c", "#4eb840",  # BAS conditions
              "#ff7f0e", "#ffaa40", "#d62728", "#e85555"]  # SGS conditions
    
    # Figure 1: CER and WER (full scale)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, metric in enumerate(["CER", "WER"]):
        ax = axes1[ax_idx]
        values = []
        labels = []
        
        for condition in condition_order:
            if "by_condition" in ablation and condition in ablation["by_condition"]:
                values.append(ablation["by_condition"][condition].get(metric, 0))
                label = condition.replace("bas_", "B").replace("sgs_", "S").replace("exag", "E").replace("cfg", "C")
                labels.append(label)
        
        if values:
            x = np.arange(len(labels))
            ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#1f77b4", alpha=0.7, edgecolor="black", label="BAS"),
                Patch(facecolor="#ff7f0e", alpha=0.7, edgecolor="black", label="SGS"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    output_path1 = output_dir / "ablation_v1_cer_wer.png"
    plt.savefig(output_path1, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    # Figure 2: UTMOS_Synthesized (smart zoom)
    fig2, ax = plt.subplots(figsize=(12, 5))
    metric = "UTMOS_Synthesized"
    values = []
    labels = []
    
    for condition in condition_order:
        if "by_condition" in ablation and condition in ablation["by_condition"]:
            values.append(ablation["by_condition"][condition].get(metric, 0))
            label = condition.replace("bas_", "B").replace("sgs_", "S").replace("exag", "E").replace("cfg", "C")
            labels.append(label)
    
    if values:
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        
        # Smart y-axis scaling: zoom into reasonable range
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 0.15
        ax.set_ylim(max(0, min_val - margin), min(5, max_val + margin))
        
        ax.grid(True, alpha=0.3, axis="y")
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", alpha=0.7, edgecolor="black", label="BAS"),
            Patch(facecolor="#ff7f0e", alpha=0.7, edgecolor="black", label="SGS"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    output_path2 = output_dir / "ablation_v1_utmos.png"
    plt.savefig(output_path2, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    # Figure 3: Speaker_Cosine_Similarity (smart zoom)
    fig3, ax = plt.subplots(figsize=(12, 5))
    metric = "Speaker_Cosine_Similarity"
    values = []
    labels = []
    
    for condition in condition_order:
        if "by_condition" in ablation and condition in ablation["by_condition"]:
            values.append(ablation["by_condition"][condition].get(metric, 0))
            label = condition.replace("bas_", "B").replace("sgs_", "S").replace("exag", "E").replace("cfg", "C")
            labels.append(label)
    
    if values:
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        
        # Smart y-axis scaling: zoom into reasonable range
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 0.15
        ax.set_ylim(max(0, min_val - margin), min(1.0, max_val + margin))
        
        ax.grid(True, alpha=0.3, axis="y")
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", alpha=0.7, edgecolor="black", label="BAS"),
            Patch(facecolor="#ff7f0e", alpha=0.7, edgecolor="black", label="SGS"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    output_path3 = output_dir / "ablation_v1_speaker_similarity.png"
    plt.savefig(output_path3, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path1


def plot_ablation_by_synthesizer(
    ablation: Dict[str, Any], output_dir: Path, dpi: int = 300
) -> Path:
    """Version 2: Separate plots for each synthesizer showing parameter effects.
    
    Two panels (BAS, SGS), each showing exag and cfg effects on all 4 metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    for synth_idx, synthesizer in enumerate(["BAS", "SGS"]):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[synth_idx, metric_idx]
            
            if "main_effects" in ablation and metric in ablation["main_effects"]:
                effects = ablation["main_effects"][metric]
                
                # Get exag and cfg values for this synthesizer
                synth_lower = synthesizer.lower()
                exag03 = effects.get(f"{synth_lower}_exag03", 0)
                exag07 = effects.get(f"{synth_lower}_exag07", 0)
                cfg03 = effects.get(f"{synth_lower}_cfg03", 0)
                cfg05 = effects.get(f"{synth_lower}_cfg05", 0)
                
                categories = ["Exag 0.3", "Exag 0.7", "Cfg 0.3", "Cfg 0.5"]
                values = [exag03, exag07, cfg03, cfg05]
                colors = ["#2ca02c", "#4eb840", "#d62728", "#e85555"]  # Green for exag, red for cfg
                
                x = np.arange(len(categories))
                ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
                ax.set_ylabel(metric, fontsize=10)
                ax.set_title(f"{metric}", fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
        
        # Add synthesizer label to the left
        axes[synth_idx, 0].text(-0.5, 0.5, synthesizer, transform=axes[synth_idx, 0].transAxes,
                               fontsize=14, fontweight="bold", va="center", ha="right")
    
    plt.suptitle("Version 2: Parameter Effects by Synthesizer", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path = output_dir / "ablation_v2_by_synthesizer.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path


def plot_ablation_interaction_heatmap(
    ablation: Dict[str, Any], output_dir: Path, dpi: int = 300
) -> Path:
    """Version 3: Heatmaps showing 2×2 interaction of exag×cfg for each synthesizer.
    
    For each synthesizer and metric, shows a 2×2 heatmap with:
    - Rows: exaggeration (0.3, 0.7)
    - Cols: guidance scale (0.3, 0.5)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    for synth_idx, synthesizer in enumerate(["BAS", "SGS"]):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[synth_idx, metric_idx]
            
            if "by_condition" in ablation:
                # Extract 2×2 matrix: exag (rows) × cfg (cols)
                prefix = synthesizer.lower() + "_"
                matrix = np.array([
                    [
                        ablation["by_condition"].get(f"{prefix}exag0.3_cfg0.3", {}).get(metric, 0),
                        ablation["by_condition"].get(f"{prefix}exag0.3_cfg0.5", {}).get(metric, 0),
                    ],
                    [
                        ablation["by_condition"].get(f"{prefix}exag0.7_cfg0.3", {}).get(metric, 0),
                        ablation["by_condition"].get(f"{prefix}exag0.7_cfg0.5", {}).get(metric, 0),
                    ]
                ])
                
                # Create heatmap
                im = ax.imshow(matrix, cmap="RdYlGn_r" if metric in ["CER", "WER"] else "RdYlGn",
                             aspect="auto", vmin=np.min(matrix), vmax=np.max(matrix))
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                                     ha="center", va="center", color="black", fontsize=11, fontweight="bold")
                
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Cfg 0.3", "Cfg 0.5"], fontsize=9)
                ax.set_yticklabels(["Exag 0.3", "Exag 0.7"], fontsize=9)
                ax.set_ylabel("Exaggeration", fontsize=10)
                ax.set_xlabel("Guidance Scale", fontsize=10)
                ax.set_title(f"{metric}", fontsize=11)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(metric, fontsize=9)
        
        # Add synthesizer label
        axes[synth_idx, 0].text(-0.5, 0.5, synthesizer, transform=axes[synth_idx, 0].transAxes,
                               fontsize=14, fontweight="bold", va="center", ha="right")
    
    plt.suptitle("Version 3: Interaction Heatmaps (Exag × Cfg)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path = output_dir / "ablation_v3_interaction_heatmap.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path


def plot_ablation_effects(
    ablation: Dict[str, Any], output_dir: Path, dpi: int = 300
) -> Path:
    """Legacy function: calls all three versions."""
    plot_ablation_all_conditions_grouped(ablation, output_dir, dpi)
    plot_ablation_by_synthesizer(ablation, output_dir, dpi)
    plot_ablation_interaction_heatmap(ablation, output_dir, dpi)
    return output_dir / "ablation_v1_all_conditions.png"
