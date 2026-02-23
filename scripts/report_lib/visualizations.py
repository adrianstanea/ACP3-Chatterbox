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
    """Plot training/validation loss curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if "training_loss" in trainer_state:
        losses = trainer_state["training_loss"]
        epochs = range(len(losses))
        ax.plot(epochs, losses, marker="o", linestyle="-", label="Training Loss", color="#1f77b4")

    if "validation_loss" in trainer_state:
        val_losses = trainer_state["validation_loss"]
        epochs = range(len(val_losses))
        ax.plot(
            epochs, val_losses, marker="s", linestyle="-", label="Validation Loss", color="#ff7f0e"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
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
