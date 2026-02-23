"""Analysis functions for academic report generation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Set


def analyze_dataset_characteristics(metadata: pd.DataFrame, splits: Dict) -> Dict[str, Any]:
    """Analyze dataset composition, text, and audio properties."""
    results = {
        "metadata": {},
        "speaker_distribution": {},
        "text_statistics": {},
        "audio_properties": {},
    }

    # Basic stats
    results["metadata"]["total_samples"] = len(metadata)
    results["metadata"]["unique_speakers"] = metadata["speaker"].nunique()

    # Speaker distribution
    speaker_counts = metadata["speaker"].value_counts().to_dict()
    results["speaker_distribution"] = {
        "total": len(speaker_counts),
        "per_speaker": speaker_counts,
        "mean_samples_per_speaker": float(np.mean(list(speaker_counts.values()))),
        "std_samples_per_speaker": float(np.std(list(speaker_counts.values()))),
    }

    # Text statistics
    text_lengths = metadata["transcription"].str.len()
    word_counts = metadata["transcription"].str.split().str.len()

    results["text_statistics"] = {
        "char_count_mean": float(text_lengths.mean()),
        "char_count_std": float(text_lengths.std()),
        "char_count_min": int(text_lengths.min()),
        "char_count_max": int(text_lengths.max()),
        "word_count_mean": float(word_counts.mean()),
        "word_count_std": float(word_counts.std()),
        "unique_chars": len(set("".join(metadata["transcription"]))),
    }

    # Audio properties (if duration column exists)
    if "duration" in metadata.columns:
        results["audio_properties"] = {
            "duration_mean": float(metadata["duration"].mean()),
            "duration_std": float(metadata["duration"].std()),
            "duration_min": float(metadata["duration"].min()),
            "duration_max": float(metadata["duration"].max()),
        }

    # Split analysis
    results["splits"] = {}
    for split_name in ["train", "val", "test_holdout"]:
        if split_name in splits and isinstance(splits[split_name], pd.DataFrame):
            split_df = splits[split_name]
            results["splits"][split_name] = {
                "samples": len(split_df),
                "speakers": (
                    int(split_df["speaker"].nunique())
                    if "speaker" in split_df.columns
                    else 0
                ),
            }

    # Holdout speakers
    results["holdout_speakers"] = splits.get("test_holdout_speakers", set())

    return results


def analyze_training_dynamics(trainer_state: Dict) -> Dict[str, Any]:
    """Analyze training process and convergence."""
    results = {
        "metadata": {},
        "convergence": {},
        "best_metrics": {},
    }

    # Basic metadata
    results["metadata"]["total_epochs"] = trainer_state.get("epoch", None)
    results["metadata"]["global_steps"] = trainer_state.get("global_step", None)

    # Loss curves (if available in training_loss key)
    if "training_loss" in trainer_state:
        losses = trainer_state["training_loss"]
        results["convergence"]["training_loss_initial"] = float(losses[0]) if losses else None
        results["convergence"]["training_loss_final"] = float(losses[-1]) if losses else None
        results["convergence"]["training_loss_min"] = float(min(losses)) if losses else None

    # Best metrics
    for key in ["best_loss", "best_eval_loss", "best_model_checkpoint"]:
        if key in trainer_state:
            results["best_metrics"][key] = trainer_state[key]

    return results


def analyze_performance(metrics: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze model performance across conditions."""
    from .statistics import compare_conditions

    results = {
        "conditions": {},
        "overall": {},
        "statistical_tests": {},
    }

    metric_names = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]

    for condition, df in metrics.items():
        results["conditions"][condition] = {}

        for metric in metric_names:
            if metric in df.columns:
                values = df[metric].dropna()
                results["conditions"][condition][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                    "n": int(len(values)),
                }

    # Run pairwise comparisons
    for metric in metric_names:
        results["statistical_tests"][metric] = compare_conditions(metrics, metric)

    # Overall statistics
    for metric in metric_names:
        all_means = []
        for condition_data in results["conditions"].values():
            if metric in condition_data:
                all_means.append(condition_data[metric]["mean"])

        if all_means:
            results["overall"][metric] = {
                "mean_of_means": float(np.mean(all_means)),
                "std_of_means": float(np.std(all_means)),
            }

    return results


def analyze_voice_adaptation(metrics: Dict[str, pd.DataFrame], splits: Dict) -> Dict[str, Any]:
    """Analyze 1-shot voice adaptation performance on holdout speakers."""
    results = {
        "holdout_speakers": list(splits.get("test_holdout_speakers", set())),
        "generalization_gap": {},
    }

    # Placeholder: this requires linking metrics to specific speakers
    # Will be populated by metadata matching

    return results


def analyze_ablation_effects(metrics: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze parameter ablation effects."""
    results = {
        "synthesizer_effect": {},
        "exaggeration_effect": {},
        "guidance_effect": {},
        "interactions": {},
    }

    # Group conditions
    bas_conditions = [c for c in metrics.keys() if c.startswith("bas_")]
    sgs_conditions = [c for c in metrics.keys() if c.startswith("sgs_")]

    metric_names = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]

    # Synthesizer effect
    for metric in metric_names:
        bas_means = []
        sgs_means = []

        for cond in bas_conditions:
            if metric in metrics[cond].columns:
                bas_means.append(metrics[cond][metric].mean())
        for cond in sgs_conditions:
            if metric in metrics[cond].columns:
                sgs_means.append(metrics[cond][metric].mean())

        if bas_means and sgs_means:
            results["synthesizer_effect"][metric] = {
                "bas_mean": float(np.mean(bas_means)),
                "sgs_mean": float(np.mean(sgs_means)),
                "difference": float(np.mean(bas_means) - np.mean(sgs_means)),
            }

    return results
