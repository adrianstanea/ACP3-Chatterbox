#!/usr/bin/env python3
"""
Select best samples based on objective metrics.

Selects top samples for GitHub Pages gallery:
- 40 total samples (20 per speaker)
- 2 samples per experiment condition (5 conditions per speaker x 2 samples = 10 per speaker... wait)
- Actually: 4 conditions per speaker, so 5 samples per condition to get 20 per speaker

Updated requirement: 2 samples per experiment condition
- 8 conditions total (2 speakers x 4 parameter combos)
- 2 samples per condition = 16 samples
- To get 40 total with 2 per experiment: need 5 samples per condition for 20 per speaker

Final: 20 samples per speaker, 5 samples per condition (8 conditions total = 40 samples)

Usage:
    python scripts/select_best_samples.py \
        --experiment-dir experiments/2026-02-22 \
        --samples-per-condition 5 \
        --output selected_samples.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best samples for gallery")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory with metrics/",
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=5,
        help="Number of best samples to select per condition (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: experiment-dir/selected_samples.json)",
    )
    return parser.parse_args()


def load_metrics_csv(csv_path: Path) -> List[Dict]:
    """Load metrics CSV and return list of sample dicts."""
    samples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")

        for row in reader:
            sample = {
                "audio_file": row.get("Audio_File", ""),
                "original_transcription": row.get("Original_Transcription", ""),
                "synthesized_transcription": row.get("Synthesized_Transcription", ""),
            }

            # Parse numeric metrics
            for metric in ["CER", "WER", "UTMOS_Synthesized", "Log_F0_RMSE", "MCD", "Speaker_Cosine_Similarity"]:
                try:
                    sample[metric] = float(row.get(metric, "nan"))
                except (ValueError, TypeError):
                    sample[metric] = float("nan")

            samples.append(sample)

    return samples


def compute_composite_score(sample: Dict) -> float:
    """
    Compute composite quality score for sample ranking.

    Higher score = better sample.

    Scoring logic:
    - UTMOS (quality): Higher is better, weight 0.3
    - CER (intelligibility): Lower is better, weight 0.25
    - WER (intelligibility): Lower is better, weight 0.25
    - Speaker Similarity: Higher is better, weight 0.2

    We ignore MCD and F0 RMSE for ranking as they're less reliable for cross-speaker comparison.
    """
    score = 0.0
    weights_sum = 0.0

    # UTMOS: normalized to 0-1 (assuming range 1-5)
    utmos = sample.get("UTMOS_Synthesized", float("nan"))
    if not np.isnan(utmos):
        utmos_norm = (utmos - 1.0) / 4.0  # Map 1-5 to 0-1
        utmos_norm = max(0, min(1, utmos_norm))
        score += 0.3 * utmos_norm
        weights_sum += 0.3

    # CER: lower is better, so use (1 - CER)
    cer = sample.get("CER", float("nan"))
    if not np.isnan(cer):
        cer_score = 1.0 - min(cer, 1.0)  # Cap at 1.0
        score += 0.25 * cer_score
        weights_sum += 0.25

    # WER: lower is better
    wer = sample.get("WER", float("nan"))
    if not np.isnan(wer):
        wer_score = 1.0 - min(wer, 1.0)
        score += 0.25 * wer_score
        weights_sum += 0.25

    # Speaker similarity: higher is better (already 0-1)
    spk_sim = sample.get("Speaker_Cosine_Similarity", float("nan"))
    if not np.isnan(spk_sim):
        spk_sim_norm = max(0, min(1, spk_sim))
        score += 0.2 * spk_sim_norm
        weights_sum += 0.2

    # Normalize by actual weights used
    if weights_sum > 0:
        return score / weights_sum
    else:
        return 0.0


def select_best_samples(
    metrics_dir: Path,
    samples_per_condition: int,
) -> Dict[str, List[Dict]]:
    """
    Select best samples from each condition.

    Returns:
        Dict mapping condition name to list of selected samples.
    """
    selected = {}

    for csv_path in sorted(metrics_dir.glob("*.csv")):
        condition = csv_path.stem

        # Load samples
        samples = load_metrics_csv(csv_path)

        if not samples:
            print(f"WARNING: No samples in {csv_path}")
            continue

        # Compute composite scores
        for sample in samples:
            sample["composite_score"] = compute_composite_score(sample)

        # Sort by composite score (descending)
        samples_sorted = sorted(
            samples,
            key=lambda s: s["composite_score"],
            reverse=True
        )

        # Select top N
        selected[condition] = samples_sorted[:samples_per_condition]

        print(f"{condition}: selected {len(selected[condition])}/{len(samples)} samples")

        # Print score range
        if selected[condition]:
            best_score = selected[condition][0]["composite_score"]
            worst_score = selected[condition][-1]["composite_score"]
            print(f"  Score range: {best_score:.3f} - {worst_score:.3f}")

    return selected


def summarize_selection(selected: Dict[str, List[Dict]]) -> Dict:
    """Generate summary statistics for selected samples."""

    # Group by speaker
    by_speaker = {"bas": [], "sgs": []}

    for condition, samples in selected.items():
        speaker = condition.split("_")[0].lower()
        if speaker in by_speaker:
            by_speaker[speaker].extend(samples)

    # Compute stats per speaker
    stats = {}

    for speaker, samples in by_speaker.items():
        if not samples:
            continue

        stats[speaker] = {
            "count": len(samples),
            "avg_composite_score": np.mean([s["composite_score"] for s in samples]),
            "avg_utmos": np.mean([s["UTMOS_Synthesized"] for s in samples if not np.isnan(s["UTMOS_Synthesized"])]),
            "avg_cer": np.mean([s["CER"] for s in samples if not np.isnan(s["CER"])]),
            "avg_wer": np.mean([s["WER"] for s in samples if not np.isnan(s["WER"])]),
            "avg_speaker_sim": np.mean([s["Speaker_Cosine_Similarity"] for s in samples if not np.isnan(s["Speaker_Cosine_Similarity"])]),
        }

    # Overall stats
    all_samples = by_speaker["bas"] + by_speaker["sgs"]
    stats["overall"] = {
        "total_samples": len(all_samples),
        "conditions": len(selected),
        "samples_per_condition": len(all_samples) // max(len(selected), 1),
    }

    return stats


def main():
    args = parse_args()

    metrics_dir = args.experiment_dir / "metrics"

    if not metrics_dir.exists():
        print(f"ERROR: Metrics directory not found: {metrics_dir}")
        print("Run compute_experiment_metrics.py first to generate metrics.")
        sys.exit(1)

    output_path = args.output or (args.experiment_dir / "selected_samples.json")

    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Samples per condition: {args.samples_per_condition}")
    print(f"Output: {output_path}")
    print()

    # Select best samples
    selected = select_best_samples(metrics_dir, args.samples_per_condition)

    if not selected:
        print("ERROR: No samples selected")
        sys.exit(1)

    # Generate summary
    stats = summarize_selection(selected)

    # Prepare output
    output_data = {
        "config": {
            "samples_per_condition": args.samples_per_condition,
            "experiment_dir": str(args.experiment_dir),
        },
        "summary": stats,
        "selected": selected,
    }

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats['overall']['total_samples']}")
    print(f"Conditions: {stats['overall']['conditions']}")

    for speaker in ["bas", "sgs"]:
        if speaker in stats:
            s = stats[speaker]
            print(f"\n{speaker.upper()}:")
            print(f"  Samples: {s['count']}")
            print(f"  Avg composite score: {s['avg_composite_score']:.3f}")
            print(f"  Avg UTMOS: {s['avg_utmos']:.2f}")
            print(f"  Avg CER: {s['avg_cer']:.3f}")
            print(f"  Avg WER: {s['avg_wer']:.3f}")
            print(f"  Avg Speaker Sim: {s['avg_speaker_sim']:.3f}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
