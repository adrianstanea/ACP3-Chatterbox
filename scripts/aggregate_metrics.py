#!/usr/bin/env python3
"""
Metrics Aggregation Script
Reads per-sample metrics CSVs from TTS-Evaluation-Pipeline and computes mean/std per condition.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Metrics columns to aggregate (numeric columns from evaluation pipeline)
METRICS_TO_AGGREGATE = [
    'CER',
    'WER',
    'UTMOS_Synthesized',
    'Log_F0_RMSE',
    'MCD',
    'Speaker_Cosine_Similarity',
]

# Expected CSV columns from TTS-Evaluation-Pipeline
EXPECTED_COLUMNS = [
    'Audio_File',
    'Original_Transcription',
    'Synthesized_Transcription',
    'CER',
    'WER',
    'UTMOS_Original',
    'UTMOS_Synthesized',
    'Log_F0_RMSE',
    'MCD',
    'Speaker_Cosine_Similarity',
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate metrics from TTS-Evaluation-Pipeline CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --experiment-dir ./experiments/exp001
  %(prog)s --experiment-dir ./experiments/exp001 --output ./results.csv

Expected input format (pipe-delimited CSV):
  Audio_File|Original_Transcription|Synthesized_Transcription|CER|WER|...
        '''
    )

    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        help='Directory containing metrics/ folder with per-sample CSVs'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: experiment-dir/results_summary.csv)'
    )

    return parser.parse_args()


def read_metrics_csv(csv_path: Path) -> List[Dict[str, float]]:
    """
    Read a metrics CSV file and extract numeric values.

    Args:
        csv_path: Path to the metrics CSV file

    Returns:
        List of dictionaries with metric values per sample
    """
    samples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')

        # Validate columns
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        missing_cols = set(METRICS_TO_AGGREGATE) - set(reader.fieldnames)
        if missing_cols:
            print(f"WARNING: Missing columns in {csv_path}: {missing_cols}")

        for row in reader:
            sample_metrics = {}
            for metric in METRICS_TO_AGGREGATE:
                if metric in row:
                    try:
                        value = float(row[metric])
                        sample_metrics[metric] = value
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        pass

            if sample_metrics:
                samples.append(sample_metrics)

    return samples


def compute_statistics(samples: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and standard deviation for each metric.

    Args:
        samples: List of sample metric dictionaries

    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    stats = {}

    for metric in METRICS_TO_AGGREGATE:
        values = [s[metric] for s in samples if metric in s]

        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            stats[metric] = (mean_val, std_val)
        else:
            stats[metric] = (float('nan'), float('nan'))

    return stats


def aggregate_metrics(experiment_dir: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Aggregate metrics from all CSV files in the metrics directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary mapping condition name to metric statistics
    """
    metrics_dir = experiment_dir / 'metrics'

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    csv_files = sorted(metrics_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metrics_dir}")

    results = {}

    for csv_path in csv_files:
        condition = csv_path.stem  # Filename without .csv

        try:
            samples = read_metrics_csv(csv_path)

            if not samples:
                print(f"WARNING: No valid samples in {csv_path}")
                continue

            stats = compute_statistics(samples)
            results[condition] = stats

            print(f"Processed {condition}: {len(samples)} samples")

        except Exception as e:
            print(f"ERROR processing {csv_path}: {e}")
            continue

    return results


def write_summary_csv(results: Dict[str, Dict[str, Tuple[float, float]]], output_path: Path) -> None:
    """
    Write aggregated results to a summary CSV file.

    Args:
        results: Dictionary of condition -> metric statistics
        output_path: Path to output CSV file
    """
    # Build header
    header = ['condition']
    for metric in METRICS_TO_AGGREGATE:
        header.append(f'{metric}_mean')
        header.append(f'{metric}_std')

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for condition in sorted(results.keys()):
            stats = results[condition]
            row = [condition]

            for metric in METRICS_TO_AGGREGATE:
                mean_val, std_val = stats.get(metric, (float('nan'), float('nan')))
                row.append(f'{mean_val:.4f}')
                row.append(f'{std_val:.4f}')

            writer.writerow(row)


def print_summary(results: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """
    Print a formatted summary of the results to stdout.

    Args:
        results: Dictionary of condition -> metric statistics
    """
    print("\n" + "=" * 100)
    print("METRICS SUMMARY")
    print("=" * 100)

    for condition in sorted(results.keys()):
        stats = results[condition]
        print(f"\n{condition}:")
        print("-" * 60)

        for metric in METRICS_TO_AGGREGATE:
            mean_val, std_val = stats.get(metric, (float('nan'), float('nan')))
            print(f"  {metric:30s}: {mean_val:8.4f} +/- {std_val:8.4f}")

    print("\n" + "=" * 100)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
        return 1

    # Set default output path
    output_path = Path(args.output) if args.output else experiment_dir / 'results_summary.csv'

    print(f"Experiment directory: {experiment_dir}")
    print(f"Output file: {output_path}")
    print()

    try:
        # Aggregate metrics from all CSVs
        results = aggregate_metrics(experiment_dir)

        if not results:
            print("ERROR: No metrics could be aggregated")
            return 1

        # Write summary CSV
        write_summary_csv(results, output_path)
        print(f"\nSummary written to: {output_path}")

        # Print summary to stdout
        print_summary(results)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
