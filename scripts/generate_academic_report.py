#!/usr/bin/env python3
"""Generate comprehensive academic report for Romanian Chatterbox TTS adaptation."""

import argparse
import sys
from pathlib import Path
from report_lib.main import run_analysis_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive academic report for Romanian Chatterbox TTS adaptation"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("doc/reports"),
        help="Directory containing existing metrics summaries",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("experiments/2026-02-23/metrics"),
        help="Directory containing per-condition metric CSVs",
    )
    parser.add_argument(
        "--filelists-dir",
        type=Path,
        default=Path("experiments/2026-02-23/filelists"),
        help="Directory containing sample file lists",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/chatterbox_output/trainer_state.json"),
        help="Path to trainer_state.json",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata_SWARA1.0_text.csv"),
        help="Path to SWARA dataset metadata CSV",
    )
    parser.add_argument(
        "--dataset-splits",
        type=Path,
        default=Path("data/processed/MyTTSDataset/splits"),
        help="Directory containing train/val/test split CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("academic_report"),
        help="Output directory for all generated files",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["html", "pdf", "latex", "markdown", "json"],
        choices=["html", "pdf", "latex", "markdown", "json"],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for figure export",
    )
    parser.add_argument(
        "--style",
        choices=["academic", "presentation"],
        default="academic",
        help="Visual style for figures",
    )

    args = parser.parse_args()

    try:
        run_analysis_pipeline(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
