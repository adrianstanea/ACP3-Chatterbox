"""Data loading functions for academic report generation."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Set
import warnings


def load_metrics_csvs(metrics_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all metric CSVs from directory."""
    metrics_dir = Path(metrics_dir)
    metrics = {}

    csv_files = sorted(metrics_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metrics_dir}")

    for csv_file in csv_files:
        condition = csv_file.stem
        try:
            df = pd.read_csv(csv_file, sep='|')
            metrics[condition] = df
            print(f"Loaded {condition}: {len(df)} samples")
        except Exception as e:
            warnings.warn(f"Failed to load {csv_file}: {e}")

    return metrics


def load_trainer_state(trainer_state_path: Path) -> Dict:
    """Load trainer state JSON file."""
    trainer_state_path = Path(trainer_state_path)
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Trainer state file not found: {trainer_state_path}")

    with open(trainer_state_path, "r") as f:
        data = json.load(f)

    print(f"Loaded trainer state: {len(data)} keys")
    return data


def load_dataset_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load dataset metadata CSV."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path, sep='|', header=None, names=['audio_file', 'transcription', 'quality'])
    # Extract speaker ID from filename (e.g., "bas_rnd1" from "bas_rnd1_001.wav")
    df['speaker'] = df['audio_file'].str.extract(r'([a-z]+_rnd\d+)_\d+\.wav')[0]
    print(f"Loaded dataset metadata: {len(df)} entries")
    return df


def load_dataset_splits(splits_dir: Path) -> Dict:
    """Load train/val/test splits and identify holdout speakers."""
    splits_dir = Path(splits_dir)
    splits = {}

    # Load splits
    for split_file in ["train.csv", "val.csv", "test_holdout.csv"]:
        path = splits_dir / split_file
        if path.exists():
            df = pd.read_csv(path, sep='|', header=None, names=['sample_id', 'original_text', 'synthesized_text'])
            # Extract speaker ID from sample_id (e.g., "bas_rnd1" from "bas_rnd1_101")
            df['speaker'] = df['sample_id'].str.extract(r'^([a-z]+_rnd\d+)_\d+')[0]
            splits[split_file.replace(".csv", "")] = df
            print(
                f"Loaded {split_file}: {len(splits[split_file.replace('.csv', '')])}"
                " samples"
            )

    # Identify holdout speakers (speakers only in test set)
    if "test_holdout" in splits:
        test_speakers = (
            set(splits["test_holdout"]["speaker"].unique())
            if "speaker" in splits["test_holdout"].columns
            else set()
        )
        train_speakers = (
            set(splits.get("train", pd.DataFrame()).get("speaker", pd.Series()).unique())
            if "train" in splits
            else set()
        )

        holdout_speakers = test_speakers - train_speakers
        splits["test_holdout_speakers"] = holdout_speakers
        print(f"Identified {len(holdout_speakers)} holdout speakers: {holdout_speakers}")

    return splits


def load_filelists(filelists_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load filelists for each condition."""
    filelists_dir = Path(filelists_dir)
    filelists = {}

    csv_files = sorted(filelists_dir.glob("*.csv"))
    for csv_file in csv_files:
        condition = csv_file.stem
        df = pd.read_csv(csv_file, sep='|')
        filelists[condition] = df
        print(f"Loaded filelist {condition}: {len(df)} files")

    return filelists


def validate_loaded_data(
    metrics: Dict,
    trainer_state: Dict,
    metadata: pd.DataFrame,
    splits: Dict,
    filelists: Dict,
) -> bool:
    """Validate that all loaded data is complete and consistent."""
    checks = []

    # Check metrics
    if not metrics:
        checks.append("WARNING: No metrics loaded")
    else:
        checks.append(f"✓ Metrics: {len(metrics)} conditions loaded")

    # Check trainer state
    if not trainer_state:
        checks.append("WARNING: No trainer state loaded")
    elif "epoch" not in trainer_state:
        checks.append("WARNING: Trainer state missing 'epoch' key")
    else:
        checks.append(f"✓ Trainer state: epoch {trainer_state['epoch']}")

    # Check metadata
    if metadata.empty:
        checks.append("WARNING: Metadata is empty")
    else:
        checks.append(f"✓ Metadata: {len(metadata)} entries")

    # Check splits
    if not splits:
        checks.append("WARNING: No splits loaded")
    else:
        checks.append(f"✓ Splits: {len(splits)} split files")

    # Check filelists
    if not filelists:
        checks.append("WARNING: No filelists loaded")
    else:
        checks.append(f"✓ Filelists: {len(filelists)} conditions")

    for check in checks:
        print(check)

    return all(c.startswith("✓") for c in checks)
