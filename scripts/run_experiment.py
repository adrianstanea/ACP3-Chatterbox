#!/usr/bin/env python3
"""
Experiment Runner for Romanian Chatterbox TTS Evaluation

This script generates audio samples across a parameter matrix for systematic
evaluation of the fine-tuned Romanian TTS model.

Parameter Grid:
- Speakers: BAS, SGS (holdout speakers)
- Exaggeration: [0.3, 0.7]
- CFG Weight: [0.3, 0.5]
- Samples per condition: 50 (configurable)

Output Structure:
    experiments/YYYY-MM-DD/
    ├── bas_exag0.3_cfg0.3/
    │   └── *.wav
    ├── sgs_exag0.7_cfg0.5/
    │   └── *.wav
    ├── filelists/
    │   └── *.csv (format: base_file|transcript_original)
    └── experiment_config.json

Usage:
    python scripts/run_experiment.py \
        --holdout-csv data/processed/MyTTSDataset/splits/test_holdout.csv \
        --swara-dir /data/swara/ \
        --output-dir experiments/2026-02-22

Author: Adrian Stanea / ACP3
License: Apache 2.0
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from chatterbox_ro import ChatterboxRomanian

# Default parameter grid
DEFAULT_SPEAKERS = ["bas", "sgs"]
DEFAULT_EXAGGERATIONS = [0.3, 0.7]
DEFAULT_CFG_WEIGHTS = [0.3, 0.5]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio samples across parameter matrix for Romanian TTS evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        required=True,
        help="Path to test_holdout.csv with format: id|raw_text|norm_text",
    )
    parser.add_argument(
        "--swara-dir",
        type=Path,
        required=True,
        help="Path to SWARA dataset directory for reference audio",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save experiment outputs",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/chatterbox_output"),
        help="Path to fine-tuned checkpoint directory",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/pretrained_models"),
        help="Path to pretrained models directory",
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=50,
        help="Number of samples to generate per (speaker, exag, cfg) condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference",
    )

    # Parameter grid options
    parser.add_argument(
        "--exaggerations",
        type=float,
        nargs="+",
        default=DEFAULT_EXAGGERATIONS,
        help="Exaggeration values to test",
    )
    parser.add_argument(
        "--cfg-weights",
        type=float,
        nargs="+",
        default=DEFAULT_CFG_WEIGHTS,
        help="CFG weight values to test",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=DEFAULT_SPEAKERS,
        help="Speaker prefixes to include (lowercase)",
    )

    return parser.parse_args()


def load_holdout_samples(csv_path: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Load holdout samples from CSV and group by speaker.

    Args:
        csv_path: Path to holdout CSV file

    Returns:
        Dictionary mapping speaker prefix (lowercase) to list of (id, raw_text, norm_text)
    """
    samples_by_speaker: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) != 3:
                continue

            sample_id, raw_text, norm_text = row

            # Extract speaker prefix (e.g., "bas" from "bas_rnd1_001")
            parts = sample_id.split("_")
            if len(parts) < 2:
                continue
            speaker = parts[0].lower()

            samples_by_speaker[speaker].append((sample_id, raw_text, norm_text))

    return samples_by_speaker


def get_reference_audio_path(swara_dir: Path, speaker: str) -> Path:
    """
    Get reference audio path for a speaker.

    Uses a consistent reference file: {speaker}_rnd1_011.wav

    Args:
        swara_dir: Path to SWARA dataset directory
        speaker: Speaker prefix (lowercase)

    Returns:
        Path to reference audio file
    """
    # Use consistent reference: rnd1_011 (exists for all speakers)
    ref_name = f"{speaker}_rnd1_011.wav"
    ref_path = swara_dir / ref_name

    if not ref_path.exists():
        # Fall back to searching for any file from this speaker
        pattern = f"{speaker}_*.wav"
        matches = list(swara_dir.glob(pattern))
        if matches:
            ref_path = sorted(matches)[0]
        else:
            raise FileNotFoundError(
                f"No reference audio found for speaker '{speaker}' in {swara_dir}"
            )

    return ref_path


def create_condition_directory(
    output_dir: Path, speaker: str, exag: float, cfg: float
) -> Path:
    """Create and return directory for a specific condition."""
    dir_name = f"{speaker}_exag{exag}_cfg{cfg}"
    condition_dir = output_dir / dir_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir


def save_filelist(
    output_dir: Path,
    speaker: str,
    exag: float,
    cfg: float,
    samples: List[Tuple[str, str]],
) -> Path:
    """
    Save filelist CSV for metrics pipeline.

    Args:
        output_dir: Base experiment output directory
        speaker: Speaker prefix
        exag: Exaggeration value
        cfg: CFG weight value
        samples: List of (sample_id, transcript) tuples

    Returns:
        Path to created filelist CSV
    """
    filelists_dir = output_dir / "filelists"
    filelists_dir.mkdir(parents=True, exist_ok=True)

    filelist_name = f"{speaker}_exag{exag}_cfg{cfg}.csv"
    filelist_path = filelists_dir / filelist_name

    with open(filelist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for sample_id, transcript in samples:
            # Format: base_file|transcript_original
            writer.writerow([sample_id, transcript])

    return filelist_path


def save_experiment_config(
    output_dir: Path,
    args: argparse.Namespace,
    samples_by_speaker: Dict[str, int],
    total_samples: int,
) -> Path:
    """
    Save experiment configuration as JSON.

    Args:
        output_dir: Experiment output directory
        args: Parsed command-line arguments
        samples_by_speaker: Dictionary of speaker -> sample count
        total_samples: Total number of samples generated

    Returns:
        Path to config JSON file
    """
    config = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "speakers": args.speakers,
            "exaggerations": args.exaggerations,
            "cfg_weights": args.cfg_weights,
            "samples_per_condition": args.samples_per_condition,
            "seed": args.seed,
            "device": args.device,
        },
        "paths": {
            "checkpoint": str(args.checkpoint.resolve()),
            "pretrained": str(args.pretrained.resolve()),
            "holdout_csv": str(args.holdout_csv.resolve()),
            "swara_dir": str(args.swara_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "statistics": {
            "samples_by_speaker": samples_by_speaker,
            "total_samples": total_samples,
            "conditions": len(args.speakers) * len(args.exaggerations) * len(args.cfg_weights),
        },
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_path


def main() -> int:
    """Main experiment runner."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Validate paths
    if not args.holdout_csv.exists():
        print(f"Error: Holdout CSV not found: {args.holdout_csv}")
        return 1

    if not args.swara_dir.exists():
        print(f"Error: SWARA directory not found: {args.swara_dir}")
        return 1

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint directory not found: {args.checkpoint}")
        return 1

    if not args.pretrained.exists():
        print(f"Error: Pretrained directory not found: {args.pretrained}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Romanian Chatterbox Experiment Runner")
    print("=" * 60)

    # Load holdout samples
    print(f"\nLoading holdout samples from: {args.holdout_csv}")
    all_samples = load_holdout_samples(args.holdout_csv)

    available_speakers = [s for s in args.speakers if s in all_samples]
    if not available_speakers:
        print(f"Error: No samples found for speakers: {args.speakers}")
        print(f"Available speakers: {list(all_samples.keys())}")
        return 1

    print(f"Found speakers: {available_speakers}")
    for speaker in available_speakers:
        print(f"  - {speaker.upper()}: {len(all_samples[speaker])} samples")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    print(f"Pretrained models: {args.pretrained}")

    model = ChatterboxRomanian.from_local(
        checkpoint_dir=args.checkpoint,
        pretrained_dir=args.pretrained,
        device=args.device,
    )
    print(f"Model loaded: {model}")

    # Calculate total conditions
    n_conditions = len(available_speakers) * len(args.exaggerations) * len(args.cfg_weights)
    n_samples_per_speaker = min(args.samples_per_condition, min(len(all_samples[s]) for s in available_speakers))
    total_samples = n_conditions * n_samples_per_speaker

    print(f"\nExperiment Grid:")
    print(f"  Speakers: {available_speakers}")
    print(f"  Exaggerations: {args.exaggerations}")
    print(f"  CFG weights: {args.cfg_weights}")
    print(f"  Samples per condition: {n_samples_per_speaker}")
    print(f"  Total conditions: {n_conditions}")
    print(f"  Total samples: {total_samples}")

    # Prepare sample selection for each speaker
    selected_samples: Dict[str, List[Tuple[str, str, str]]] = {}
    for speaker in available_speakers:
        speaker_samples = all_samples[speaker]
        if len(speaker_samples) > n_samples_per_speaker:
            selected = random.sample(speaker_samples, n_samples_per_speaker)
        else:
            selected = speaker_samples
        selected_samples[speaker] = selected
        print(f"  Selected {len(selected)} samples for {speaker.upper()}")

    # Get reference audio paths
    reference_paths: Dict[str, Path] = {}
    for speaker in available_speakers:
        ref_path = get_reference_audio_path(args.swara_dir, speaker)
        reference_paths[speaker] = ref_path
        print(f"  Reference for {speaker.upper()}: {ref_path.name}")

    print(f"\nOutput directory: {args.output_dir}")
    print("\nStarting generation...")

    # Track statistics
    stats_by_speaker: Dict[str, int] = defaultdict(int)
    total_generated = 0

    # Generate samples across all conditions
    with tqdm(total=total_samples, desc="Generating samples", unit="sample") as pbar:
        for speaker in available_speakers:
            ref_audio = reference_paths[speaker]
            samples = selected_samples[speaker]

            for exag in args.exaggerations:
                for cfg in args.cfg_weights:
                    # Create condition directory
                    condition_dir = create_condition_directory(
                        args.output_dir, speaker, exag, cfg
                    )

                    # Track filelist entries
                    filelist_entries: List[Tuple[str, str]] = []

                    for sample_id, raw_text, norm_text in samples:
                        # Generate audio
                        try:
                            wav = model.generate(
                                text=raw_text,
                                reference_audio=ref_audio,
                                cfg_weight=cfg,
                                exaggeration=exag,
                            )

                            # Save audio
                            output_path = condition_dir / f"{sample_id}.wav"

                            # Ensure wav is on CPU and correct shape
                            if wav.dim() == 1:
                                wav = wav.unsqueeze(0)
                            wav_cpu = wav.cpu()

                            torchaudio.save(
                                str(output_path),
                                wav_cpu,
                                sample_rate=model.sr,
                            )

                            # Add to filelist
                            filelist_entries.append((sample_id, raw_text))

                            stats_by_speaker[speaker] += 1
                            total_generated += 1

                        except Exception as e:
                            tqdm.write(f"Error generating {sample_id}: {e}")

                        pbar.update(1)

                    # Save filelist for this condition
                    save_filelist(
                        args.output_dir, speaker, exag, cfg, filelist_entries
                    )

    # Save experiment config
    config_path = save_experiment_config(
        args.output_dir, args, dict(stats_by_speaker), total_generated
    )

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"\nGenerated {total_generated} samples across {n_conditions} conditions")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")

    for speaker in available_speakers:
        for exag in args.exaggerations:
            for cfg in args.cfg_weights:
                dir_name = f"{speaker}_exag{exag}_cfg{cfg}"
                n_files = len(list((args.output_dir / dir_name).glob("*.wav")))
                print(f"    {dir_name}/ ({n_files} files)")

    print(f"    filelists/ ({n_conditions} CSVs)")
    print(f"    experiment_config.json")
    print(f"\nConfig saved to: {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
