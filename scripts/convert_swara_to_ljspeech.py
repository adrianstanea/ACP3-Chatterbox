#!/usr/bin/env python3
"""
Convert SWARA dataset to LJSpeech format for Chatterbox TTS fine-tuning.

This script:
1. Reads SWARA metadata and normalizes Romanian text
2. Creates LJSpeech directory structure with symlinked audio
3. Performs speaker-stratified train/val split (90/10)
4. Holds out BAS and SGS speakers for zero-shot testing

Output structure:
    data/processed/MyTTSDataset/
        metadata.csv          # filename|raw_text|normalized_text
        wavs/                 # symlinks to original audio files
        splits/
            train.csv         # 15 speakers, 90% each
            val.csv           # 15 speakers, 10% each
            test_holdout.csv  # BAS and SGS only
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SWARA_PATH = os.getenv("SWARA_PATH", "/home/astanea/data/SWARA1.0_22k_noSil")
METADATA_FILE = "metadata_SWARA1.0_text.csv"
OUTPUT_DIR = Path("data/processed/MyTTSDataset")
HOLDOUT_SPEAKERS = {"BAS", "SGS"}
VAL_SAMPLES_PER_SPEAKER = 5  # Small validation set for listening during training
RANDOM_SEED = 42


def punc_norm(text: str) -> str:
    """
    Text normalization using Chatterbox's standard normalization.

    This function ensures consistency with the preprocessing pipeline
    by applying the same normalization rules used during training.

    Based on Chatterbox's punc_norm function from src/chatterbox_/tts.py

    Args:
        text: Raw text input

    Returns:
        Normalized text with cleaned punctuation
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # # Capitalize first letter
    # if text[0].islower():
    #     text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punctuation
    # punc_to_replace = [
    #     ("...", ", "),
    #     ("…", ", "),
    #     (":", ","),
    #     (" - ", ", "),
    #     (";", ", "),
    #     ("—", "-"),
    #     ("–", "-"),
    #     (" ,", ","),
    #     (""", "\""),
    #     (""", "\""),
    #     ("'", "'"),
    #     ("'", "'"),
    # ]
    # for old_char_sequence, new_char in punc_to_replace:
    #     text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    # Convert to lowercase (Chatterbox is case-insensitive)
    text = text.lower()

    return text


def extract_speaker_prefix(filename: str) -> str:
    """
    Extract speaker prefix from SWARA filename.

    Example: bas_rnd1_001.wav -> BAS

    Args:
        filename: SWARA audio filename

    Returns:
        Speaker prefix in uppercase
    """
    return filename.split("_")[0].upper()


def map_audio_path(original_path: str, swara_base_path: str) -> Path:
    """
    Map original metadata path to actual SWARA audio file location.

    The metadata contains paths like:
        /media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/bas_rnd1_001.wav

    We need to map to:
        $SWARA_PATH/bas_rnd1_001.wav

    Args:
        original_path: Path from metadata CSV
        swara_base_path: Base directory containing SWARA audio files

    Returns:
        Actual path to audio file
    """
    filename = Path(original_path).name
    return Path(swara_base_path) / filename


def load_swara_metadata(
    metadata_path: str, swara_base_path: str
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Load SWARA metadata and organize by speaker.

    Args:
        metadata_path: Path to metadata_SWARA1.0_text.csv
        swara_base_path: Base directory containing SWARA audio files

    Returns:
        Tuple of (all_samples, samples_by_speaker)
    """
    all_samples = []
    samples_by_speaker = defaultdict(list)

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) != 3:
                print(f"Warning: Skipping malformed row: {row}")
                continue

            original_path, text, quality = row

            # Map to actual audio file location
            audio_path = map_audio_path(original_path, swara_base_path)

            # Verify file exists
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue

            # Extract speaker prefix
            filename = audio_path.name
            speaker = extract_speaker_prefix(filename)

            # Normalize text using Chatterbox's punc_norm
            normalized = punc_norm(text)

            sample = {
                "filename": filename.replace(".wav", ""),  # Remove extension for LJSpeech format
                "audio_path": audio_path,
                "raw_text": text,
                "normalized_text": normalized,
                "speaker": speaker,
                "quality": quality,
            }

            all_samples.append(sample)
            samples_by_speaker[speaker].append(sample)

    return all_samples, samples_by_speaker


def create_ljspeech_structure(output_dir: Path) -> Dict[str, Path]:
    """
    Create LJSpeech directory structure.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of important paths
    """
    paths = {
        "base": output_dir,
        "wavs": output_dir / "wavs",
        "splits": output_dir / "splits",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def create_symlinks(samples: List[Dict], wavs_dir: Path) -> None:
    """
    Create symbolic links from wavs/ to original audio files.

    Args:
        samples: List of sample dictionaries
        wavs_dir: Directory to create symlinks in
    """
    for sample in samples:
        link_path = wavs_dir / f"{sample['filename']}.wav"
        target_path = sample["audio_path"].resolve()

        # Create symlink if it doesn't exist
        if not link_path.exists():
            link_path.symlink_to(target_path)


def split_samples(
    samples_by_speaker: Dict[str, List[Dict]],
    holdout_speakers: set,
    val_samples_per_speaker: int,
    random_seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform speaker-stratified train/val split with holdout speakers.

    Validation set is intentionally small (5 samples per speaker) because
    it's primarily used for listening during training, not for metrics.

    Args:
        samples_by_speaker: Samples organized by speaker
        holdout_speakers: Speakers to hold out for zero-shot testing
        val_samples_per_speaker: Number of validation samples per speaker
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, holdout_samples)
    """
    random.seed(random_seed)

    train_samples = []
    val_samples = []
    holdout_samples = []

    for speaker, samples in samples_by_speaker.items():
        # Shuffle samples for this speaker
        shuffled = samples.copy()
        random.shuffle(shuffled)

        if speaker in holdout_speakers:
            # All samples go to holdout
            holdout_samples.extend(shuffled)
            print(f"Speaker {speaker}: {len(shuffled)} samples -> holdout")
        else:
            # Take fixed number for validation, rest for training
            val_count = min(val_samples_per_speaker, len(shuffled))
            val_speaker = shuffled[:val_count]
            train_speaker = shuffled[val_count:]

            train_samples.extend(train_speaker)
            val_samples.extend(val_speaker)

            print(f"Speaker {speaker}: {len(train_speaker)} train, {len(val_speaker)} val")

    return train_samples, val_samples, holdout_samples


def write_metadata_csv(samples: List[Dict], output_path: Path) -> None:
    """
    Write samples to LJSpeech format metadata CSV.

    Format: filename|raw_text|normalized_text

    Args:
        samples: List of sample dictionaries
        output_path: Path to output CSV file
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for sample in samples:
            writer.writerow([
                sample["filename"],
                sample["raw_text"],
                sample["normalized_text"],
            ])


def print_statistics(
    train_samples: List[Dict],
    val_samples: List[Dict],
    holdout_samples: List[Dict]
) -> None:
    """
    Print dataset statistics.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        holdout_samples: Holdout samples
    """
    total = len(train_samples) + len(val_samples) + len(holdout_samples)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples:    {total:6d}")
    print(f"Training:         {len(train_samples):6d} ({len(train_samples)/total*100:.1f}%)")
    print(f"Validation:       {len(val_samples):6d} ({len(val_samples)/total*100:.1f}%)")
    print(f"Holdout (test):   {len(holdout_samples):6d} ({len(holdout_samples)/total*100:.1f}%)")
    print("=" * 60)

    # Get unique speakers
    train_speakers = set(s["speaker"] for s in train_samples)
    val_speakers = set(s["speaker"] for s in val_samples)
    holdout_speakers = set(s["speaker"] for s in holdout_samples)

    print(f"\nTrain speakers:   {sorted(train_speakers)}")
    print(f"Val speakers:     {sorted(val_speakers)}")
    print(f"Holdout speakers: {sorted(holdout_speakers)}")
    print()


def validate_dataset(paths: Dict[str, Path]) -> None:
    """
    Perform validation checks on the converted dataset.

    Args:
        paths: Dictionary of dataset paths
    """
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    # Check metadata.csv format
    metadata_path = paths["base"] / "metadata.csv"
    with open(metadata_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        parts = first_line.split("|")
        print(f"✓ metadata.csv format: {len(parts)} fields")
        print(f"  Example: {first_line[:80]}...")

    # Check symlinks
    wavs_dir = paths["wavs"]
    symlink_count = sum(1 for p in wavs_dir.iterdir() if p.is_symlink())
    broken_links = sum(1 for p in wavs_dir.iterdir() if p.is_symlink() and not p.exists())
    print(f"✓ Symlinks: {symlink_count} created, {broken_links} broken")

    # Check split files
    splits_dir = paths["splits"]
    for split_file in ["train.csv", "val.csv", "test_holdout.csv"]:
        split_path = splits_dir / split_file
        with open(split_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"✓ {split_file}: {len(lines)} samples")

    # Check diacritics preservation
    print("\n✓ Checking diacritics preservation...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        sample_row = next(reader)
        diacritics = "ăâîșțĂÂÎȘȚ"
        has_diacritics = any(c in sample_row[2] for c in diacritics)
        if has_diacritics:
            print(f"  Example with diacritics: {sample_row[2][:60]}...")
        else:
            print(f"  Warning: First sample has no diacritics")

    print("=" * 60)


def main():
    """Main conversion pipeline."""
    print("=" * 60)
    print("SWARA to LJSpeech Conversion")
    print("=" * 60)
    print(f"SWARA path:            {SWARA_PATH}")
    print(f"Metadata file:         {METADATA_FILE}")
    print(f"Output directory:      {OUTPUT_DIR}")
    print(f"Holdout speakers:      {HOLDOUT_SPEAKERS}")
    print(f"Val samples/speaker:   {VAL_SAMPLES_PER_SPEAKER}")
    print("=" * 60)

    # Check if SWARA directory exists
    if not Path(SWARA_PATH).exists():
        print(f"\nError: SWARA directory not found: {SWARA_PATH}")
        print("Please set SWARA_PATH environment variable or update .env file")
        return

    # Check if metadata file exists
    if not Path(METADATA_FILE).exists():
        print(f"\nError: Metadata file not found: {METADATA_FILE}")
        return

    # Load metadata
    print("\n1. Loading SWARA metadata...")
    all_samples, samples_by_speaker = load_swara_metadata(METADATA_FILE, SWARA_PATH)
    print(f"   Loaded {len(all_samples)} samples from {len(samples_by_speaker)} speakers")

    # Create output structure
    print("\n2. Creating LJSpeech directory structure...")
    paths = create_ljspeech_structure(OUTPUT_DIR)
    print(f"   Created: {OUTPUT_DIR}")

    # Create symlinks
    print("\n3. Creating symbolic links to audio files...")
    create_symlinks(all_samples, paths["wavs"])
    print(f"   Created {len(all_samples)} symlinks in {paths['wavs']}")

    # Split dataset
    print("\n4. Splitting dataset (speaker-stratified)...")
    train_samples, val_samples, holdout_samples = split_samples(
        samples_by_speaker,
        HOLDOUT_SPEAKERS,
        VAL_SAMPLES_PER_SPEAKER,
        RANDOM_SEED
    )

    # Write metadata files
    print("\n5. Writing metadata files...")
    write_metadata_csv(all_samples, paths["base"] / "metadata.csv")
    write_metadata_csv(train_samples, paths["splits"] / "train.csv")
    write_metadata_csv(val_samples, paths["splits"] / "val.csv")
    write_metadata_csv(holdout_samples, paths["splits"] / "test_holdout.csv")
    print(f"   Written: metadata.csv and split files")

    # Print statistics
    print_statistics(train_samples, val_samples, holdout_samples)

    # Validate dataset
    validate_dataset(paths)

    print("\n✓ Conversion complete!")
    print(f"\nDataset ready at: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
