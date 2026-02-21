#!/usr/bin/env python3
"""
SWARA Dataset Analysis Script
Analyzes the SWARA 1.0 Romanian speech dataset for TTS model training.
"""

import argparse
import csv
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


# Romanian diacritics to verify
ROMANIAN_DIACRITICS = {'ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț'}


def extract_speaker_id(filename: str) -> str:
    """Extract speaker ID from filename (prefix before underscore)."""
    basename = os.path.basename(filename)
    # Speaker ID is the prefix before first underscore
    speaker_id = basename.split('_')[0]
    return speaker_id


def map_metadata_path_to_actual(metadata_path: str, actual_base_dir: str) -> str:
    """
    Map metadata path to actual file location.

    Metadata paths: /media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/bas_rnd1_001.wav
    Actual location: /home/astanea/data/SWARA1.0_22k_noSil/bas_rnd1_001.wav
    """
    filename = os.path.basename(metadata_path)
    actual_path = os.path.join(actual_base_dir, filename)
    return actual_path


def analyze_audio_file(audio_path: str) -> Tuple[float, int]:
    """
    Analyze a single audio file.

    Returns:
        (duration_seconds, sample_rate)
    """
    try:
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        return duration, sample_rate
    except Exception as e:
        raise RuntimeError(f"Error analyzing {audio_path}: {e}")


def analyze_dataset(metadata_path: str, audio_base_dir: str) -> None:
    """
    Main analysis function.

    Args:
        metadata_path: Path to metadata CSV file
        audio_base_dir: Base directory containing audio files
    """
    print("="*80)
    print("SWARA 1.0 DATASET ANALYSIS")
    print("="*80)
    print(f"\nMetadata file: {metadata_path}")
    print(f"Audio base directory: {audio_base_dir}")
    print()

    # Storage for analysis
    durations = []
    sample_rates = []
    speaker_stats = defaultdict(lambda: {'count': 0, 'duration': 0.0})
    missing_files = []
    all_text = []
    char_counter = Counter()

    # Read metadata
    total_entries = 0
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')

        for row in reader:
            total_entries += 1
            if len(row) < 2:
                print(f"WARNING: Malformed row {total_entries}: {row}")
                continue

            metadata_audio_path = row[0]
            text = row[1]

            # Map to actual file location
            actual_audio_path = map_metadata_path_to_actual(
                metadata_audio_path,
                audio_base_dir
            )

            # Extract speaker ID
            speaker_id = extract_speaker_id(metadata_audio_path)

            # Check if file exists
            if not os.path.exists(actual_audio_path):
                missing_files.append(actual_audio_path)
                continue

            # Analyze audio
            try:
                duration, sample_rate = analyze_audio_file(actual_audio_path)

                durations.append(duration)
                sample_rates.append(sample_rate)

                speaker_stats[speaker_id]['count'] += 1
                speaker_stats[speaker_id]['duration'] += duration

                all_text.append(text)
                char_counter.update(text)

            except RuntimeError as e:
                print(f"ERROR: {e}")
                missing_files.append(actual_audio_path)

    # Convert to numpy arrays for analysis
    durations = np.array(durations)
    sample_rates = np.array(sample_rates)

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total entries in metadata: {total_entries}")
    print(f"Successfully analyzed files: {len(durations)}")
    print(f"Missing/inaccessible files: {len(missing_files)}")

    if missing_files:
        print(f"\nFirst 10 missing files:")
        for f in missing_files[:10]:
            print(f"  - {f}")

    print("\n" + "="*80)
    print("DURATION ANALYSIS")
    print("="*80)
    print(f"Total duration: {durations.sum()/3600:.2f} hours")
    print(f"Mean duration: {durations.mean():.2f} seconds")
    print(f"Median duration: {np.median(durations):.2f} seconds")
    print(f"Min duration: {durations.min():.2f} seconds")
    print(f"Max duration: {durations.max():.2f} seconds")
    print(f"Std deviation: {durations.std():.2f} seconds")

    # Outliers
    very_short = durations[durations < 0.5]
    very_long = durations[durations > 30.0]

    print(f"\nDuration outliers:")
    print(f"  Clips < 0.5s: {len(very_short)} ({len(very_short)/len(durations)*100:.2f}%)")
    if len(very_short) > 0:
        print(f"    Min: {very_short.min():.2f}s, Max: {very_short.max():.2f}s")

    print(f"  Clips > 30s: {len(very_long)} ({len(very_long)/len(durations)*100:.2f}%)")
    if len(very_long) > 0:
        print(f"    Min: {very_long.min():.2f}s, Max: {very_long.max():.2f}s")

    # Duration distribution
    print(f"\nDuration distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(durations, p):.2f}s")

    print("\n" + "="*80)
    print("SAMPLE RATE ANALYSIS")
    print("="*80)
    unique_rates = np.unique(sample_rates)
    print(f"Unique sample rates: {unique_rates}")

    for rate in unique_rates:
        count = np.sum(sample_rates == rate)
        print(f"  {rate} Hz: {count} files ({count/len(sample_rates)*100:.2f}%)")

    print("\n" + "="*80)
    print("SPEAKER STATISTICS")
    print("="*80)
    print(f"Total speakers: {len(speaker_stats)}")

    # Speaker distribution
    speaker_counts = [stats['count'] for stats in speaker_stats.values()]
    speaker_durations = [stats['duration'] for stats in speaker_stats.values()]

    print(f"\nUtterances per speaker:")
    print(f"  Mean: {np.mean(speaker_counts):.1f}")
    print(f"  Median: {np.median(speaker_counts):.1f}")
    print(f"  Min: {min(speaker_counts)}")
    print(f"  Max: {max(speaker_counts)}")

    print(f"\nDuration per speaker:")
    print(f"  Mean: {np.mean(speaker_durations)/60:.2f} minutes")
    print(f"  Median: {np.median(speaker_durations)/60:.2f} minutes")
    print(f"  Min: {min(speaker_durations)/60:.2f} minutes")
    print(f"  Max: {max(speaker_durations)/60:.2f} minutes")

    # Speakers with few utterances
    few_utterances = {sid: stats for sid, stats in speaker_stats.items()
                     if stats['count'] < 50}

    if few_utterances:
        print(f"\nSpeakers with < 50 utterances: {len(few_utterances)}")
        print("Top 10 speakers with fewest utterances:")
        sorted_speakers = sorted(few_utterances.items(),
                                key=lambda x: x[1]['count'])
        for sid, stats in sorted_speakers[:10]:
            print(f"  {sid}: {stats['count']} utterances, "
                  f"{stats['duration']/60:.2f} minutes")

    # Top speakers
    print(f"\nTop 10 speakers by utterance count:")
    sorted_speakers = sorted(speaker_stats.items(),
                            key=lambda x: x[1]['count'],
                            reverse=True)
    for sid, stats in sorted_speakers[:10]:
        print(f"  {sid}: {stats['count']} utterances, "
              f"{stats['duration']/60:.2f} minutes")

    print("\n" + "="*80)
    print("TEXT ANALYSIS")
    print("="*80)

    # Romanian diacritics
    print("Romanian diacritics presence:")
    found_diacritics = set()
    for char in char_counter.keys():
        if char in ROMANIAN_DIACRITICS:
            found_diacritics.add(char)

    for diacritic in sorted(ROMANIAN_DIACRITICS):
        count = char_counter.get(diacritic, 0)
        if count > 0:
            status = "✓"
            found_diacritics.add(diacritic)
        else:
            status = "✗"
        print(f"  {status} '{diacritic}': {count} occurrences")

    missing_diacritics = ROMANIAN_DIACRITICS - found_diacritics
    if missing_diacritics:
        print(f"\nWARNING: Missing diacritics: {missing_diacritics}")
    else:
        print(f"\n✓ All Romanian diacritics present in dataset!")

    # Character inventory
    print(f"\nTotal unique characters: {len(char_counter)}")
    print(f"Total character count: {sum(char_counter.values())}")

    # Most common characters
    print(f"\nTop 30 most common characters:")
    for char, count in char_counter.most_common(30):
        if char == ' ':
            char_display = '<space>'
        elif char == '\n':
            char_display = '<newline>'
        elif char == '\t':
            char_display = '<tab>'
        else:
            char_display = char
        print(f"  '{char_display}': {count}")

    # Special characters and digits
    digits = {char: count for char, count in char_counter.items()
             if char.isdigit()}
    if digits:
        print(f"\nDigits found: {len(digits)}")
        for char, count in sorted(digits.items()):
            print(f"  '{char}': {count}")

    # Punctuation
    punctuation = {char: count for char, count in char_counter.items()
                  if not char.isalnum() and not char.isspace()}
    print(f"\nPunctuation marks: {len(punctuation)}")
    for char, count in sorted(punctuation.items(),
                             key=lambda x: x[1],
                             reverse=True)[:20]:
        print(f"  '{char}': {count}")

    # Full character inventory (for tokenizer)
    print("\n" + "="*80)
    print("FULL CHARACTER INVENTORY (for tokenizer)")
    print("="*80)
    all_chars = sorted(char_counter.keys())
    print(f"Characters ({len(all_chars)}):")

    # Group by type
    letters = [c for c in all_chars if c.isalpha()]
    digits = [c for c in all_chars if c.isdigit()]
    spaces = [c for c in all_chars if c.isspace()]
    punctuation = [c for c in all_chars
                  if not c.isalnum() and not c.isspace()]

    print(f"\nLetters ({len(letters)}):")
    print(''.join(letters))

    print(f"\nDigits ({len(digits)}):")
    print(''.join(digits))

    print(f"\nPunctuation ({len(punctuation)}):")
    print(''.join(punctuation))

    print(f"\nWhitespace ({len(spaces)}):")
    for char in spaces:
        if char == ' ':
            print("  <space>")
        elif char == '\n':
            print("  <newline>")
        elif char == '\t':
            print("  <tab>")
        else:
            print(f"  <U+{ord(char):04X}>")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Dataset contains {len(durations)} valid audio files")
    print(f"✓ Total duration: {durations.sum()/3600:.2f} hours")
    print(f"✓ Sample rate: {unique_rates[0]} Hz (consistent)"
          if len(unique_rates) == 1
          else f"⚠ Multiple sample rates: {unique_rates}")
    print(f"✓ {len(speaker_stats)} speakers")
    print(f"✓ All Romanian diacritics present"
          if not missing_diacritics
          else f"⚠ Missing diacritics: {missing_diacritics}")

    if len(very_short) > 0:
        print(f"⚠ {len(very_short)} clips < 0.5s (consider filtering)")
    if len(very_long) > 0:
        print(f"⚠ {len(very_long)} clips > 30s (consider filtering)")
    if len(missing_files) > 0:
        print(f"⚠ {len(missing_files)} files missing/inaccessible")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SWARA dataset for TTS training'
    )
    parser.add_argument(
        'metadata',
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--audio-dir',
        default='/home/astanea/data/SWARA1.0_22k_noSil',
        help='Base directory containing audio files '
             '(default: /home/astanea/data/SWARA1.0_22k_noSil)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.metadata):
        print(f"ERROR: Metadata file not found: {args.metadata}")
        return 1

    if not os.path.isdir(args.audio_dir):
        print(f"ERROR: Audio directory not found: {args.audio_dir}")
        return 1

    # Run analysis
    try:
        analyze_dataset(args.metadata, args.audio_dir)
        return 0
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
