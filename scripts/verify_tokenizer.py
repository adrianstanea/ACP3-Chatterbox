#!/usr/bin/env python3
"""
Tokenizer Verification Script for Romanian Character Coverage

This script verifies whether the Chatterbox grapheme tokenizer includes
all necessary Romanian diacritical characters (ă, â, î, ș, ț and capitals).

Usage:
    python verify_tokenizer.py <tokenizer_json_path> [metadata_csv_path]

Arguments:
    tokenizer_json_path: Path to the tokenizer.json file
    metadata_csv_path: Optional path to metadata.csv for text analysis

The script will:
1. Load the tokenizer vocabulary
2. Check for Romanian character coverage
3. Analyze text samples if metadata is provided
4. Report findings and recommendations
"""

import json
import sys
import os
import argparse
from collections import Counter
from pathlib import Path


# Romanian diacritics (both cases)
ROMANIAN_CHARS = {
    'lowercase': ['ă', 'â', 'î', 'ș', 'ț'],
    'uppercase': ['Ă', 'Â', 'Î', 'Ș', 'Ț']
}

# Common Romanian grapheme patterns
ROMANIAN_GRAPHEMES = [
    'ă', 'â', 'î', 'ș', 'ț',
    'Ă', 'Â', 'Î', 'Ș', 'Ț',
    # Potential digraphs or trigraphs used in phonetic representations
    'ea', 'oa', 'ia', 'iu', 'ie',
]


def load_tokenizer_vocab(tokenizer_path):
    """Load vocabulary from tokenizer.json file."""
    print(f"\n{'='*80}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"{'='*80}")

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found: {tokenizer_path}")
        sys.exit(1)

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary depending on structure
    vocab = {}

    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        # Standard tokenizer.json format
        vocab = tokenizer_data["model"]["vocab"]
        print(f"✓ Loaded vocabulary from model.vocab")
    elif "vocab" in tokenizer_data:
        # Alternative format
        vocab = tokenizer_data["vocab"]
        print(f"✓ Loaded vocabulary from vocab")
    else:
        print("ERROR: Could not find vocabulary in tokenizer JSON")
        print("Available keys:", list(tokenizer_data.keys()))
        sys.exit(1)

    print(f"✓ Total vocabulary size: {len(vocab)}")
    return vocab, tokenizer_data


def check_romanian_coverage(vocab):
    """Check if all Romanian characters are present in vocabulary."""
    print(f"\n{'='*80}")
    print("Romanian Character Coverage Analysis")
    print(f"{'='*80}")

    all_romanian_chars = ROMANIAN_CHARS['lowercase'] + ROMANIAN_CHARS['uppercase']

    found = []
    missing = []

    for char in all_romanian_chars:
        # Check if character exists as a token (possibly with brackets like [ă])
        found_as_is = char in vocab
        found_with_brackets = f"[{char}]" in vocab

        if found_as_is or found_with_brackets:
            found.append(char)
            token_form = f"[{char}]" if found_with_brackets else char
            print(f"✓ '{char}' found as '{token_form}' (ID: {vocab.get(token_form, vocab.get(char))})")
        else:
            missing.append(char)
            print(f"✗ '{char}' NOT FOUND in vocabulary")

    print(f"\n{'='*80}")
    print(f"Coverage Summary:")
    print(f"  Found: {len(found)}/{len(all_romanian_chars)} characters")
    print(f"  Missing: {len(missing)} characters")

    if missing:
        print(f"  Missing characters: {', '.join(missing)}")
    print(f"{'='*80}")

    return found, missing


def analyze_grapheme_patterns(vocab):
    """Check for Romanian grapheme patterns in the vocabulary."""
    print(f"\n{'='*80}")
    print("Romanian Grapheme Pattern Analysis")
    print(f"{'='*80}")

    found_patterns = []

    for pattern in ROMANIAN_GRAPHEMES:
        # Check various bracket formats
        forms_to_check = [
            pattern,
            f"[{pattern}]",
            f"({pattern})",
        ]

        for form in forms_to_check:
            if form in vocab:
                found_patterns.append((pattern, form, vocab[form]))
                print(f"✓ Pattern '{pattern}' found as '{form}' (ID: {vocab[form]})")
                break

    print(f"\n  Total patterns found: {len(found_patterns)}/{len(ROMANIAN_GRAPHEMES)}")
    print(f"{'='*80}")

    return found_patterns


def analyze_metadata_text(metadata_path, vocab):
    """Analyze text in metadata to identify character usage."""
    print(f"\n{'='*80}")
    print(f"Metadata Text Analysis")
    print(f"{'='*80}")

    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return

    print(f"Reading: {metadata_path}")

    # Collect all characters from the text
    char_counter = Counter()
    romanian_char_usage = Counter()
    total_lines = 0

    all_romanian_chars = set(ROMANIAN_CHARS['lowercase'] + ROMANIAN_CHARS['uppercase'])

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                # Assuming format: ID|RawText|NormText
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    # Use normalized text if available, else raw text
                    text = parts[2] if len(parts) > 2 else parts[1]

                    for char in text:
                        char_counter[char] += 1
                        if char in all_romanian_chars:
                            romanian_char_usage[char] += 1

        print(f"✓ Analyzed {total_lines} lines")
        print(f"\nRomanian character usage in dataset:")

        for char in sorted(romanian_char_usage.keys()):
            count = romanian_char_usage[char]
            percentage = (count / sum(char_counter.values())) * 100
            in_vocab = char in vocab or f"[{char}]" in vocab
            status = "✓" if in_vocab else "✗"
            print(f"  {status} '{char}': {count:,} occurrences ({percentage:.3f}%)")

        if not romanian_char_usage:
            print("  No Romanian diacritics found in dataset!")
            print("  WARNING: This might indicate text preprocessing issues.")

    except Exception as e:
        print(f"ERROR reading metadata: {e}")

    print(f"{'='*80}")
    return romanian_char_usage


def generate_recommendations(missing_chars, vocab_size):
    """Generate recommendations based on findings."""
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if not missing_chars:
        print("✓ All Romanian characters are covered by the tokenizer.")
        print("✓ No action needed - proceed with preprocessing.")
    else:
        print(f"⚠ WARNING: {len(missing_chars)} Romanian characters are missing from the tokenizer!")
        print(f"  Missing: {', '.join(missing_chars)}")
        print()
        print("ACTION REQUIRED: Extend the tokenizer vocabulary")
        print()
        print("Steps to extend the tokenizer:")
        print("1. Add missing characters to tokenizer.json vocabulary")
        print("2. Assign sequential IDs starting from current vocab_size")
        print(f"   Current vocab_size: {vocab_size}")
        print(f"   New vocab_size would be: {vocab_size + len(missing_chars)}")
        print()
        print("3. Update 'new_vocab_size' in the following files:")
        print("   - vendor/chatterbox-finetuning/src/config.py")
        print("   - vendor/chatterbox-finetuning/inference.py")
        print()
        print("4. Re-run preprocessing after tokenizer update")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify Chatterbox tokenizer coverage for Romanian characters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python verify_tokenizer.py vendor/chatterbox-finetuning/pretrained_models/tokenizer.json

  # With metadata analysis
  python verify_tokenizer.py tokenizer.json data/processed/MyTTSDataset/metadata.csv
        """
    )

    parser.add_argument(
        'tokenizer_path',
        help='Path to tokenizer.json file'
    )

    parser.add_argument(
        'metadata_path',
        nargs='?',
        help='Optional path to metadata.csv for text analysis'
    )

    args = parser.parse_args()

    # Load tokenizer
    vocab, tokenizer_data = load_tokenizer_vocab(args.tokenizer_path)

    # Check Romanian character coverage
    found_chars, missing_chars = check_romanian_coverage(vocab)

    # Check grapheme patterns
    found_patterns = analyze_grapheme_patterns(vocab)

    # Analyze metadata if provided
    if args.metadata_path:
        char_usage = analyze_metadata_text(args.metadata_path, vocab)

    # Generate recommendations
    generate_recommendations(missing_chars, len(vocab))

    # Exit with appropriate code
    if missing_chars:
        print("\n⚠ Tokenizer extension required!")
        sys.exit(1)
    else:
        print("\n✓ Tokenizer verification successful!")
        sys.exit(0)


if __name__ == "__main__":
    main()
