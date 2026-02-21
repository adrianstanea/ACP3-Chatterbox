#!/usr/bin/env python3
"""
Generic Tokenizer Vocabulary Extension Tool for Chatterbox TTS

This script extends the Chatterbox tokenizer vocabulary with language-specific
characters, enabling support for new languages not covered in the base tokenizer.

Features:
  - Language-agnostic: Works with any character set
  - Safe: Creates backups before modifications
  - Validated: Verifies extension success
  - Idempotent: Can run multiple times safely
  - Well-documented: Clear logging and error messages

Usage:
    # Extend with Romanian characters
    python extend_tokenizer.py --language romanian

    # Dry run (preview changes without applying)
    python extend_tokenizer.py --language romanian --dry-run

    # Custom paths
    python extend_tokenizer.py \
        --language romanian \
        --tokenizer vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
        --config vendor/chatterbox-finetuning/src/config.py

Author: Adrian Stanea
License: Apache 2.0
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime


class TokenizerExtender:
    """Extends Chatterbox tokenizer vocabulary with language-specific characters."""

    def __init__(self, tokenizer_path: Path, language_config_path: Path, dry_run: bool = False):
        """
        Initialize the tokenizer extender.

        Args:
            tokenizer_path: Path to tokenizer.json file
            language_config_path: Path to language extension config (JSON)
            dry_run: If True, preview changes without applying them
        """
        self.tokenizer_path = Path(tokenizer_path)
        self.language_config_path = Path(language_config_path)
        self.dry_run = dry_run

        # Validate paths
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        if not self.language_config_path.exists():
            raise FileNotFoundError(f"Language config not found: {self.language_config_path}")

        # Load configurations
        self.tokenizer_data = self._load_json(self.tokenizer_path)
        self.language_config = self._load_json(self.language_config_path)

        # Extract vocabulary
        if "model" in self.tokenizer_data and "vocab" in self.tokenizer_data["model"]:
            self.vocab = self.tokenizer_data["model"]["vocab"]
        else:
            raise ValueError("Invalid tokenizer format: missing model.vocab")

        self.original_vocab_size = len(self.vocab)

    def _load_json(self, path: Path) -> dict:
        """Load and parse JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, path: Path, data: dict):
        """Save data to JSON file with pretty formatting."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _create_backup(self, path: Path) -> Path:
        """Create timestamped backup of file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.parent / f"{path.stem}.backup_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_path)
        return backup_path

    def analyze_coverage(self) -> Tuple[List[str], List[str]]:
        """
        Analyze which characters are already present vs. missing.

        Returns:
            Tuple of (found_chars, missing_chars)
        """
        required_chars = [item["char"] for item in self.language_config["characters"]]

        found = []
        missing = []

        for char in required_chars:
            if char in self.vocab:
                found.append(char)
            else:
                missing.append(char)

        return found, missing

    def extend_vocabulary(self) -> Dict[str, int]:
        """
        Extend tokenizer vocabulary with missing characters.

        Returns:
            Dictionary mapping new characters to their assigned token IDs
        """
        found, missing = self.analyze_coverage()

        if not missing:
            print(f"✓ All {len(found)} characters already present in vocabulary")
            return {}

        # Assign sequential IDs starting from current max
        next_id = self.original_vocab_size
        new_tokens = {}

        for char in missing:
            new_tokens[char] = next_id
            next_id += 1

        return new_tokens

    def apply_extension(self, new_tokens: Dict[str, int]) -> int:
        """
        Apply vocabulary extension to tokenizer.

        Args:
            new_tokens: Dictionary mapping characters to token IDs

        Returns:
            New vocabulary size
        """
        if not new_tokens:
            return self.original_vocab_size

        # Update vocabulary
        for char, token_id in new_tokens.items():
            self.vocab[char] = token_id

        new_vocab_size = len(self.vocab)

        # Save extended tokenizer
        if not self.dry_run:
            backup_path = self._create_backup(self.tokenizer_path)
            print(f"✓ Backup created: {backup_path}")

            self._save_json(self.tokenizer_path, self.tokenizer_data)
            print(f"✓ Tokenizer updated: {self.tokenizer_path}")
        else:
            print(f"[DRY RUN] Would update tokenizer: {self.tokenizer_path}")

        return new_vocab_size

    def update_config_files(self, new_vocab_size: int, config_files: List[Path]):
        """
        Update new_vocab_size in config files.

        Args:
            new_vocab_size: The new vocabulary size
            config_files: List of config files to update
        """
        for config_path in config_files:
            if not config_path.exists():
                print(f"⚠ Config file not found: {config_path}")
                continue

            # Read file
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find and replace vocab size
            # Pattern: new_vocab_size: int = 52260 if is_turbo else 2454
            import re
            pattern = r'(new_vocab_size:\s*int\s*=\s*\d+\s*if\s*is_turbo\s*else\s*)(\d+)'

            match = re.search(pattern, content)
            if match:
                old_size = int(match.group(2))
                new_content = re.sub(pattern, rf'\g<1>{new_vocab_size}', content)

                if not self.dry_run:
                    backup_path = self._create_backup(config_path)
                    print(f"✓ Backup created: {backup_path}")

                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    print(f"✓ Updated {config_path.name}: {old_size} → {new_vocab_size}")
                else:
                    print(f"[DRY RUN] Would update {config_path.name}: {old_size} → {new_vocab_size}")
            else:
                print(f"⚠ Could not find vocab_size pattern in {config_path}")

    def print_report(self, new_tokens: Dict[str, int], new_vocab_size: int):
        """Print detailed extension report."""
        lang_info = self.language_config

        print("\n" + "="*80)
        print(f"Tokenizer Extension Report: {lang_info['language']}")
        print("="*80)

        print(f"\n📁 Files:")
        print(f"  Tokenizer: {self.tokenizer_path}")
        print(f"  Language:  {self.language_config_path}")

        print(f"\n📊 Vocabulary:")
        print(f"  Original size: {self.original_vocab_size}")
        print(f"  New size:      {new_vocab_size}")
        print(f"  Added tokens:  {len(new_tokens)}")

        if new_tokens:
            print(f"\n➕ New Characters:")
            for char, token_id in sorted(new_tokens.items(), key=lambda x: x[1]):
                # Find character info
                char_info = next((c for c in lang_info["characters"] if c["char"] == char), {})
                name = char_info.get("name", "Unknown")
                unicode_code = char_info.get("unicode", "")
                print(f"  '{char}' (ID: {token_id}) - {name} {unicode_code}")

        found, missing = self.analyze_coverage()
        if not new_tokens:
            print(f"\n✓ All {len(found)} characters already covered:")
            for char in found:
                print(f"  '{char}' (ID: {self.vocab[char]})")

        print(f"\n📝 Metadata:")
        if "metadata" in lang_info:
            meta = lang_info["metadata"]
            if "contributor" in meta:
                print(f"  Contributor: {meta['contributor']}")
            if "dataset" in meta:
                print(f"  Dataset:     {meta['dataset']}")
            if "date" in meta:
                print(f"  Date:        {meta['date']}")

        if "validation" in lang_info and "test_phrases" in lang_info["validation"]:
            print(f"\n🧪 Test Phrases:")
            for phrase in lang_info["validation"]["test_phrases"]:
                print(f"  - {phrase}")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extend Chatterbox tokenizer vocabulary for new languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extend with Romanian characters
  python extend_tokenizer.py --language romanian

  # Dry run (preview changes)
  python extend_tokenizer.py --language romanian --dry-run

  # Custom tokenizer path
  python extend_tokenizer.py --language romanian \\
    --tokenizer vendor/chatterbox-finetuning/pretrained_models/tokenizer.json

  # Skip config file updates
  python extend_tokenizer.py --language romanian --no-update-config

For more information, see docs/ADDING-LANGUAGES.md
        """
    )

    parser.add_argument(
        '--language',
        required=True,
        help='Language name (e.g., "romanian", "vietnamese")'
    )

    parser.add_argument(
        '--tokenizer',
        type=Path,
        default='vendor/chatterbox-finetuning/pretrained_models/tokenizer.json',
        help='Path to tokenizer.json (default: vendor/chatterbox-finetuning/pretrained_models/tokenizer.json)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.py (default: auto-detect)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--no-update-config',
        action='store_true',
        help='Skip updating config files'
    )

    args = parser.parse_args()

    # Construct language config path
    language_config = Path(f"scripts/vocab_extensions/{args.language}.json")

    if not language_config.exists():
        print(f"Error: Language config not found: {language_config}")
        print(f"\nAvailable languages:")
        extensions_dir = Path("scripts/vocab_extensions")
        if extensions_dir.exists():
            for config_file in sorted(extensions_dir.glob("*.json")):
                print(f"  - {config_file.stem}")
        else:
            print("  (none found)")
        return 1

    print(f"\n🌍 Extending Chatterbox tokenizer for: {args.language.title()}")
    print(f"{'='*80}\n")

    try:
        # Initialize extender
        extender = TokenizerExtender(args.tokenizer, language_config, args.dry_run)

        # Analyze current coverage
        found, missing = extender.analyze_coverage()

        print(f"Current Coverage Analysis:")
        print(f"  Found:   {len(found)} characters")
        print(f"  Missing: {len(missing)} characters")

        if missing:
            print(f"\n  Missing: {', '.join(repr(c) for c in missing)}")

        # Extend vocabulary
        new_tokens = extender.extend_vocabulary()
        new_vocab_size = extender.apply_extension(new_tokens)

        # Update config files
        if not args.no_update_config and new_tokens:
            config_files = []

            if args.config:
                config_files.append(args.config)
            else:
                # Auto-detect config files
                # Only update config.py - inference.py imports from it
                vendor_dir = args.tokenizer.parent.parent
                config_files = [
                    vendor_dir / "src" / "config.py"
                ]

            print(f"\n📝 Updating configuration files:")
            extender.update_config_files(new_vocab_size, config_files)

        # Print final report
        extender.print_report(new_tokens, new_vocab_size)

        if args.dry_run:
            print("\n⚠️  DRY RUN - No changes were made")
            print("   Run without --dry-run to apply changes")
        else:
            print("\n✅ Extension complete!")
            print(f"\n📋 Next steps:")
            print(f"   1. Verify extension:")
            print(f"      python scripts/verify_tokenizer.py {args.tokenizer} \\")
            print(f"        data/processed/MyTTSDataset/metadata.csv")
            print(f"   2. Proceed to preprocessing (Task 6)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
