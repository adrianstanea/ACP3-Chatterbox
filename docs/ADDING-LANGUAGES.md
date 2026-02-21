# Adding New Languages to Chatterbox

This guide explains how to extend the Chatterbox tokenizer to support additional languages not covered in the base vocabulary.

## Overview

Chatterbox uses a grapheme (character-level) tokenizer that supports 23 languages by default. If your target language has special characters not in the base tokenizer, you can easily extend it using our language-agnostic extension system.

## Quick Start

### 1. Create Language Extension Config

Create a JSON file in `scripts/vocab_extensions/` with your language's special characters:

```json
{
  "language": "YourLanguage",
  "language_code": "xy",
  "description": "Brief description of the characters",
  "characters": [
    {
      "char": "ñ",
      "unicode": "U+00F1",
      "name": "Latin Small Letter N with Tilde",
      "frequency_rank": "high",
      "notes": "Essential for Spanish"
    }
  ]
}
```

### 2. Run Extension Script

```bash
# Preview changes (dry run)
python scripts/extend_tokenizer.py --language yourlanguage --dry-run

# Apply extension
python scripts/extend_tokenizer.py --language yourlanguage
```

### 3. Verify Extension

```bash
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/YourDataset/metadata.csv
```

## Detailed Guide

### Step 1: Identify Missing Characters

First, determine which characters your language needs:

#### Method A: Automated Analysis
Use the verification script on your dataset:

```bash
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/YourDataset/metadata.csv
```

This will report missing characters with frequency statistics.

#### Method B: Manual Identification
List all special characters in your language:
- Diacritics (accents, tildes, etc.)
- Language-specific letters
- Both uppercase and lowercase variants
- Common digraphs (if treated as single units)

**Examples by language:**
- **Spanish**: ñ, Ñ, ¿, ¡
- **French**: é, è, ê, ë, à, ù, ç, œ, æ
- **German**: ä, ö, ü, ß, Ä, Ö, Ü
- **Vietnamese**: ă, â, đ, ê, ô, ơ, ư (+ tones)
- **Polish**: ą, ć, ę, ł, ń, ó, ś, ź, ż
- **Turkish**: ğ, ı, ş, ö, ü, ç, İ
- **Romanian**: ă, â, î, ș, ț

### Step 2: Create Language Extension Config

Create `scripts/vocab_extensions/yourlanguage.json`:

```json
{
  "language": "YourLanguage",
  "language_code": "xy",
  "description": "Character set extension for YourLanguage TTS",

  "characters": [
    {
      "char": "ñ",
      "unicode": "U+00F1",
      "name": "Latin Small Letter N with Tilde",
      "frequency_rank": "high",
      "notes": "Very common, ~2% of text"
    },
    {
      "char": "Ñ",
      "unicode": "U+00D1",
      "name": "Latin Capital Letter N with Tilde",
      "frequency_rank": "low",
      "notes": "Uppercase variant, sentence starts"
    }
  ],

  "validation": {
    "expected_coverage": [
      "all", "required", "characters"
    ],
    "test_phrases": [
      "Example phrase 1",
      "Example phrase 2 with special chars"
    ]
  },

  "metadata": {
    "contributor": "Your Name",
    "dataset": "Dataset Name",
    "date": "2026-02-21",
    "references": [
      "https://example.com/language-alphabet"
    ]
  }
}
```

#### Field Descriptions

**Required fields:**
- `language`: Full language name
- `language_code`: ISO 639-1 code (e.g., "ro", "es", "vi")
- `characters`: Array of character objects

**Character object fields:**
- `char` (required): The character itself
- `unicode` (recommended): Unicode code point (for documentation)
- `name` (recommended): Official Unicode character name
- `frequency_rank` (optional): "high", "medium", or "low"
- `notes` (optional): Additional context

**Optional sections:**
- `validation`: Test phrases and expected coverage
- `metadata`: Attribution and references

### Step 3: Run Extension Script

#### Preview Changes (Recommended First)

```bash
python scripts/extend_tokenizer.py --language yourlanguage --dry-run
```

This shows what will be changed without modifying files.

#### Apply Extension

```bash
python scripts/extend_tokenizer.py --language yourlanguage
```

The script will:
1. ✅ Create backups of tokenizer.json and config files
2. ✅ Add missing characters with sequential IDs
3. ✅ Update `new_vocab_size` in config.py and inference.py
4. ✅ Print detailed report

#### Custom Paths

```bash
python scripts/extend_tokenizer.py \
  --language yourlanguage \
  --tokenizer /path/to/tokenizer.json \
  --config /path/to/config.py
```

### Step 4: Verify Extension

Run the verification script to confirm all characters are present:

```bash
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/YourDataset/metadata.csv
```

Expected output:
```
✓ All Romanian characters are covered by the tokenizer.
✓ No action needed - proceed with preprocessing.
```

### Step 5: Test with Your Dataset

Before full training, verify the extension works:

1. **Run preprocessing**:
   ```bash
   cd vendor/chatterbox-finetuning
   python train.py  # Will preprocess first
   ```

2. **Check for errors** in character encoding
3. **Inspect preprocessed files** for correct handling

## Example: Romanian Extension

See [`scripts/vocab_extensions/romanian.json`](../scripts/vocab_extensions/romanian.json) for a complete example.

**Summary:**
- Missing characters: ș, ț, Ă, Ș, Ț
- Frequency: ș (13,135 occurrences), ț (12,762 occurrences)
- Extension: 2454 → 2459 tokens

**Usage:**
```bash
python scripts/extend_tokenizer.py --language romanian
```

## Advanced Usage

### Multiple Languages

Extend for multiple languages:

```bash
python scripts/extend_tokenizer.py --language romanian
python scripts/extend_tokenizer.py --language vietnamese
python scripts/extend_tokenizer.py --language polish
```

Each extension is cumulative and preserves previous additions.

### Skip Config Updates

If you want to manually update config files:

```bash
python scripts/extend_tokenizer.py --language yourlanguage --no-update-config
```

Then manually update:
- `vendor/chatterbox-finetuning/src/config.py`
- `vendor/chatterbox-finetuning/inference.py`

Set `new_vocab_size` to the new vocabulary size reported by the script.

### Rollback

If you need to rollback:

1. **Restore from backups**:
   ```bash
   # Backups are timestamped, e.g., tokenizer.backup_20260221_153045.json
   cp vendor/chatterbox-finetuning/pretrained_models/tokenizer.backup_*.json \
      vendor/chatterbox-finetuning/pretrained_models/tokenizer.json
   ```

2. **Or re-download** base tokenizer:
   ```bash
   cd vendor/chatterbox-finetuning
   python setup.py
   ```

## Contributing New Languages

We welcome contributions of language extensions!

### Contribution Checklist

- [ ] Create well-documented language config JSON
- [ ] Test extension with real dataset
- [ ] Verify characters cover at least 95% of corpus
- [ ] Include validation test phrases
- [ ] Add metadata (contributor, dataset, references)
- [ ] Test training with extended tokenizer
- [ ] Document any language-specific considerations

### Submission

1. Fork the repository
2. Add your language config to `scripts/vocab_extensions/`
3. Test thoroughly with your dataset
4. Submit pull request with:
   - Language config file
   - Test results
   - Brief description of your use case

## Troubleshooting

### "Character already exists in vocabulary"

This is informational, not an error. The script will skip characters already present.

### "Could not find vocab_size pattern in config.py"

The config file format may have changed. Update manually:
```python
new_vocab_size: int = 52260 if is_turbo else YOUR_NEW_SIZE
```

### "Extension applied but verification still shows missing characters"

Possible causes:
1. **Wrong tokenizer path**: Verify you're checking the same file you extended
2. **Cache issues**: Restart Python kernel or reload the file
3. **Encoding issues**: Ensure your terminal/editor uses UTF-8

### Characters render incorrectly

This is a display issue, not a tokenizer issue. The characters are correctly stored in JSON. Check:
- Terminal font supports the character set
- Editor encoding is set to UTF-8
- System locale supports the language

## Best Practices

### Character Selection

✅ **Do include:**
- All standard diacritics for your language
- Both uppercase and lowercase variants
- Characters that appear in >0.01% of your corpus

❌ **Don't include:**
- Rare historical characters (unless in your dataset)
- Characters from other languages (create separate configs)
- Combining diacritics (use pre-composed forms)

### Testing

1. **Start with dry run**: Always preview changes first
2. **Test on small subset**: Verify with 100-1000 utterances before full training
3. **Monitor training**: Check for unusual loss patterns
4. **Validate inference**: Test pronunciation of special characters

### Documentation

Document your extension:
- **Why**: Reason for the extension
- **What**: Which characters and why
- **How**: Frequency data from your corpus
- **Results**: Training outcomes (after completion)

## FAQ

**Q: Will this affect the pre-trained model?**
A: No. The extension only affects the tokenizer and model head. Pre-trained weights are preserved, and new character embeddings are initialized intelligently.

**Q: Can I remove characters later?**
A: Not recommended once training starts. Instead, create a fresh extension config for new projects.

**Q: How many characters can I add?**
A: No hard limit, but keep it minimal. Each character adds to the model size. Typically 5-20 characters per language.

**Q: Does this work for non-Latin scripts?**
A: Yes! The system is script-agnostic. See the Cyrillic, Arabic, or CJK examples in the base tokenizer.

**Q: Can I extend Turbo mode?**
A: Yes, but Turbo uses BPE tokenizer which handles most characters automatically. Extensions are rarely needed for Turbo.

## References

- [Chatterbox Paper](https://arxiv.org/abs/2506.08387)
- [Unicode Character Database](https://unicode.org/charts/)
- [Language Alphabets](https://www.omniglot.com/)

## Support

For questions or issues:
1. Check this documentation
2. Review example configs in `scripts/vocab_extensions/`
3. Run with `--dry-run` to preview changes
4. Open an issue on GitHub with your language config

---

**Happy multilingual TTS training! 🌍🎙️**
