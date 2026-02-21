# Language Vocabulary Extensions

This directory contains language-specific character extensions for the Chatterbox tokenizer.

## Available Languages

- **Romanian** (`romanian.json`) - Adds ș, ț, Ă, Ș, Ț characters

## Adding a New Language

1. Create `yourlanguage.json` in this directory
2. Follow the schema documented in `../docs/ADDING-LANGUAGES.md`
3. Test with: `python ../extend_tokenizer.py --language yourlanguage --dry-run`
4. Apply with: `python ../extend_tokenizer.py --language yourlanguage`

## JSON Schema

```json
{
  "language": "LanguageName",
  "language_code": "xy",
  "description": "Brief description",
  "characters": [
    {
      "char": "ñ",
      "unicode": "U+00F1",
      "name": "Unicode character name",
      "frequency_rank": "high|medium|low",
      "notes": "Additional context"
    }
  ],
  "validation": {
    "expected_coverage": ["list", "of", "all", "chars"],
    "test_phrases": ["Test phrase 1", "Test phrase 2"]
  },
  "metadata": {
    "contributor": "Your Name",
    "dataset": "Dataset Name",
    "date": "YYYY-MM-DD",
    "references": ["URL1", "URL2"]
  }
}
```

## Examples Needed

We welcome contributions for:
- Vietnamese (tonal diacritics)
- Polish (ą, ć, ę, ł, ń, ó, ś, ź, ż)
- Czech (ř, ě, ů, etc.)
- Hungarian (ő, ű, etc.)
- Arabic script languages
- Cyrillic languages beyond Russian
- Any other language with special characters!

See `../docs/ADDING-LANGUAGES.md` for complete guide.
