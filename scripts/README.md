# Scripts Directory

This directory contains utility scripts for the Romanian adaptation of Chatterbox TTS.

## Scripts

### 1. analyze_swara.py
Analyzes the SWARA dataset to extract statistics and prepare for conversion.

**Usage**:
```bash
python scripts/analyze_swara.py /path/to/swara/dataset
```

**Output**:
- Dataset statistics (number of speakers, recordings, duration)
- Speaker distribution analysis
- Audio format and quality metrics

---

### 2. convert_swara_to_ljspeech.py
Converts SWARA dataset format to LJSpeech format compatible with Chatterbox.

**Usage**:
```bash
python scripts/convert_swara_to_ljspeech.py \
    --input /data/swara/myTTS_corpus_final \
    --output data/processed/MyTTSDataset \
    --speaker-id "speaker_id"
```

**Features**:
- Converts SWARA metadata to LJSpeech format (ID|RawText|NormText)
- Copies and processes audio files
- Validates output structure
- Generates dataset statistics

---

### 3. verify_tokenizer.py
Verifies that the Chatterbox tokenizer includes all Romanian diacritical characters.

**Purpose**: Ensure the grapheme tokenizer covers Romanian characters (ă, â, î, ș, ț + capitals)

**Usage**:
```bash
# Basic verification
python scripts/verify_tokenizer.py <tokenizer_json_path>

# With metadata analysis (recommended)
python scripts/verify_tokenizer.py <tokenizer_json_path> <metadata_csv_path>
```

**Examples**:
```bash
# Inside Docker container
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

**Features**:
- Loads tokenizer vocabulary from JSON
- Checks coverage for all 10 Romanian diacritics
- Analyzes common grapheme patterns (ea, oa, ia, etc.)
- If metadata provided: analyzes character usage in dataset
- Generates recommendations for tokenizer extension

**Output**:
- Character coverage report
- Grapheme pattern analysis
- Character usage statistics from dataset
- Actionable recommendations

**Exit Codes**:
- 0: Success - all characters present
- 1: Warning - missing characters, extension needed

**Romanian Characters Checked**:
- Lowercase: ă, â, î, ș, ț
- Uppercase: Ă, Â, Î, Ș, Ț

**See Also**:
- `/docs/tokenizer-verification.md` - Full verification documentation
- `/docs/TOKENIZER-SETUP-GUIDE.md` - Quick setup guide

---

## Development Notes

### Running Scripts in Docker

Most preprocessing scripts need to run inside the Docker container because they require dependencies (transformers, chatterbox-tts, etc.).

**Container workflow**:
```bash
# Start container
docker compose up -d chatterbox

# Enter container
docker compose exec chatterbox bash

# Inside container, navigate to workspace
cd /workspace

# Run scripts
python scripts/script_name.py [args]
```

### Script Dependencies

Scripts in this directory may have different dependencies:

- `analyze_swara.py`: pandas, numpy (can run outside container)
- `convert_swara_to_ljspeech.py`: pandas, numpy, soundfile (can run outside container)
- `verify_tokenizer.py`: json, collections (Python stdlib only - can run outside container, but tokenizer.json only exists after setup.py runs inside container)

### Adding New Scripts

When adding new scripts:

1. Add shebang: `#!/usr/bin/env python3`
2. Include docstring with usage information
3. Make executable: `chmod +x scripts/script_name.py`
4. Update this README with script description
5. Document whether it needs to run in container

---

## Workflow Order

The scripts are designed to be run in this order:

1. **analyze_swara.py** - Understand the dataset
2. **convert_swara_to_ljspeech.py** - Convert to LJSpeech format
3. **verify_tokenizer.py** - Verify tokenizer coverage (in container)
4. *(Future)* Preprocessing pipeline (in container)
5. *(Future)* Training and evaluation scripts

---

## Troubleshooting

### ImportError in scripts
- Ensure you're inside the Docker container for scripts that need dependencies
- Check `requirements.txt` is installed: `pip install -r requirements.txt`

### File not found errors
- Use absolute paths when possible
- Verify the container volume mounts in `docker-compose.yml`
- Inside container, workspace is mounted at `/workspace`

### Permission errors
- Scripts should be executable: `chmod +x scripts/*.py`
- For Docker: ensure files are owned by correct user

---

## References

- [Chatterbox TTS](https://github.com/ResembleAI/chatterbox)
- [SWARA Dataset](https://www.swaracorpus.com/)
- [LJSpeech Format](https://keithito.com/LJ-Speech-Dataset/)
