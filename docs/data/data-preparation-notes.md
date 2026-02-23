# Data Preparation Notes

**Project:** Romanian Chatterbox Adaptation
**Phase:** Data Preparation (Task 4)
**Date:** February 21, 2026

## Overview

This document records technical notes and decisions made during the SWARA dataset conversion to LJSpeech format for Chatterbox fine-tuning. It serves as a reference for understanding the data pipeline and troubleshooting issues.

## SWARA Dataset Format

### Original Metadata Format

**File:** `metadata_SWARA1.0_text.csv`

**Format:** Pipe-delimited CSV (no header)
```
path|text|speaker_id
```

**Example:**
```
/media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/bas_rnd1_001.wav|Aceasta este o propoziție în limba română.|bas
```

### Field Descriptions

1. **Path:** Absolute path on original system (not valid on our system)
2. **Text:** Romanian transcript with diacritics
3. **Speaker ID:** Three-letter speaker identifier

### Path Mapping Challenge

**Problem:** Paths in metadata don't match actual file locations

**Metadata Path:**
```
/media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/bas_rnd1_001.wav
```

**Actual Path (Local):**
```
/home/astanea/data/SWARA1.0_22k_noSil/bas_rnd1_001.wav
```

**Actual Path (DGX):**
```
/mnt/data/SWARA1.0_22k_noSil/bas_rnd1_001.wav
```

**Solution:** Extract filename only, map to actual directory

```python
def map_metadata_path_to_actual(metadata_path: str, actual_base_dir: str) -> str:
    filename = os.path.basename(metadata_path)
    actual_path = os.path.join(actual_base_dir, filename)
    return actual_path
```

## LJSpeech Format Requirements

### Target Format

**Directory Structure:**
```
data/processed/MyTTSDataset/
├── metadata.csv
└── wavs/
    ├── bas_rnd1_001.wav
    ├── bas_rnd1_002.wav
    └── ...
```

### Metadata CSV Format

**Format:** Pipe-delimited (no header)
```
filename|raw_text|normalized_text
```

**Example:**
```
bas_rnd1_001|Aceasta este o propoziție în limba română.|aceasta este o propoziție în limba română
```

### Field Descriptions

1. **Filename:** Audio filename (without path or extension)
2. **Raw Text:** Original transcript
3. **Normalized Text:** Text after normalization (punctuation, casing)

### Audio Files

**Location:** `wavs/` subdirectory

**Naming:** Same as metadata filename
- Metadata: `bas_rnd1_001`
- Audio: `wavs/bas_rnd1_001.wav`

**No Path Modification:** Files stay in original format (22kHz, 16-bit, mono)

## Conversion Strategy

### Design Principle

**Never Alter Original Dataset**

**Rationale:**
1. Preserves provenance (can verify against original)
2. Enables re-processing with different parameters
3. Respects dataset license terms
4. Allows other researchers to replicate

**Implementation:**
- Original dataset: Read-only mount in Docker
- Processed dataset: Separate output directory
- Symlinks or copies to `wavs/` directory

### Symlink vs Copy Decision

**Initial Plan:** Symlinks (save disk space)

**Implementation Choice:** Symlinks for local, copy for DGX

**Rationale:**

**Local Development (Symlinks):**
```python
# Create symlink
os.symlink(source_audio, target_audio)
```
- **Pros:** No disk space duplication, instant
- **Cons:** Breaks if source moves
- **Use case:** Local testing with small dataset

**DGX Production (Copy):**
```python
# Copy file
shutil.copy2(source_audio, target_audio)
```
- **Pros:** Self-contained, survives source changes, better I/O
- **Cons:** Duplicates ~3GB of data
- **Use case:** Production training, preprocessed data on NVMe

**Configuration:**
```python
# In conversion script
args.add_argument('--copy', action='store_true',
                  help='Copy files instead of symlinking (recommended for production)')
```

### Validation Split Implementation

**Per-Speaker Stratification:**

```python
# Group utterances by speaker
speaker_utterances = defaultdict(list)
for row in metadata:
    speaker_id = extract_speaker_id(row[0])
    speaker_utterances[speaker_id].append(row)

# Split each speaker 90/10
train_data = []
val_data = []

for speaker_id, utterances in speaker_utterances.items():
    if speaker_id in HOLDOUT_SPEAKERS:
        # All utterances go to test set (not written to metadata.csv)
        continue

    # Shuffle and split
    random.shuffle(utterances)
    split_point = int(len(utterances) * 0.9)

    train_data.extend(utterances[:split_point])
    val_data.extend(utterances[split_point:])
```

**Holdout Speakers:**
```python
HOLDOUT_SPEAKERS = {'bas', 'sgs'}
```

**Output Files:**
- `metadata.csv` - Training set (90% of non-holdout speakers)
- `metadata_val.csv` - Validation set (10% of non-holdout speakers)
- Holdout data not written (reserved for evaluation phase)

## Text Normalization

### Normalization Function

**Choice:** Chatterbox `punc_norm`

**Usage:**
```python
from chatterbox.utils import punc_norm

raw_text = "Aceasta este o propoziție, în limba română!"
normalized_text = punc_norm(raw_text)
# Result: normalized punctuation, preserved diacritics
```

**What It Does:**
- Normalizes quotation marks (", ", ', ')
- Normalizes dashes (–, —, -)
- Normalizes whitespace (multiple spaces → single space)
- **Preserves:** Diacritics, casing, letters

**What It Does NOT Do:**
- Lower-casing (preserved for proper nouns)
- Number expansion (already done in SWARA)
- Abbreviation expansion (handle separately if needed)

### Romanian-Specific Considerations

**Diacritics (Critical):**

Must preserve all 10 Romanian diacritics:
- ă, Ă (a-breve)
- â, Â (a-circumflex)
- î, Î (i-circumflex)
- ș, Ș (s-comma)
- ț, Ț (t-comma)

**Verification:**
```python
ROMANIAN_DIACRITICS = {'ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț'}

def verify_diacritics(text_before: str, text_after: str):
    diacritics_before = sum(1 for c in text_before if c in ROMANIAN_DIACRITICS)
    diacritics_after = sum(1 for c in text_after if c in ROMANIAN_DIACRITICS)

    if diacritics_before != diacritics_after:
        raise ValueError(f"Diacritics lost: {diacritics_before} → {diacritics_after}")
```

**Numbers (Already Expanded):**

SWARA dataset already has numbers as words:
- "123" → "o sută douăzeci și trei"
- "2026" → "două mii douăzeci și șase"

**No additional processing needed.**

**Abbreviations (If Present):**

Common Romanian abbreviations:
```python
ABBREVIATIONS = {
    'str.': 'strada',
    'nr.': 'numărul',
    'dr.': 'doctor',
    'prof.': 'profesor',
    'ing.': 'inginer',
    'etc.': 'etcetera',
}

def expand_abbreviations(text: str) -> str:
    for abbr, expansion in ABBREVIATIONS.items():
        text = text.replace(f' {abbr}', f' {expansion}')
    return text
```

**Applied before `punc_norm`:**
```python
text = expand_abbreviations(raw_text)
text = punc_norm(text)
```

## Speaker ID Extraction

### Naming Convention

**Filename Format:**
```
{speaker_id}_{recording_type}_{number}.wav
```

**Examples:**
- `bas_rnd1_001.wav` → Speaker: `bas`
- `sgs_rnd2_042.wav` → Speaker: `sgs`
- `ana_rnd1_015.wav` → Speaker: `ana`

**Extraction Function:**
```python
def extract_speaker_id(filename: str) -> str:
    """Extract speaker ID from filename (prefix before first underscore)."""
    basename = os.path.basename(filename)
    speaker_id = basename.split('_')[0]
    return speaker_id
```

**Validation:**
```python
# Verify all extracted speaker IDs are valid
valid_speakers = {'bas', 'sgs', 'ana', ...}  # Known from dataset
for filename in all_files:
    speaker = extract_speaker_id(filename)
    assert speaker in valid_speakers, f"Unknown speaker: {speaker}"
```

## Quality Assurance

### Pre-Conversion Checks

**1. File Existence:**
```python
missing_files = []
for metadata_path in metadata_paths:
    actual_path = map_to_actual(metadata_path)
    if not os.path.exists(actual_path):
        missing_files.append(actual_path)

if missing_files:
    print(f"ERROR: {len(missing_files)} missing files")
    # Decision: Skip or abort?
```

**2. Audio Format Verification:**
```python
import soundfile as sf

for audio_path in audio_files:
    info = sf.info(audio_path)
    assert info.samplerate == 22050, f"Unexpected sample rate: {info.samplerate}"
    assert info.channels == 1, f"Not mono: {info.channels} channels"
```

**3. Text Quality:**
```python
# Check for Romanian diacritics
all_chars = set(''.join(all_texts))
romanian_chars_found = all_chars & ROMANIAN_DIACRITICS

if len(romanian_chars_found) < 10:
    print("WARNING: Not all Romanian diacritics found in text!")
```

### Post-Conversion Checks

**1. Metadata Integrity:**
```python
# Read generated metadata
with open('metadata.csv') as f:
    lines = f.readlines()

# Verify format
for line in lines:
    parts = line.strip().split('|')
    assert len(parts) == 3, f"Malformed line: {line}"

    filename, raw_text, norm_text = parts

    # Verify audio exists
    audio_path = f"wavs/{filename}.wav"
    assert os.path.exists(audio_path), f"Missing audio: {audio_path}"

    # Verify text not empty
    assert len(raw_text) > 0, "Empty raw text"
    assert len(norm_text) > 0, "Empty normalized text"
```

**2. Split Balance:**
```python
# Verify train/val split
train_count = len(read_metadata('metadata.csv'))
val_count = len(read_metadata('metadata_val.csv'))

ratio = train_count / (train_count + val_count)
assert 0.85 <= ratio <= 0.95, f"Unexpected split ratio: {ratio:.2f}"
```

**3. Speaker Distribution:**
```python
# Verify per-speaker splitting
train_speakers = set(extract_speakers('metadata.csv'))
val_speakers = set(extract_speakers('metadata_val.csv'))

assert train_speakers == val_speakers, "Speaker leakage detected!"
```

## Conversion Script Implementation

### Script: `scripts/convert_swara_to_ljspeech.py`

**Key Features:**

1. **Path Mapping:** Handles different dataset locations
2. **Validation Split:** Per-speaker stratification
3. **Text Normalization:** Chatterbox punc_norm
4. **Holdout Handling:** Reserves BAS and SGS
5. **Quality Checks:** Verifies all files exist
6. **Flexible Output:** Symlinks or copies

**Usage:**
```bash
python scripts/convert_swara_to_ljspeech.py \
    --metadata metadata_SWARA1.0_text.csv \
    --audio-dir /data/swara \
    --output-dir data/processed/MyTTSDataset \
    --val-ratio 0.1 \
    --seed 42 \
    --copy  # Use --copy for production, omit for local
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--metadata` | SWARA metadata file | Required |
| `--audio-dir` | SWARA audio directory | `$SWARA_PATH` or `/data/swara` |
| `--output-dir` | LJSpeech output directory | `data/processed/MyTTSDataset` |
| `--val-ratio` | Validation split ratio | `0.1` (10%) |
| `--seed` | Random seed | `42` |
| `--copy` | Copy files instead of symlink | Flag (default: symlink) |
| `--holdout-speakers` | Speakers to exclude | `bas,sgs` |

**Output:**
```
data/processed/MyTTSDataset/
├── metadata.csv          # Training set
├── metadata_val.csv      # Validation set
├── conversion_log.txt    # Conversion statistics
└── wavs/
    ├── bas_rnd1_001.wav  # (symlink or copy)
    └── ...
```

### Conversion Statistics

**Logged Information:**

```
Total entries in metadata: 21,304
Holdout speakers (excluded): bas, sgs (2,400 utterances)
Available for training: 18,904 utterances from 16 speakers
Training set: 17,013 utterances (90%)
Validation set: 1,891 utterances (10%)

Files created: 18,904
Symlinks created: 18,904 (or "Files copied: 18,904")

Diacritics preserved: ✓
All files verified: ✓
```

## Troubleshooting

### Issue: Missing Files

**Symptom:**
```
FileNotFoundError: /data/swara/bas_rnd1_001.wav
```

**Diagnosis:**
1. Check metadata paths vs actual paths
2. Verify `--audio-dir` argument
3. Check environment variable `SWARA_PATH`

**Solution:**
```bash
# Verify dataset location
ls -la /data/swara | head

# Set correct path
python convert_swara_to_ljspeech.py --audio-dir /actual/path/to/SWARA1.0_22k_noSil
```

### Issue: Diacritics Lost

**Symptom:**
```
WARNING: Diacritics count changed: 1500 → 1200
```

**Diagnosis:**
- Unicode encoding issue
- Incorrect normalization function
- File encoding (UTF-8 vs Latin-1)

**Solution:**
```python
# Ensure UTF-8 encoding
with open(metadata_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)

with open(output_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
```

### Issue: Speaker Leakage

**Symptom:**
```
AssertionError: Speaker 'ana' in training but not in validation
```

**Diagnosis:**
- Stratification logic error
- Holdout speaker accidentally included

**Solution:**
```python
# Verify stratification
for speaker_id in train_speakers:
    assert speaker_id in val_speakers, f"Speaker {speaker_id} only in training!"
```

### Issue: Symlink Broken

**Symptom:**
```
FileNotFoundError: wavs/bas_rnd1_001.wav (symlink target doesn't exist)
```

**Diagnosis:**
- Source dataset moved
- Relative symlink instead of absolute

**Solution:**
```bash
# Use absolute symlinks
os.symlink(os.path.abspath(source), target)

# Or use --copy flag for production
python convert_swara_to_ljspeech.py --copy
```

## Best Practices

### 1. Always Verify Output

```bash
# After conversion
python scripts/verify_ljspeech.py data/processed/MyTTSDataset
```

### 2. Document Statistics

```bash
# Save conversion log
python convert_swara_to_ljspeech.py ... 2>&1 | tee conversion.log
```

### 3. Reproducible Seeds

```python
# Always use same seed for reproducibility
random.seed(42)
np.random.seed(42)
```

### 4. Validate Sample

```bash
# Check first 5 entries manually
head -5 data/processed/MyTTSDataset/metadata.csv

# Listen to audio
ffplay data/processed/MyTTSDataset/wavs/bas_rnd1_001.wav
```

### 5. Backup Original

```bash
# Before conversion
cp metadata_SWARA1.0_text.csv metadata_SWARA1.0_text.csv.backup
```

## Performance Considerations

### Conversion Speed

**Symlink Mode:**
- Speed: ~1,000 files/second
- Total time: ~20 seconds for full dataset
- Bottleneck: CSV parsing

**Copy Mode:**
- Speed: ~100 files/second (disk I/O bound)
- Total time: ~3-5 minutes for full dataset
- Bottleneck: Disk I/O

**Optimization:**
```python
# Use multiprocessing for copying
from multiprocessing import Pool

with Pool(processes=8) as pool:
    pool.map(copy_file, file_pairs)
```

### Disk Usage

**Symlink Mode:**
- Additional space: ~10 MB (metadata only)
- Total: Original dataset size + metadata

**Copy Mode:**
- Additional space: ~3 GB (full audio duplication)
- Total: 2× original dataset size

**Recommendation:**
- Local: Symlink (save space)
- DGX: Copy to NVMe (better I/O)

## Integration with Fine-Tuning Kit

### Configuration

**File:** `vendor/chatterbox-finetuning/src/config.py`

```python
@dataclass
class TrainConfig:
    # Point to our converted dataset
    csv_path: str = "/workspace/data/processed/MyTTSDataset/metadata.csv"
    wav_dir: str = "/workspace/data/processed/MyTTSDataset/wavs"
    preprocessed_dir: str = "/workspace/data/processed/MyTTSDataset/preprocess"

    # LJSpeech format
    ljspeech: bool = True

    # Run preprocessing
    preprocess: bool = True
```

### Validation Set Usage

**Option 1: Separate Validation Run**
```python
# Train on training set
csv_path = "metadata.csv"

# Later: evaluate on validation set
csv_path = "metadata_val.csv"
preprocess = True  # Preprocess validation set
```

**Option 2: Manual Split in Training Code**
```python
# Modify dataset.py to use both files
train_dataset = load_dataset("metadata.csv")
val_dataset = load_dataset("metadata_val.csv")
```

## Future Improvements

### 1. Automated Validation

Create comprehensive validation script:
```python
# scripts/validate_ljspeech.py
- Check metadata format
- Verify all audio files exist
- Validate speaker distribution
- Check text quality (diacritics, etc.)
- Generate statistics report
```

### 2. Incremental Processing

For large datasets:
```python
# Process in batches
for batch in chunks(metadata, batch_size=1000):
    process_batch(batch)
    save_checkpoint()
```

### 3. Data Augmentation Hooks

Prepare for future augmentation:
```python
# Add augmentation flag
--augment speed,pitch,noise

# Apply during conversion
if args.augment:
    augmented_audio = apply_augmentation(audio)
```

## References

### Code
- Conversion script: `scripts/convert_swara_to_ljspeech.py`
- Analysis script: `scripts/analyze_swara.py`

### Documentation
- LJSpeech format: https://keithito.com/LJ-Speech-Dataset/
- Chatterbox normalization: https://github.com/ResembleAI/chatterbox
- Fine-tuning kit dataset docs: `vendor/chatterbox-finetuning/README.md`

### Project Docs
- [SWARA Analysis Report](swara-analysis-report.md)
- [Technical Decisions](../technical-decisions.md)
- [Design Document](../plans/2026-02-21-chatterbox-romanian-design.md)

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-02-21 | Initial conversion script | Adrian Stanea |
| 2026-02-21 | Add Chatterbox punc_norm | Adrian Stanea |
| 2026-02-21 | Implement 5-sample validation | Adrian Stanea |
| 2026-02-21 | Add per-speaker stratification | Adrian Stanea |
