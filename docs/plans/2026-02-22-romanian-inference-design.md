# Romanian Chatterbox Inference Script Design

**Date:** 2026-02-22
**Status:** Approved
**Author:** Adrian Stanea

## Overview

Design for a CLI inference script that synthesizes Romanian speech using the finetuned Chatterbox model (checkpoint-5300 from Experiment 3). The script integrates Romanian text preprocessing and exposes key generation parameters.

## Requirements

| Requirement | Choice |
|-------------|--------|
| Deployment target | CLI in Docker environment (HuggingFace deferred) |
| Output modes | Single text + batch processing |
| Evaluation | Separate step (audio generation only) |
| Batch reference | Single reference audio for all utterances |
| Parameter control | CLI flags with sensible defaults |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  scripts/inference_romanian.py                   │
├─────────────────────────────────────────────────────────────────┤
│  CLI Layer (argparse)                                           │
│  ├── Single mode: --text, --reference, --output                 │
│  └── Batch mode:  --batch input.csv, --output-dir               │
├─────────────────────────────────────────────────────────────────┤
│  RomanianTTSEngine (core class)                                 │
│  ├── __init__(checkpoint_dir, device)                           │
│  ├── load_model() → loads T3 from checkpoint                    │
│  ├── synthesize(text, reference_path, **params) → wav           │
│  └── synthesize_batch(texts, reference_path, **params) → wavs   │
├─────────────────────────────────────────────────────────────────┤
│  Imports from vendor/chatterbox-finetuning/                     │
│  ├── src.romanian_preprocessor.preprocess_and_normalize         │
│  ├── src.chatterbox_.tts.ChatterboxTTS                          │
│  └── src.chatterbox_.models.t3.t3.T3                            │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility |
|-----------|---------------|
| `RomanianTTSEngine` | Model loading, preprocessing, synthesis (reusable class) |
| CLI layer | Argument parsing, file I/O, batch orchestration |
| Romanian preprocessor | Text normalization (ș→sh, ț→ts) - imported from vendor |

**File Location:** `scripts/inference_romanian.py`

## CLI Interface

### Single Text Mode

```bash
python scripts/inference_romanian.py \
    --text "Bună ziua, ce mai faci?" \
    --reference /path/to/speaker.wav \
    --output output.wav \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output \
    --cfg-weight 0.5 \
    --exaggeration 0.5
```

### Batch Mode

```bash
python scripts/inference_romanian.py \
    --batch data/eval/test_sentences.csv \
    --reference /path/to/speaker.wav \
    --output-dir outputs/ \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--text` | Single mode | - | Romanian text to synthesize |
| `--batch` | Batch mode | - | CSV file with `id,text` columns |
| `--reference` | Yes | - | Speaker reference audio (WAV, 3-10s) |
| `--output` | Single mode | - | Output WAV path |
| `--output-dir` | Batch mode | `./outputs` | Directory for batch outputs |
| `--checkpoint` | No | `vendor/.../chatterbox_output` | Checkpoint directory |
| `--cfg-weight` | No | `0.5` | Classifier-free guidance weight |
| `--exaggeration` | No | `0.5` | Prosodic expressiveness |
| `--temperature` | No | `0.8` | Sampling temperature |
| `--seed` | No | `None` | Random seed for reproducibility |

### Batch CSV Format

```csv
id,text
001,Bună ziua, ce mai faci?
002,Țara mea frumoasă.
003,Înțelepciunea înseamnă să știi.
```

**Output:** `outputs/001.wav`, `outputs/002.wav`, etc.

## Data Flow

### Text Processing Pipeline

```
Input text (Romanian with diacritics)
    │
    ▼
┌─────────────────────────────────────┐
│ preprocess_and_normalize()          │
│ • ș/Ș → "sh"                        │
│ • ț/Ț → "ts"                        │
│ • ă/â/î kept (already in vocab)    │
│ • lowercase + punctuation norm      │
└─────────────────────────────────────┘
    │
    ▼
Preprocessed text: "Știință" → "shtiintsă"
    │
    ▼
┌─────────────────────────────────────┐
│ ChatterboxTTS.generate()            │
│ • Tokenize with grapheme tokenizer  │
│ • T3 transformer (finetuned)        │
│ • S3Gen vocoder                     │
│ • Perth watermarking                │
└─────────────────────────────────────┘
    │
    ▼
Output: WAV audio (24kHz)
```

### Model Loading Sequence

```python
# 1. Load base Chatterbox (pretrained components)
engine = ChatterboxTTS.from_local(pretrained_dir, device="cpu")

# 2. Create new T3 with original vocab size (2454)
t3_config = engine.t3.hp
t3_config.text_tokens_dict_size = 2454
new_t3 = T3(hp=t3_config)

# 3. Load finetuned weights from checkpoint
state_dict = load_file(checkpoint_dir / "model.safetensors")
new_t3.load_state_dict(state_dict)

# 4. Replace T3 in engine
engine.t3 = new_t3.to(device).eval()
```

**Key point:** Romanian preprocessing happens *before* the model sees the text, so the finetuned T3 only ever sees tokens it was trained on (IDs 0-2453).

## Error Handling

### Input Validation

| Error Case | Handling |
|------------|----------|
| Empty text | Log warning, skip (batch) or exit with message (single) |
| Text too long (>256 tokens) | Truncate with warning, or split into sentences |
| Reference audio missing | Exit with clear error message |
| Reference audio too short (<1s) | Warning (may affect voice quality) |
| Reference audio wrong format | Attempt conversion via torchaudio, or exit with error |
| Checkpoint not found | Exit with error pointing to expected path |
| CUDA OOM | Catch, suggest reducing batch size or text length |

### Batch Mode Resilience

```python
from tqdm import tqdm

failed = []
for row in tqdm(batch, desc="Synthesizing", unit="utterance"):
    try:
        wav = engine.synthesize(row.text, reference)
        save_wav(wav, output_dir / f"{row.id}.wav")
    except Exception as e:
        log.error(f"Failed {row.id}: {e}")
        failed.append(row.id)

# Final report
if failed:
    log.warning(f"Failed utterances: {failed}")
```

### Logging

- Use Python `logging` module (consistent with existing vendor code)
- Default level: INFO (show progress)
- `--verbose` flag for DEBUG level

## Testing & Verification

### Smoke Test

```bash
# Single utterance - verify audio is intelligible Romanian
python scripts/inference_romanian.py \
    --text "Bună ziua, mă numesc Maria." \
    --reference data/processed/MyTTSDataset/wavs/bas_rnd1_011.wav \
    --output test_output.wav \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output

# Play and verify
aplay test_output.wav
```

### Verification Checklist

| Check | Method |
|-------|--------|
| Script runs without error | Execute with test input |
| Romanian preprocessing applied | Log shows ș→sh, ț→ts mappings |
| Checkpoint loads correctly | Log shows "Loaded checkpoint" |
| Audio is intelligible | Listen to output |
| Parameters affect output | Compare --exaggeration 0.3 vs 0.7 |
| Batch mode works | Run on 5-10 test sentences |
| Failed rows don't crash batch | Include one invalid row, verify others complete |

### Test Sentences

```csv
id,text
test_01,Bună ziua, ce mai faci?
test_02,Știința și înțelepciunea sunt importante.
test_03,Țara mea frumoasă are munți înalți.
test_04,De asemenea, contează și dacă imobilul este la stradă.
test_05,Ăsta e un test pentru vocala ă.
```

## Future Work

- **HuggingFace Spaces deployment:** Gradio UI wrapper around `RomanianTTSEngine`
- **Evaluation integration:** Optional `--evaluate` flag to compute WER/CER
- **Streaming synthesis:** For long texts, stream audio chunks
- **Multi-speaker batch:** Per-row reference audio support

## References

- Experiment 3 checkpoint: `vendor/chatterbox-finetuning/chatterbox_output/checkpoint-5300/`
- Romanian preprocessor: `vendor/chatterbox-finetuning/src/romanian_preprocessor.py`
- Existing inference code: `vendor/chatterbox-finetuning/inference.py`
- Original Chatterbox: https://github.com/resemble-ai/chatterbox
