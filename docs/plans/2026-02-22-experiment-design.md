# Romanian Chatterbox Experiment Design

**Date:** 2026-02-22
**Status:** Approved
**Author:** Adrian Stanea

## Overview

Design for a systematic evaluation of the Romanian Chatterbox TTS model across different generation parameters. Includes HuggingFace-compatible inference class, experiment runner, metrics integration, and academic-style GitHub Pages gallery.

## Goals

1. Create HF-compatible Romanian TTS class for future model release
2. Evaluate model quality across exaggeration and CFG parameter combinations
3. Generate metrics using established TTS evaluation pipeline
4. Provide clean, academic audio comparison gallery

## Experiment Matrix

| Speaker | Exaggeration | CFG | Samples |
|---------|--------------|-----|---------|
| BAS | 0.3 | 0.3 | 50 |
| BAS | 0.3 | 0.5 | 50 |
| BAS | 0.7 | 0.3 | 50 |
| BAS | 0.7 | 0.5 | 50 |
| SGS | 0.3 | 0.3 | 50 |
| SGS | 0.3 | 0.5 | 50 |
| SGS | 0.7 | 0.3 | 50 |
| SGS | 0.7 | 0.5 | 50 |

**Total:** 8 conditions × 50 samples = 400 audio files

### Parameter Descriptions

**Exaggeration** (prosodic expressiveness):
- 0.3: Calm, neutral delivery
- 0.7: More dramatic, expressive

**CFG** (classifier-free guidance):
- 0.3: Slower, more deliberate speech
- 0.5: Natural pacing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        src/chatterbox_ro.py                         │
│  ChatterboxRomanian class (HF-compatible)                           │
│  ├── from_local(checkpoint_dir, pretrained_dir, device)             │
│  ├── from_pretrained(repo_id, device)  [future]                     │
│  └── generate(text, reference_audio, cfg_weight, exaggeration)      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    scripts/run_experiment.py                        │
│  Generates 400 samples across experiment matrix                     │
│  ├── Reads holdout CSV for ground truth text                        │
│  ├── Selects reference audio per speaker                            │
│  ├── Outputs: audio files + filelists for metrics                   │
│  └── Saves experiment_config.json                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TTS-Evaluation-Pipeline (external)                     │
│  ├── transcription.py: Whisper transcription                        │
│  └── metrics.py: CER, WER, UTMOS, MCD, F0 RMSE, Speaker Sim         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  scripts/aggregate_metrics.py                       │
│  Computes mean/std per condition, outputs summary CSV               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    scripts/build_gallery.py                         │
│  Generates static HTML site in docs/ for GitHub Pages               │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ChatterboxRomanian Class

**File:** `src/chatterbox_ro.py`

```python
class ChatterboxRomanian:
    """
    Romanian-adapted Chatterbox TTS with HuggingFace-style loading.

    Usage:
        tts = ChatterboxRomanian.from_local(
            checkpoint_dir="path/to/finetuned",
            pretrained_dir="path/to/pretrained",
            device="cuda"
        )
        wav = tts.generate("Bună ziua!", reference_audio="speaker.wav")
    """

    VOCAB_SIZE = 2454  # Original vocab, no extension
    SAMPLE_RATE = 24000

    @classmethod
    def from_local(cls, checkpoint_dir, pretrained_dir, device):
        """Load from local checkpoint directory."""
        # Load pretrained: ve, s3gen, tokenizer
        # Load finetuned T3 with key prefix stripping
        pass

    @classmethod
    def from_pretrained(cls, repo_id="adrianstanea/chatterbox-romanian", device="cuda"):
        """Load from HuggingFace Hub (future)."""
        raise NotImplementedError("HF release pending. Use from_local() for now.")

    def generate(
        self,
        text: str,
        reference_audio: str,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """Generate speech with automatic Romanian preprocessing."""
        pass
```

**Key features:**
- Automatic Romanian preprocessing (ș→sh, ț→ts)
- Strips `t3.` prefix from checkpoint keys
- API matches upstream ChatterboxTTS

### 2. Experiment Runner

**File:** `scripts/run_experiment.py`

**CLI:**
```bash
python scripts/run_experiment.py \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output \
    --holdout-csv data/processed/MyTTSDataset/splits/test_holdout.csv \
    --swara-dir /path/to/SWARA1.0_22k_noSil \
    --output-dir experiments/2026-02-22/ \
    --samples-per-condition 50 \
    --seed 42
```

**Output structure:**
```
experiments/2026-02-22/
├── bas_exag0.3_cfg0.3/
│   ├── bas_rnd1_001.wav
│   └── ...
├── bas_exag0.3_cfg0.5/
├── bas_exag0.7_cfg0.3/
├── bas_exag0.7_cfg0.5/
├── sgs_exag0.3_cfg0.3/
├── sgs_exag0.3_cfg0.5/
├── sgs_exag0.7_cfg0.3/
├── sgs_exag0.7_cfg0.5/
├── filelists/
│   ├── bas_exag0.3_cfg0.3.csv
│   └── ...
└── experiment_config.json
```

**Filelist format** (pipeline-compatible):
```csv
base_file|transcript_original
bas_rnd1_001.wav|Printre ei chiar campioana Estera Dobre.
```

### 3. Metrics Pipeline Integration

**Wrapper:** `scripts/run_metrics.sh`

```bash
#!/bin/bash
EXPERIMENT_DIR=$1
PIPELINE_DIR=/home/astanea/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline
SWARA_DIR=/home/astanea/data/SWARA1.0_22k_noSil

for filelist in $EXPERIMENT_DIR/filelists/*.csv; do
    condition=$(basename "$filelist" .csv)

    # Step 1: Transcribe synthesized audio
    python $PIPELINE_DIR/transcription.py \
        --input_filelist "$filelist" \
        --base_dir_synthesized "$EXPERIMENT_DIR/$condition" \
        --generator_model "chatterbox-ro-$condition"

    # Step 2: Compute metrics
    python $PIPELINE_DIR/metrics.py \
        --input_filelist "$EXPERIMENT_DIR/transcriptions/$condition.csv" \
        --base_dir_synthesized "$EXPERIMENT_DIR/$condition" \
        --base_dir_original "$SWARA_DIR" \
        --output_csv "$EXPERIMENT_DIR/metrics/$condition.csv"
done

# Step 3: Aggregate results
python scripts/aggregate_metrics.py --experiment-dir "$EXPERIMENT_DIR"
```

**Metrics computed:**
- CER (Character Error Rate)
- WER (Word Error Rate)
- UTMOS (speech quality)
- MCD (Mel Cepstral Distortion)
- Log F0 RMSE (pitch accuracy)
- Speaker Cosine Similarity

### 4. Metrics Aggregation

**File:** `scripts/aggregate_metrics.py`

**Output:** `experiments/2026-02-22/results_summary.csv`

```csv
condition,cer_mean,cer_std,wer_mean,wer_std,utmos_mean,utmos_std,...
bas_exag0.3_cfg0.3,0.12,0.05,0.18,0.08,3.45,0.32,...
bas_exag0.3_cfg0.5,0.11,0.04,0.16,0.07,3.52,0.28,...
```

### 5. GitHub Pages Gallery

**File:** `scripts/build_gallery.py`

**Output structure:**
```
docs/
├── index.html
├── bas_exag0.3_cfg0.3.html
├── bas_exag0.3_cfg0.5.html
├── ...
├── results.html
├── audio/
│   └── [condition]/[samples].wav
└── style.css
```

**Design principles:**
- Clean, academic appearance
- Minimal CSS (system fonts, muted colors)
- Clear typography and spacing
- No JavaScript frameworks
- Native HTML5 audio players

**Index page:**
- Experiment overview
- Parameter descriptions
- Links to all conditions
- Link to results summary

**Condition pages:**
- Header: Speaker + parameters
- List of 50 samples with text and audio player
- Navigation to other conditions

**Results page:**
- Summary metrics table
- Mean ± std for each metric per condition

## Data Flow

```
test_holdout.csv (2484 samples)
        │
        ▼
┌───────────────────┐
│ Select 50 BAS +   │
│ 50 SGS samples    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ run_experiment.py │──────► 400 WAV files (8 directories)
│                   │──────► 8 filelists for metrics
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ run_metrics.sh    │──────► Transcriptions (Whisper)
│ (pipeline)        │──────► Per-sample metrics CSVs
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ aggregate_metrics │──────► results_summary.csv
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ build_gallery.py  │──────► docs/ (GitHub Pages)
└───────────────────┘
```

## Reference Audio Selection

For each speaker, select one sample as reference audio:
- BAS: `bas_rnd1_011.wav` (already validated, ~4s)
- SGS: `sgs_rnd1_011.wav` (to be validated, similar position)

Reference audio should be 3-10 seconds for optimal voice cloning.

## File Summary

| File | Type | Purpose |
|------|------|---------|
| `src/chatterbox_ro.py` | New | HF-compatible Romanian TTS class |
| `scripts/run_experiment.py` | New | Generate experiment samples |
| `scripts/aggregate_metrics.py` | New | Summarize metrics per condition |
| `scripts/run_metrics.sh` | New | Wrapper for evaluation pipeline |
| `scripts/build_gallery.py` | New | Generate GitHub Pages site |
| `docs/style.css` | New | Minimal academic CSS |

## Future Work

- **HuggingFace release:** Upload model weights to Hub, enable `from_pretrained()`
- **Extended parameter grid:** Test more exaggeration/CFG values
- **Additional metrics:** MOS listening study
- **Multi-language comparison:** Compare with other Romanian TTS systems
