# Chatterbox Romanian Adaptation Plan

## Current Research State (Summary)
- Official Chatterbox repo, community fine-tuning kit, and SWARA dataset identified.
- Romanian is not supported out-of-the-box; tokenizer extension and fine-tuning required.
- Pipeline requires LJSpeech-format data and preprocessing to generate `.pt` artifacts.
- Decision: multi-speaker training and shared tokenizer for fair Turbo vs Standard comparison.
- Environment: DGX with 32GB VRAM GPUs, 1 GPU baseline, scalable.

## Implementation Plan

### 1. Requirements
Hard requirements
- Local copy of the community fine-tuning kit in this workspace.
- Local copy of SWARA dataset (or validated subset) with transcripts.
- Romanian-capable tokenizer (`ă â î ș ț`, upper/lower).
- Devcontainer support for local and DGX usage.

Soft requirements
- Romanian ASR for WER evaluation.
- MOS/MUSHRA tooling and native listeners.

### 2. Repository Setup
Tasks
- Import the fine-tuning kit repo locally.
- Add devcontainer config (see docs/codex/devcontainers_requirements.md).
- Add minimal dataset path config via env vars or config file.

Validation
- Repo runs inside devcontainer.
- Dataset path injection works in training scripts.

### 3. Dataset Preparation (SWARA → LJSpeech)
Tasks
- Convert SWARA to LJSpeech format:
  - `MyTTSDataset/wavs/*.wav`
  - `MyTTSDataset/metadata.csv` using `filename|raw_text|normalized_text`
- Normalize Romanian text without stripping diacritics.

Validation
- All metadata entries map to files.
- Romanian diacritics preserved.
- Clip lengths roughly 3–10 seconds.
- Speaker-stratified train/val/test splits.

### 4. Tokenizer (Shared)
Tasks
- Validate tokenizer coverage for Romanian diacritics.
- Extend tokenizer if needed; generate `tokenizer.json`.
- Update `NEW_VOCAB_SIZE` to match token count.

Validation
- Tokenize all `normalized_text` without unknowns.
- `NEW_VOCAB_SIZE` equals tokenizer token count.

### 5. Preprocessing
Tasks
- Run preprocessing to generate speaker embeddings and acoustic tokens.

Validation
- `.pt` artifacts exist for all entries.
- Random sample of 10 items confirms complete artifacts.

### 6. Training (Turbo + Standard)
Baseline hyperparameters (1 GPU, 32GB)
- Precision: `bf16` preferred, `fp16` fallback.
- Batch size: 4 (drop to 2 if OOM).
- Grad accumulation: 4 (effective batch 16).
- Steps: 50k–150k depending on convergence.
- Checkpoints: every 1k–2k steps.

Tasks
- Train Turbo model.
- Train Standard model with identical data/tokenizer/hyperparams.

Validation
- Loss curves stable; no NaNs.
- Periodic audio samples for fixed Romanian prompts.

### 7. Evaluation
Objective
- WER (Romanian ASR).
- Optional F0 correlation / pitch RMSE.

Subjective
- MOS (naturalness, speaker similarity).
- MUSHRA (Turbo vs Standard).

Splits
- In-speaker: reference from same speaker.
- Out-of-speaker: reference from different speaker.

Validation
- WER pipeline runs on held-out set.
- MOS/MUSHRA templates prepared and reviewed.

## Blocking Inputs
- Location of local fine-tuning kit repo.
- Location of SWARA dataset.
- GPU count for final scaling configs.
