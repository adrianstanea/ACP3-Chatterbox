# TTS Researcher Details (Consolidated)

This document captures the research details produced by the TTS Researcher workflow so far, focusing on Chatterbox TTS adaptation to Romanian with SWARA.

## Model and Repository Findings
- Official Chatterbox repository provides installation, inference usage, and language list; Romanian is not in the supported language list, so adaptation is required.
- Official PyPI package is available for Chatterbox (`chatterbox-tts`).
- Community fine-tuning kit exists and targets Chatterbox Standard/Turbo fine-tuning; it includes preprocessing, tokenizer handling, dataset format, and training workflow.

## Dataset Findings (SWARA)
- SWARA is a Romanian read speech dataset with 17 speakers, 19,279 utterances, and 21+ hours of audio.
- Transcripts are provided.
- Research license agreement is required; CC BY-NC 4.0.

## Architecture and Adaptation Requirements
- Romanian is not supported out-of-the-box, so a tokenizer/vocabulary update and fine-tuning are necessary.
- Shared tokenizer is preferred for controlled comparisons between Standard and Turbo.
- Tokenizer must include Romanian diacritics: `ă â î ș ț` (upper/lower).
- `NEW_VOCAB_SIZE` must match the tokenizer token count exactly.

## Data Preparation Requirements
- LJSpeech-style dataset format with `metadata.csv` and `wavs/`.
- `metadata.csv` format: `filename|raw_text|normalized_text`.
- Audio is resampled to 16 kHz by the preprocessing pipeline; vocoder output is 24 kHz.
- Recommended clip length: 3–10 seconds.

## Preprocessing Requirements
- Preprocessing is mandatory; it generates speaker embeddings and acoustic tokens saved as `.pt` artifacts for training.

## Turbo vs Standard (Comparative Study)
- Run both models with identical data, tokenizer, preprocessing, and hyperparameters to isolate architectural differences.
- Turbo may converge faster and handle smaller data better; Standard may offer higher expressiveness (inference based on model size and intent).

## Training Baseline (1 GPU, 32GB VRAM)
- Mixed precision: `bf16` preferred, `fp16` fallback.
- Batch size: 4 (drop to 2 if OOM).
- Gradient accumulation: 4 (effective batch size 16).
- Steps: 50k–150k depending on convergence.
- Checkpoints every 1k–2k steps.

## Evaluation Plan
Objective
- WER with Romanian ASR on held-out set.
- Optional F0 correlation / pitch RMSE.

Subjective
- MOS for naturalness and speaker similarity.
- MUSHRA for direct Turbo vs Standard comparisons.

Splits
- In-speaker: reference from same speaker.
- Out-of-speaker: reference from different speaker.

## Risks and Mitigations
- Romanian not supported natively: tokenizer extension required.
- Dataset size moderate: enforce consistent normalization and clean splits; consider augmentation later.
- License constraints: SWARA is research-only and CC BY-NC 4.0.

## Environment Notes
- DGX available with 32GB VRAM GPUs; 1 GPU baseline, scalable to more.
- Network access in this environment is restricted; GitHub cloning requires local copy or enabled network.
