# Executive Summary: Chatterbox TTS Romanian Adaptation

## Goal
Adapt and compare Chatterbox **Standard** and **Turbo** for Romanian using the **SWARA** multi-speaker dataset, with a **shared tokenizer** and the **community fine-tuning kit** as the initial training backbone.

## Key Decisions
- **Multi-speaker training**: Aligns with SWARA’s 17 speakers and the pipeline’s speaker-embedding conditioning.
- **Shared tokenizer**: Ensures a controlled, fair comparison between Standard and Turbo.
- **Community kit first**: Faster path to a working baseline; fork or patch only if needed.

## Baseline Plan (Single GPU, 32GB VRAM)
- Mixed precision: `bf16` preferred, `fp16` fallback.
- Batch size: `4` (drop to `2` if OOM).
- Gradient accumulation: `4` (effective batch size 16).
- Steps: 50k–150k depending on convergence.

## Critical Requirements
- Ensure Romanian diacritics are present in the tokenizer (`ă â î ș ț`, upper/lower).
- `NEW_VOCAB_SIZE` must match the tokenizer token count.
- Convert SWARA to LJSpeech format with normalized Romanian text.

## Evaluation (Comparative Study)
- **Objective**: WER (Romanian ASR), optional F0 correlation.
- **Subjective**: MOS + MUSHRA with native Romanian listeners.
- Use in-speaker and out-of-speaker reference clips for generalization tests.

## Dependencies
- Official Chatterbox repo (base models, inference).
- Community fine-tuning kit (training pipeline).
- SWARA dataset (research license required).

## Open Items
- Local copy of the fine-tuning kit repo is needed here to implement changes.
- Confirm GPU count for final scaling configs.
- Provide local SWARA dataset path for conversion and preprocessing.
