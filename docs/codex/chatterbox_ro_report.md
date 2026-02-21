# Chatterbox TTS Romanian Adaptation Report (Status)

## Scope
Adapt and compare Chatterbox Standard and Turbo for Romanian, using the SWARA multi-speaker dataset, with a shared tokenizer and the community fine-tuning kit as the training backbone.

## Key Decisions
- **Multi-speaker training**: Use multi-speaker because SWARA is multi-speaker and the pipeline extracts speaker embeddings, which aligns with reference-audio conditioning for voice cloning.
- **Shared tokenizer**: One tokenizer for both Standard and Turbo to keep comparisons controlled.
- **Community fine-tuning kit first**: Start with the community kit; fork/modify only if necessary.

## Constraints / Environment
- DGX system available.
- Baseline: 1 GPU assumed; 32GB VRAM each (scales to more).
- Network access from this environment to GitHub is currently blocked, so repo cloning requires local copy or enabling network.

## Sources Identified (For the project, not fetched here)
- Official Chatterbox repo (installation, inference, supported languages list).
- Official PyPI package for versioning.
- Community fine-tuning kit (training pipeline, tokenizer requirements, preprocessing, dataset format).
- SWARA dataset (Romanian, 17 speakers, 19,279 utterances, 21+ hours, transcripts, research license).

## Language Gap
- Romanian is not in the official supported language list for the multilingual model. Adaptation is required via tokenizer coverage and fine-tuning.

## Data Requirements and Preparation
- **Format**: LJSpeech-style dataset with `metadata.csv` and `wavs/`.
  - `metadata.csv` lines: `filename|raw_text|normalized_text`
- **Audio**: WAV; 3–10 seconds per clip recommended. Preprocessing resamples to 16 kHz.
- **Text normalization**:
  - Preserve Romanian diacritics: `ă â î ș ț` (both cases).
  - Keep punctuation and casing stable between raw and normalized fields.

## Tokenizer Requirements
- Shared tokenizer must cover Romanian diacritics.
- If missing, extend tokenizer and update `NEW_VOCAB_SIZE` to exactly match token count.
- Use the same tokenizer for Standard and Turbo runs.

## Training Workflow (Planned)
1. Install official Chatterbox (base models, inference sanity checks).
2. Clone or import community fine-tuning kit.
3. Convert SWARA to LJSpeech format.
4. Build/validate shared tokenizer; set `NEW_VOCAB_SIZE`.
5. Run preprocessing to create `.pt` artifacts (speaker embeddings + acoustic tokens).
6. Train Turbo and Standard separately using identical data, tokenizer, and hyperparameters.
7. Evaluate with identical prompts and held-out splits.

## Baseline Hyperparameters (1 GPU, 32GB VRAM)
- Mixed precision: `bf16` preferred, `fp16` fallback.
- Batch size: start at `4` (drop to `2` if OOM).
- Grad accumulation: `4` (effective batch size 16).
- Steps: 50k–150k depending on convergence.
- Checkpoint every 1k–2k steps.

## Scaling to Multiple GPUs
- Keep effective batch size constant across Turbo/Standard.
- For N GPUs: either increase batch size linearly or reduce grad accumulation.

## Evaluation Plan
**Objective**
- WER (Romanian ASR) on held-out set.
- F0 correlation / pitch RMSE (optional).

**Subjective**
- MOS (naturalness, speaker similarity).
- MUSHRA (Turbo vs Standard direct comparisons).

**Splits**
- In-speaker: reference clip from same speaker.
- Out-of-speaker: reference from different speaker.

## Risks and Mitigations
- Romanian not in supported language list: must verify/extend tokenizer.
- Dataset size modest: enforce clean normalization and careful splits; consider augmentation or extra data later.
- License: SWARA research license and CC BY-NC 4.0 must be respected.

## Next Steps (Pending)
- Provide or import the fine-tuning kit repo locally to enable code-level changes.
- Confirm GPU count for final configs (1, 2, 4, 8 variants).
- Provide SWARA dataset path for conversion and preprocessing.
