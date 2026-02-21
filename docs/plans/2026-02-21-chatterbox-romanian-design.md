# Chatterbox Romanian Adaptation -- Design Document

**Date:** 2026-02-21
**Goal:** Fine-tune Chatterbox Multilingual for Romanian using the SWARA dataset, with a focus on academic research quality.

---

## 1. Architecture & Model Choice

### Base Model
- **Model:** `ResembleAI/chatterbox` Multilingual (500M parameter Llama 3 backbone)
- **Why Multilingual, not Turbo:** The Turbo variant (GPT-2 based, 350M) has reported failures when fine-tuned on new languages (hallucinated audio, posterior collapse -- GitHub Issues #6, #12 in the community kit). The Multilingual model already handles 23 languages with cross-lingual transfer, making it the safer and more principled starting point.
- **Romanian is not in the 23 supported languages**, so fine-tuning is required.

### Trainable Components
- **T3 (text-to-speech transformer):** Trainable -- this is the main language model that maps text tokens to speech tokens.
- **Voice Encoder (VE):** Frozen -- extracts speaker embeddings from reference audio.
- **S3Gen vocoder (HiFT-GAN derived):** Frozen -- converts speech tokens to waveforms.

### Tokenizer
- Grapheme-level character tokenizer, ~2,454 tokens covering 23 languages.
- Romanian diacritics (ă, â, î, ș, ț, Ă, Â, Î, Ș, Ț) must be verified in the existing tokenizer.
- Many are likely present (shared Latin script with Italian, French, Spanish, Polish).
- If any are missing: extend `tokenizer.json`, update `NEW_VOCAB_SIZE` to match exact token count.

### Speech Tokens
- Extracted by S3Tokenizer; stop token ID 6562.
- Max speech length: 850 tokens; max text length: 256 tokens.

---

## 2. Data Preparation (SWARA -> LJSpeech)

### Source Dataset
- **SWARA:** 21+ hours, 17 speakers, 19,279 utterances, Romanian read speech.
- **Audio:** WAV, 22kHz (based on path naming `SWARA1.0_22k`; verify from actual files).
- **Metadata format:** `path|text|speaker_id` (pipe-delimited CSV).
- **License:** CC BY-NC 4.0, research use, requires signed license agreement.

### Conversion Pipeline
**Principle: Never alter the original dataset.** All outputs go to `data/processed/`.

```
data/processed/
  MyTTSDataset/
    metadata.csv      # filename|raw_text|normalized_text
    wavs/
      bas_rnd1_001.wav
      ...
```

### Text Normalization
- Preserve all Romanian diacritics (ă, â, î, ș, ț).
- SWARA transcripts appear to already have numbers expanded to words (confirmed from samples).
- Expand remaining abbreviations if found (str. -> strada, nr. -> numărul, etc.).
- Normalize punctuation (consistent quote marks, dashes).
- Keep casing as-is.

### Audio Processing
- The fine-tuning kit resamples to 16kHz internally -- no manual resampling needed.
- Run a duration analysis on all utterances before any filtering.
- Exclude only extreme outliers (<0.5s or >30s) where audio quality is suspect.
- No truncation that would desync audio-text pairs.

### Data Split Strategy
- **Hold-out speakers (test only):** BAS and SGS -- all their utterances excluded from training/validation, reserved for zero-shot voice cloning evaluation.
- **Remaining 15 speakers:** 90/10 per-speaker stratified split for train/val.

---

## 3. Environment & Infrastructure

### Container Stack
```
Base image: nvcr.io/nvidia/pytorch:26.01-py3
  (CUDA 13.1, PyTorch 2.10, Python 3.12)

+ chatterbox-tts (pip install)
+ chatterbox-finetuning (git clone into workspace)
+ espeak-ng (apt-get install)
+ evaluation dependencies
```

### Docker Compose + Devcontainer
- `docker-compose.yml` defines the training service with GPU access.
- `.devcontainer/devcontainer.json` references the compose file for IDE integration (VS Code / Cursor).
- Dataset and output paths injected via environment variables:
  - `SWARA_PATH` -> mounted read-only
  - `OUTPUT_PATH` -> processed data + checkpoints
- Same compose file works locally (subset of SWARA) and on DGX (full dataset + all GPUs).

### DGX Specifics
- GPUs: V100 32GB (multiple available).
- Launch: `docker compose run --gpus all train` or `torchrun --nproc_per_node=N train.py`.
- V100 constraint: **fp16 only** (no bf16 support).

---

## 4. Training Configuration

### Hardware: V100 32GB

| Parameter | Value |
|---|---|
| Precision | `fp16` (V100 constraint) |
| Batch size per GPU | 4 (adjust based on actual VRAM usage) |
| Gradient accumulation | Flexible: `effective_batch / (batch_per_gpu * n_gpus)` |
| Target effective batch | 32 |
| Learning rate | 1e-5 |
| Epochs/steps | 120 epochs or 50k-150k steps (monitor convergence) |
| Checkpoints | Every 500 steps, keep last 5 |
| Gradient checkpointing | Enabled (~60% VRAM reduction) |
| Optimizer | AdamW (HuggingFace Trainer default) |

### Multi-GPU Scaling
- Effective batch size stays constant at 32.
- Gradient accumulation adjusts automatically.
- Exact configuration deferred until we inspect what the training scripts support.

### Monitoring
- Training loss curve (smooth decrease, no NaN).
- Periodic sample audio generation for fixed Romanian prompts.
- TensorBoard or Weights & Biases logging.

---

## 5. Evaluation Framework

### Integration with Existing Evaluation Infrastructure
The project will build upon an existing evaluation script infrastructure that expects directories of generated speech samples and computes multiple metrics. This will be adapted as needed for the Romanian evaluation.

### Metrics

| Category | Metric | Description |
|---|---|---|
| Intelligibility | WER | Word Error Rate via Romanian ASR (gigant/romanian-wav2vec2 or Whisper large-v3) |
| Intelligibility | CER | Character Error Rate |
| Spectral | MCD | Mel Cepstral Distortion (synthesized vs reference) |
| Spectral | RMSE | Root Mean Square Error on spectral features |
| Perceptual | PESQ | Perceptual Evaluation of Speech Quality |
| Perceptual | STOI | Short-Time Objective Intelligibility |
| Signal | SI-SDR | Scale-Invariant Signal-to-Distortion Ratio |
| Speaker | SECS | Speaker Embedding Cosine Similarity |
| Subjective | MOS | Mean Opinion Score (native Romanian listeners, 1-5 scale) |

### Evaluation Splits
1. **In-speaker (15 training speakers):** Synthesize val set utterances using reference clips from the same speaker. Measures quality on known speakers.
2. **Out-of-speaker (BAS, SGS):** Synthesize test utterances using BAS/SGS reference clips. Tests zero-shot voice cloning for unseen Romanian speakers.

### Diacritic Stress Test
A special test set with Romanian minimal pairs (e.g., "paturi" vs "pături") to verify diacritic-sensitive pronunciation.

---

## 6. Inference Integration & Release

### Chatterbox Inference Pipeline Integration
- The fine-tuned T3 checkpoint (`t3_finetuned.safetensors`) must integrate cleanly with the official Chatterbox inference code.
- The extended tokenizer (if diacritics were added) must be bundled with the model.
- Goal: `chatterbox.synthesize(text="Bună ziua!", ref_audio="ref.wav")` works out of the box for Romanian.

### HuggingFace Release
- Package the fine-tuned model with model card, tokenizer, and usage examples.
- Include evaluation results and dataset attribution (SWARA, CC BY-NC 4.0).
- Consider upstream contribution to the official Chatterbox repo (Romanian language support).

---

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Romanian diacritics missing from tokenizer | Low-Medium | Verify early; extend if needed |
| 21 hours insufficient for quality | Medium | Clean normalization, careful splits; SWARA is read speech (high quality) |
| V100 fp16 instability | Low | Use loss scaling; monitor for NaN |
| Community kit bugs or limitations | Medium | Fork and patch as needed; inspect code before running |
| Multi-GPU not supported by training scripts | Medium | Defer to `torchrun`/`accelerate` integration or single-GPU fallback |

---

## 8. Non-Goals (Explicit Exclusions)
- No Turbo model training (deferred due to known issues).
- No data augmentation or additional datasets beyond SWARA.
- No production deployment optimization (academic research focus).
- No real-time streaming inference.
