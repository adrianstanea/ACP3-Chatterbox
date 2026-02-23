# Romanian Chatterbox Adaptation Project

**Project Start Date:** February 2026
**Current Status:** Data preparation phase
**Repository:** https://github.com/adrianstanea/ACP3-Chatterbox
**Branch:** romanian-adaptation (worktree)

## Project Goal

Adapt the Chatterbox text-to-speech (TTS) model to Romanian by fine-tuning the multilingual variant on the SWARA 1.0 dataset. This project focuses on academic research quality, reproducibility, and systematic evaluation.

## Executive Summary

This project aims to create the first open-source, neural TTS model for Romanian based on the state-of-the-art Chatterbox architecture. By fine-tuning the multilingual Chatterbox model (500M parameter Llama 3 backbone) on the SWARA dataset (21+ hours of high-quality read speech), we will develop a model capable of:

- High-quality Romanian speech synthesis
- Zero-shot voice cloning for unseen Romanian speakers
- Proper pronunciation of Romanian diacritics (ă, â, î, ș, ț)
- Natural prosody and intonation

### Key Innovation: Phoneme-Level Preprocessing

An initial experiment extending the tokenizer vocabulary with new Romanian characters resulted in **posterior collapse** — the model produced unintelligible speech (see [Tokenization Experiments](TOKENIZATION-EXPERIMENTS.md)). This failure was independently corroborated by upstream community reports for Turkish, Norwegian, German, and Arabic.

Our current approach uses **phoneme-level text preprocessing** instead of vocabulary extension:
- Characters missing from the vocabulary (ș, ț) are mapped to phonetically equivalent existing tokens (`sh`, `ts`)
- Characters already in the vocabulary (ă, â, î) are kept unchanged
- This preserves pretrained embedding quality and avoids the weak-signal posterior collapse

## Architecture Overview

### Base Model: Chatterbox Multilingual

**Model Details:**
- Architecture: Llama 3-based transformer (500M parameters)
- Original languages: 23 (excluding Romanian)
- Tokenizer: Character-level grapheme tokenizer (2,454 tokens)
- Components:
  - **T3 (Text-to-Speech Transformer):** Trainable - maps text to speech tokens
  - **Voice Encoder (VE):** Frozen - extracts speaker embeddings
  - **S3Gen Vocoder:** Frozen - converts speech tokens to audio

**Why Multilingual (not Turbo)?**

The Chatterbox Turbo variant (GPT-2 based, 350M parameters) has documented failures when fine-tuned on new languages:
- Hallucinated audio artifacts (GitHub Issue #6)
- Posterior collapse during training (GitHub Issue #12)
- BPE tokenizer optimized for English makes adaptation difficult

The Multilingual model already handles 23 languages with cross-lingual transfer, making it the more stable and principled choice for Romanian adaptation.

### Training Strategy

**Trainable vs Frozen:**
- T3 Transformer: **Trainable** (embedding layer + prediction head + backbone)
- Voice Encoder: **Frozen** (pretrained speaker extraction)
- S3Gen Vocoder: **Frozen** (pretrained speech generation)

**Key Innovation:**
- Phoneme-level text preprocessing maps missing Romanian characters to existing vocabulary tokens (ș→"sh", ț→"ts")
- Original vocabulary size (2,454) preserved — no extension needed
- Avoids posterior collapse caused by weak mean-initialized embeddings
- See [Tokenization Experiments](TOKENIZATION-EXPERIMENTS.md) and [Technical Decision #7](technical-decisions.md#decision-7-phoneme-mapping-over-vocabulary-extension)

## Dataset

### SWARA 1.0 (Romanian Speech Corpus)

**Official Statistics:**
- Total speakers: 18
- Total utterances: 21,304
- Total duration: 21.67 hours
- Sample rate: 22 kHz
- Format: WAV (mono)
- License: CC BY-NC 4.0

**Our Verified Statistics (from analysis):**
- Successfully analyzed files: 21,304
- Sample rate: 22,050 Hz (consistent)
- All Romanian diacritics present
- Quality: Read speech, high intelligibility

**Holdout Strategy:**
- BAS and SGS speakers: Reserved for zero-shot evaluation
- Remaining 16 speakers: 90/10 train/validation split (per-speaker stratified)

For detailed statistics, see [SWARA Analysis Report](data/swara-analysis-report.md).

## Technology Stack

### Container Infrastructure
- **Base Image:** `nvcr.io/nvidia/pytorch:26.01-py3`
  - CUDA 13.1
  - PyTorch 2.10
  - Python 3.12
- **Orchestration:** Docker Compose
- **Development:** VS Code devcontainer integration

### Training Environment
- **Hardware Target:** NVIDIA V100 32GB
- **Precision:** FP16 (V100 constraint, no bf16 support)
- **Framework:** HuggingFace Trainer
- **Monitoring:** TensorBoard

### Fine-Tuning Infrastructure
- **Repository:** [chatterbox-finetuning](https://github.com/adrianstanea/chatterbox-finetuning) (forked)
- **Integration:** Git submodule in `vendor/chatterbox-finetuning/`
- **Pinned Commit:** `18ffb2d` (Voice Conditioning Dropout feature)

## Project Timeline

### Phase 1: Infrastructure Setup (Completed)
- [x] Task 1: Docker Compose and devcontainer configuration
- [x] Task 2: Community fine-tuning kit integration
- [x] Task 3: SWARA dataset analysis
- [x] Task 4: SWARA to LJSpeech format conversion

### Phase 2: Training Preparation (In Progress)
- [ ] Task 5: Verify and extend tokenizer for Romanian diacritics
- [ ] Task 6: Run preprocessing pipeline (speech token extraction)

### Phase 3: Training and Evaluation
- [ ] Task 7: Training run (150 epochs, ~50k steps)
- [ ] Task 8: Fine-tuned model integration
- [ ] Task 9: Generate evaluation samples
- [ ] Task 10: Compute evaluation metrics

### Phase 4: Release
- [ ] Task 11: Package model for HuggingFace release

## Key Design Decisions

1. **Multilingual over Turbo:** Stability and proven cross-lingual transfer
2. **PyTorch 26.01:** Latest CUDA support, Python 3.12 compatibility
3. **Git Submodule for vendor code:** Clean separation, version tracking
4. **LJSpeech format:** Industry standard, tool compatibility
5. **Chatterbox punc_norm:** Use official text normalization
6. **Holdout speakers:** Enable zero-shot voice cloning evaluation

For detailed rationale, see [Technical Decisions Log](technical-decisions.md).

## Evaluation Framework

### Metrics

**Intelligibility:**
- WER (Word Error Rate) via Romanian ASR
- CER (Character Error Rate)

**Quality:**
- MCD (Mel Cepstral Distortion)
- PESQ (Perceptual Evaluation)
- STOI (Short-Time Objective Intelligibility)

**Speaker Similarity:**
- SECS (Speaker Embedding Cosine Similarity)

**Subjective:**
- MOS (Mean Opinion Score) with native Romanian listeners

### Evaluation Splits
1. In-speaker validation: Known speakers from training set
2. Zero-shot: BAS and SGS holdout speakers
3. Diacritic stress test: Minimal pairs (e.g., "paturi" vs "pături")

## Repository Structure

```
.
├── .devcontainer/           # VS Code devcontainer config
├── data/
│   ├── processed/           # LJSpeech format conversion output
│   └── metadata_SWARA1.0_text.csv  # Original metadata
├── docs/
│   ├── codex/               # AI assistant knowledge base
│   ├── data/                # Dataset documentation
│   ├── decisions/           # Technical decisions
│   ├── gemini/              # Research notes
│   ├── plans/               # Implementation plans
│   ├── setup/               # Environment setup docs
│   └── vendor/              # Vendor code analysis
├── scripts/
│   ├── analyze_swara.py     # Dataset analysis
│   └── convert_swara_to_ljspeech.py  # Format conversion
├── vendor/
│   └── chatterbox-finetuning/  # Git submodule
├── docker-compose.yml       # Container orchestration
├── .env.example             # Environment variables template
└── requirements.txt         # Python dependencies
```

## Environment Variables

Key environment variables (see `.env.example`):

```bash
# Dataset paths
SWARA_PATH=/path/to/SWARA1.0_22k_noSil  # Read-only mount
OUTPUT_PATH=/path/to/output             # Processed data + checkpoints

# GPU configuration
NUM_GPUS=1                              # Number of GPUs to use

# Training configuration
BATCH_SIZE=4                            # Per-GPU batch size
GRAD_ACCUM=8                            # Gradient accumulation steps
LEARNING_RATE=1e-5                      # Learning rate
```

## Reproducibility

All work is tracked and documented:
- Git commits for code changes
- Design documents for major decisions
- Analysis scripts for dataset verification
- Training logs for hyperparameter tracking
- Evaluation reports for quality assessment

## References

### Model Architecture
- Chatterbox: https://github.com/ResembleAI/chatterbox
- Fine-tuning kit: https://github.com/gokhaneraslan/chatterbox-finetuning

### Dataset
- SWARA 1.0: https://github.com/crosslingual-voices/SWARA1.0
- License: CC BY-NC 4.0
- Citation: Required for academic use

### Related Documentation
- [Design Document](plans/2026-02-21-chatterbox-romanian-design.md)
- [Fine-tuning Kit Analysis](vendor/chatterbox-finetuning-analysis.md)
- [SWARA Analysis Report](data/swara-analysis-report.md)

## Contact

**Project Lead:** Adrian Stanea
**Institution:** ACP3 (Academic Project)
**Purpose:** Research and education

## License

This adaptation work is licensed under the same terms as the original Chatterbox model. The SWARA dataset is used under CC BY-NC 4.0 for academic research purposes.
