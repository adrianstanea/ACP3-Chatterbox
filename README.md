# Romanian Chatterbox Fine-Tuning Project

Fine-tuning Chatterbox Multilingual (500M Llama 3) for Romanian using the SWARA 1.0 dataset.

## Project Status

**Current**: ✅ Task 6 Complete - Ready for Task 7 (Training on DGX)

### Completed Tasks (6/11)

1. ✅ **Docker & Devcontainer Setup**
   - Clean base image with devcontainer architecture
   - Python 3.12 compatible
   - Portable across local dev and DGX

2. ✅ **Vendor Integration**
   - Chatterbox fine-tuning kit as git submodule
   - Analysis completed: Use Standard mode (not Turbo)

3. ✅ **Dataset Analysis**
   - SWARA 1.0: 21,304 files, 18 speakers, 21.67 hours
   - All Romanian diacritics present

4. ✅ **Dataset Conversion**
   - Converted to LJSpeech format
   - Splits: train (18,740), validation (80), holdout (2 speakers)

5. ✅ **Tokenizer Verification & Extension**
   - Implemented generic language-agnostic extension system
   - Extended for Romanian: 2454 → 2459 tokens
   - 100% character coverage (10/10 Romanian chars)
   - **Ready for upstreaming**

6. ✅ **Preprocessing**
   - All 21,304 files processed successfully
   - Output: 157 MB preprocessed .pt files
   - Extended tokenizer (2459 tokens) working correctly
   - **Ready for training**

### Next Steps

7. ⏳ **Training** - Fine-tune on DGX (days)
8. ⏳ **Inference Testing**
9. ⏳ **Evaluation** (WER/MOS)
10. ⏳ **Documentation**
11. ⏳ **Release**

## Key Innovation: Generic Language Extension System

We built an **upstreamable, language-agnostic tokenizer extension system** that benefits the entire Chatterbox community.

### Features
- 🌍 **Language-agnostic**: Works for any character set
- 🔒 **Safe**: Automatic backups before modifications
- ✅ **Validated**: Built-in verification
- 🔄 **Idempotent**: Can run multiple times safely
- 📚 **Documented**: Comprehensive guides

### Usage
```bash
# Extend for Romanian
python scripts/extend_tokenizer.py --language romanian

# Verify extension
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/MyTTSDataset/metadata.csv
```

### Results
- **Before**: 5/10 characters (50% coverage)
- **After**: 10/10 characters (100% coverage)
- **Impact**: ~2% of dataset now properly encoded

## Quick Start

### Prerequisites
- Docker & docker-compose
- NVIDIA GPU with CUDA support
- SWARA 1.0 dataset

### Setup
```bash
# Navigate to workspace
cd /path/to/ACP3-Chatterbox/.worktrees/romanian-adaptation

# Configure environment
cp .env.example .env
# Edit .env with your paths

# Build and start container
docker compose build
docker compose up -d

# Run setup (inside container)
docker compose exec chatterbox bash .devcontainer/post-create.sh

# Enter container
docker compose exec chatterbox bash
```

## Training Guide (DGX Deployment)

### Step 1: Clone and Setup on DGX

```bash
# Clone repository
git clone https://github.com/adrianstanea/ACP3-Chatterbox.git
cd ACP3-Chatterbox

# Switch to Romanian adaptation branch
git checkout feature/romanian-adaptation

# Initialize vendor submodule
git submodule update --init --recursive

# Configure environment
cp .env.example .env
nano .env  # Edit paths for DGX:
# SWARA_PATH=/path/to/swara/dataset
# OUTPUT_PATH=/path/to/output
# NUM_GPUS=4  # Adjust based on available GPUs
```

### Step 2: Build Docker Container

```bash
# Build container (PyTorch 24.11-py3 base)
docker compose build

# Start container
docker compose up -d

# Run post-create setup (automated environment setup)
docker compose exec chatterbox bash .devcontainer/post-create.sh
```

**What post-create does**:
- Installs setuptools 69.5.1 (Perth watermarker dependency)
- Installs vendor dependencies (PyTorch 2.6.0, transformers, etc.)
- Removes flash-attention (ABI compatibility fix)
- Installs workspace dependencies (jiwer, pesq, pystoi)
- Downloads pretrained models (~3GB, 5-10 minutes)
- Fixes dataset symlinks for container paths

### Step 3: Verify Preprocessing

```bash
# Enter container
docker compose exec chatterbox bash

# Verify preprocessed files
ls /workspace/data/processed/MyTTSDataset/preprocess/*.pt | wc -l
# Should output: 21304

# Check preprocessing size
du -sh /workspace/data/processed/MyTTSDataset/preprocess/
# Should be ~157M

# Test tokenizer loading
cd /workspace/vendor/chatterbox-finetuning
python -c "
from src.chatterbox_.tts import ChatterboxTTS
tts = ChatterboxTTS.from_local('./pretrained_models', device='cpu')
print(f'✓ Tokenizer loaded: {len(tts.tokenizer.vocab)} tokens')
"
# Should output: ✓ Tokenizer loaded: 2459 tokens
```

### Step 4: Configure Training (if needed)

Edit `vendor/chatterbox-finetuning/src/config.py` to adjust hyperparameters:

```python
# Key training settings
batch_size: int = 16         # Adjust based on VRAM (2, 4, 8, 16, 32)
grad_accum: int = 2          # Effective batch = batch_size * grad_accum
learning_rate: float = 1e-5  # Conservative for T3 (sensitive model)
num_epochs: int = 120        # Recommended for fine-tuning
save_steps: int = 500        # Checkpoint frequency
```

**Multi-GPU Training**: Set `NUM_GPUS` in `.env` (e.g., `NUM_GPUS=4`)

### Step 5: Run Training

```bash
# Inside container
cd /workspace/vendor/chatterbox-finetuning

# Start training (this will run for days)
python train.py 2>&1 | tee training.log

# Or run in background with nohup
nohup python train.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or use screen/tmux for persistent sessions
screen -S chatterbox-train
python train.py
# Detach: Ctrl+A, D
# Reattach: screen -r chatterbox-train
```

### Step 6: Monitor Training

**Tensorboard** (recommended):
```bash
# In a separate terminal
docker compose exec chatterbox bash
tensorboard --logdir /workspace/vendor/chatterbox-finetuning/chatterbox_output --host 0.0.0.0

# Access: http://dgx-host:6006
```

**Training Logs**:
```bash
# Watch training progress
tail -f training.log | grep -E "Epoch|Loss|Step"

# Check checkpoint saves
ls -lht /workspace/vendor/chatterbox-finetuning/chatterbox_output/
```

**Expected Training Time**:
- **Single GPU**: ~5-7 days
- **4 GPUs**: ~1.5-2 days (with proper data parallelism)
- **8 GPUs**: ~1 day

### Step 7: Checkpoints and Resuming

**Checkpoints saved at**:
- Location: `vendor/chatterbox-finetuning/chatterbox_output/`
- Frequency: Every 500 steps (`save_steps` config)
- Total kept: Last 5 checkpoints (`save_total_limit`)

**Resume from checkpoint**:
```python
# Training automatically resumes from latest checkpoint if present
# To force resume from specific checkpoint, modify train.py:
# trainer.train(resume_from_checkpoint="./chatterbox_output/checkpoint-2000")
```

**Backup checkpoints regularly**:
```bash
# Copy to safe location
rsync -av /workspace/vendor/chatterbox-finetuning/chatterbox_output/ \
  /path/to/backup/checkpoints/
```

### Step 8: Troubleshooting

**Common Issues**: See `docs/TROUBLESHOOTING.md`

Quick fixes:
```bash
# Perth watermarker error
pip install setuptools==69.5.1

# Flash-attention ABI errors
pip uninstall -y flash-attn

# Dataset not found
cd /workspace/data/processed/MyTTSDataset/wavs
rm *.wav && for f in /data/swara/*.wav; do ln -s "$f" .; done

# Check environment
python -c "import perth; import transformers; print('✓ OK')"
```

**Out of Memory (OOM)**:
- Reduce `batch_size` in config.py (try 8, 4, or 2)
- Increase `grad_accum` to maintain effective batch size
- Monitor GPU memory: `nvidia-smi -l 1`

**Slow Training**:
- Verify GPU usage: `nvidia-smi` (should show ~100% utilization)
- Check data loading: `dataloader_num_workers` (try 8, 16)
- Enable mixed precision if available

## Quick Reference Commands

```bash
# Build and start
docker compose build && docker compose up -d

# Setup environment
docker compose exec chatterbox bash .devcontainer/post-create.sh

# Enter container
docker compose exec chatterbox bash

# Start training
cd vendor/chatterbox-finetuning && python train.py

# Monitor
tail -f training.log

# Check GPU
nvidia-smi

# Count checkpoints
ls chatterbox_output/checkpoint-* | wc -l

# Stop gracefully
# Ctrl+C in training terminal (saves checkpoint before exit)

# Stop container
docker compose down
```

## Project Structure

```
.
├── .devcontainer/
│   ├── devcontainer.json       # VS Code devcontainer config
│   └── post-create.sh          # Environment setup script
├── data/
│   └── processed/
│       └── MyTTSDataset/       # LJSpeech format dataset
├── docs/
│   ├── ADDING-LANGUAGES.md     # Language extension guide
│   ├── DOCKER-SETUP.md         # Docker documentation
│   ├── TOKENIZER-EXTENSION-SUMMARY.md
│   └── UPSTREAMING-PLAN.md     # Contribution strategy
├── scripts/
│   ├── extend_tokenizer.py     # Generic extension tool
│   ├── vocab_extensions/
│   │   ├── romanian.json       # Romanian extension
│   │   └── README.md
│   ├── analyze_swara.py        # Dataset analysis
│   ├── convert_swara_to_ljspeech.py
│   └── verify_tokenizer.py    # Validation tool
├── vendor/
│   └── chatterbox-finetuning/  # Git submodule
├── docker-compose.yml
├── Dockerfile
└── README.md                   # This file
```

## Documentation

- **[ADDING-LANGUAGES.md](docs/ADDING-LANGUAGES.md)**: Complete guide for adding new languages
- **[DOCKER-SETUP.md](docs/DOCKER-SETUP.md)**: Docker architecture and workflows
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Environment setup issues and solutions
- **[TOKENIZER-EXTENSION-SUMMARY.md](docs/TOKENIZER-EXTENSION-SUMMARY.md)**: Implementation details
- **[UPSTREAMING-PLAN.md](docs/UPSTREAMING-PLAN.md)**: Contribution strategy
- **[DGX-DEPLOYMENT.md](docs/DGX-DEPLOYMENT.md)**: Complete DGX deployment guide

## Key Technical Decisions

1. **Multilingual Model** (not Turbo)
   - Reason: Turbo has known issues with new languages
   - Better for character-level learning

2. **Devcontainer Architecture**
   - Clean base image (system deps only)
   - Project setup via post-create script
   - Portable across environments

3. **Generic Extension System**
   - Language-agnostic design
   - Upstreamable to Chatterbox
   - Community contribution model

4. **Validation Split**
   - 5 samples/speaker (~0.5%)
   - Academic use case approved
   - Holdout speakers for zero-shot

## Dataset: SWARA 1.0

- **Size**: 21,304 utterances, 21.67 hours
- **Speakers**: 18 (16 train, 2 holdout)
- **Quality**: 22050 Hz, mono
- **Language**: Romanian with full diacritics
- **Splits**:
  - Train: 18,740 samples (16 speakers)
  - Validation: 80 samples (5 per training speaker)
  - Holdout: 1,484 samples (2 speakers for zero-shot)

## Tokenizer Extension Results

### Characters Added
| Character | Unicode | Frequency | Priority |
|-----------|---------|-----------|----------|
| ș | U+0219 | 13,135 (0.988%) | High |
| ț | U+021B | 12,762 (0.960%) | High |
| Ș | U+0218 | 535 (0.040%) | Medium |
| Ț | U+021A | 18 (0.001%) | Low |
| Ă | U+0102 | 18 (0.001%) | Low |

### Validation
```
✓ All Romanian characters are covered by the tokenizer.
✓ Vocabulary size: 2459 (was 2454)
✓ Coverage: 10/10 characters (100%)
✓ Ready for preprocessing
```

## Environment

### Docker
- **Base**: `nvcr.io/nvidia/pytorch:24.11-py3`
- **Python**: 3.12
- **CUDA**: 12.4
- **PyTorch**: 2.6.0
- **Note**: Changed from 26.01 to 24.11 for vendor compatibility

### Dependencies
- chatterbox-tts
- librosa, soundfile
- tensorboard
- jiwer, pesq, pystoi
- See `requirements.txt` for full list

## Contributing

### To This Project
1. Follow git workflow (feature branches)
2. Document decisions
3. Test thoroughly before committing

### To Upstream Chatterbox
After successful training:
1. See `docs/UPSTREAMING-PLAN.md`
2. Prepare PR with training results
3. Include validation metrics
4. Share with community

## Upstreaming Plan

We're building for **community contribution**:

✅ **Generic design** (not Romanian-specific)
✅ **Well-documented** (comprehensive guides)
✅ **Battle-tested** (validated with SWARA)
✅ **Upstreamable** (clean, maintainable code)

**Timeline**:
- Phase 1: Implementation ✅
- Phase 2: Training validation 🔄
- Phase 3: Metrics collection ⏳
- Phase 4: Upstream PR ⏳

## Team

- **Researcher**: Adrian Stanea
- **Dataset**: SWARA 1.0
- **Framework**: Chatterbox (ResembleAI)
- **Assistant**: Claude Opus 4.6

## References

- [Chatterbox Paper](https://arxiv.org/abs/2506.08387)
- [SWARA Dataset](https://github.com/adrianstanea/SWARA)
- [Romanian Alphabet](https://en.wikipedia.org/wiki/Romanian_alphabet)

## License

This project follows the Chatterbox license (Apache 2.0).

## Support

For questions:
1. Check documentation in `docs/`
2. Review examples in `scripts/vocab_extensions/`
3. See session summaries in `docs/SESSION-*.md`

---

**Status**: ✅ Tasks 1-6 Complete | **Next**: Task 7 (Training on DGX) | **Branch**: `feature/romanian-adaptation`

**Last Updated**: 2026-02-21
