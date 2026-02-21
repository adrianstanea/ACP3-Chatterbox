# Romanian Chatterbox Fine-Tuning Project

Fine-tuning Chatterbox Multilingual (500M Llama 3) for Romanian using the SWARA 1.0 dataset.

## Project Status

**Current**: ✅ Task 5 Complete - Ready for Task 6 (Preprocessing)

### Completed Tasks (5/11)

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

### Next Steps

6. ⏳ **Preprocessing** - Prepare dataset with extended tokenizer
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

### Next: Run Preprocessing
```bash
# Inside container
cd vendor/chatterbox-finetuning
python train.py  # Will preprocess first
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
- **[TOKENIZER-EXTENSION-SUMMARY.md](docs/TOKENIZER-EXTENSION-SUMMARY.md)**: Implementation details
- **[UPSTREAMING-PLAN.md](docs/UPSTREAMING-PLAN.md)**: Contribution strategy

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
- **Base**: `nvcr.io/nvidia/pytorch:26.01-py3`
- **Python**: 3.12
- **CUDA**: 12.4
- **PyTorch**: 2.6.0

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

**Status**: ✅ Tasks 1-5 Complete | **Next**: Task 6 (Preprocessing) | **Branch**: `feature/romanian-adaptation`
