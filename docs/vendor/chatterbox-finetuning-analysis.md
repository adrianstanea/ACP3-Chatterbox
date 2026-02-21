# Chatterbox Fine-Tuning Kit Analysis

**Upstream Repository**: https://github.com/gokhaneraslan/chatterbox-finetuning
**Our Fork**: https://github.com/adrianstanea/chatterbox-finetuning
**Author**: Gokhan Eraslan
**Submodule Path**: `vendor/chatterbox-finetuning/`
**Pinned Version**: commit `18ffb2d` (added Voice Conditioning Dropout)
**Analysis Date**: 2026-02-21

## Executive Summary

This is a community-maintained fine-tuning infrastructure for Chatterbox TTS models (both Standard and Turbo modes). The kit provides offline preprocessing, vocabulary extension, and training capabilities specifically designed for adapting the model to new languages. For Romanian adaptation, we will use **Standard Mode** (`is_turbo = False`) with the multilingual grapheme-based tokenizer.

## Repository Structure

```
vendor/chatterbox-finetuning/
├── pretrained_models/          # Downloaded by setup.py (not in repo)
├── MyTTSDataset/               # LJSpeech format dataset (metadata.csv + wavs/)
├── FileBasedDataset/           # Alternative: file-based format (.wav + .txt pairs)
├── speaker_reference/          # Reference audio for inference
├── src/
│   ├── config.py               # All configuration parameters
│   ├── dataset.py              # Dataset loader and collators
│   ├── model.py                # Weight transfer and training wrapper
│   ├── preprocess_ljspeech.py  # LJSpeech preprocessor
│   ├── preprocess_file_based.py # File-based preprocessor
│   ├── preprocess_json.py      # JSON-based preprocessor
│   ├── utils.py                # Logger and VAD utilities
│   ├── inference_callback.py   # Inference during training
│   └── chatterbox_/            # Model architectures (forked from Chatterbox)
│       ├── tts.py              # Standard mode TTS engine
│       ├── tts_turbo.py        # Turbo mode TTS engine
│       └── models/
│           ├── t3/             # T3 transformer model (what we fine-tune)
│           ├── s3tokenizer/    # S3Gen tokenizer (speech)
│           ├── s3gen/          # S3Gen vocoder (frozen)
│           └── voice_encoder/  # Voice encoder (frozen)
├── train.py                    # Main training script
├── inference.py                # Speech synthesis script
├── setup.py                    # Model downloader and tokenizer merger
└── requirements.txt            # Python dependencies
```

## Key Components

### 1. Training Modes

The kit supports two distinct modes via `src/config.py`:

#### Standard Mode (`is_turbo = False`) - **RECOMMENDED FOR ROMANIAN**
- **Architecture**: Llama-based
- **Tokenizer**: Grapheme (character-level)
- **Vocabulary**: 2,454 tokens covering 23 languages
- **Best for**: Full control over language-specific characters
- **Default vocab size**: `new_vocab_size = 2454`
- **Tokenizer file**: `pretrained_models/tokenizer.json` (downloaded from `grapheme_mtl_merged_expanded_v1.json`)

#### Turbo Mode (`is_turbo = True`) - **NOT RECOMMENDED FOR ROMANIAN**
- **Architecture**: GPT-2 based
- **Tokenizer**: BPE-based with English foundation
- **Vocabulary**: ~50,000+ tokens (English) + merged graphemes
- **Best for**: Leveraging English base for faster training
- **Vocab size**: ~52,260 (depends on merge operation)
- **Known issues**: Documentation mentions issues with new languages in Turbo mode

### 2. Configuration System (`src/config.py`)

```python
@dataclass
class TrainConfig:
    # === PATHS ===
    model_dir: str = "./pretrained_models"
    csv_path: str = "./MyTTSDataset/metadata.csv"
    wav_dir: str = "./MyTTSDataset/wavs"
    preprocessed_dir: str = "./MyTTSDataset/preprocess"
    output_dir: str = "./chatterbox_output"

    # === DATASET FORMAT ===
    ljspeech: bool = True         # True for LJSpeech format
    json_format: bool = False     # True for JSON format
    preprocess: bool = True       # Run preprocessing

    # === MODE ===
    is_turbo: bool = False        # True for Turbo, False for Standard

    # === VOCABULARY ===
    new_vocab_size: int = 52260 if is_turbo else 2454

    # === HYPERPARAMETERS ===
    batch_size: int = 16
    grad_accum: int = 2
    learning_rate: float = 1e-5
    num_epochs: int = 120
    save_steps: int = 500
    save_total_limit: int = 5
    dataloader_num_workers: int = 8

    # === CONSTRAINTS ===
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    # === INFERENCE ===
    is_inference: bool = False
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "..."
```

**Key Parameters Explanation**:
- `batch_size=16`: Adjust based on GPU VRAM (4 for 12GB, 2 for lower)
- `grad_accum=2`: Effective batch size = batch_size × grad_accum
- `learning_rate=1e-5`: Conservative for transformer stability
- `num_epochs=120`: Default; 150+ recommended for 1 hour of data
- `max_text_len=256`: Maximum text sequence length
- `max_speech_len=850`: Maximum speech token sequence
- `prompt_duration=3.0`: Seconds of audio used as conditioning

### 3. Preprocessing Pipeline

The kit uses **offline preprocessing** to maximize training speed. Preprocessing is mandatory and extracts:

1. **Speaker Embeddings**: Voice encoder (VE) extracts speaker identity
2. **Acoustic Tokens**: S3Gen tokenizer converts audio to discrete tokens
3. **Text Tokens**: Tokenizer converts text to token IDs
4. **Prompt Tokens**: First 3 seconds of audio for conditioning

**Output**: `.pt` files containing:
```python
{
    "speech_tokens": torch.Tensor,    # Acoustic tokens + stop token
    "speaker_emb": torch.Tensor,      # 256-dim speaker embedding
    "prompt_tokens": torch.Tensor,    # First 3s of audio tokens
    "text_tokens": torch.Tensor       # Tokenized text
}
```

**Preprocessing Scripts**:
- `src/preprocess_ljspeech.py`: Reads `metadata.csv` (format: `filename|raw_text|normalized_text`)
- `src/preprocess_file_based.py`: Reads `.wav` + `.txt` pairs from directory
- `src/preprocess_json.py`: Reads JSON metadata

**Audio Handling**:
- Input: Any sample rate (automatically resampled to 16kHz)
- Processing: Mono conversion, 16kHz resampling
- Output: 24kHz synthesis

### 4. Model Architecture

**Components**:
1. **VE (Voice Encoder)**: Extracts speaker embeddings - **FROZEN**
2. **S3Gen (Vocoder)**: Converts mel-spectrograms to waveforms - **FROZEN**
3. **T3 (Text-to-Speech)**: Main transformer model - **TRAINABLE**

**T3 Architecture** (Standard Mode):
- Base: Llama-based transformer
- Weights: `pretrained_models/t3_cfg.safetensors`
- Components:
  - `text_emb`: Text token embeddings (vocab_size × hidden_dim)
  - `text_head`: Text prediction head (vocab_size × hidden_dim)
  - `tfmr`: Transformer backbone
  - `cond_enc`: Conditioning encoder (speaker + prompt)

**Weight Transfer Strategy** (`src/model.py`):
```python
def resize_and_load_t3_weights(new_model, pretrained_state_dict):
    # 1. Copy all matching layers
    # 2. Resize text_emb.weight:
    #    - Copy old tokens
    #    - Initialize new tokens with mean of existing embeddings
    # 3. Resize text_head.weight:
    #    - Copy old tokens
    #    - Initialize new neurons with mean of existing weights
```

This **mean initialization** strategy accelerates convergence for new vocabulary tokens.

### 5. Training Pipeline (`train.py`)

**Flow**:
1. Check pretrained models exist
2. Load original TTS engine (CPU)
3. Create new T3 model with `new_vocab_size`
4. Transfer weights using mean initialization
5. Reload engine with new T3, freeze VE and S3Gen
6. Run preprocessing if `preprocess=True`
7. Load dataset from `.pt` files
8. Train with HuggingFace Trainer
9. Save final model to `chatterbox_output/t3_finetuned.safetensors`

**Training Arguments** (HuggingFace Trainer):
```python
TrainingArguments(
    per_device_train_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.learning_rate,
    num_train_epochs=cfg.num_epochs,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=cfg.save_total_limit,
    fp16=False,
    bf16=True,                      # BF16 for modern GPUs
    gradient_checkpointing=True,    # Reduces VRAM by ~60%
    dataloader_persistent_workers=True,
    dataloader_num_workers=cfg.dataloader_num_workers,
    report_to=["tensorboard"]
)
```

**Loss Function** (`src/model.py`):
```python
total_loss = loss_text + loss_speech

# Where:
# loss_text: CrossEntropy on text token predictions
# loss_speech: CrossEntropy on speech token predictions
#              (with prompt tokens and padding masked out)
```

**Memory Optimizations**:
- Mixed precision: BF16 (better for transformers than FP16)
- Gradient checkpointing: Enabled (trades compute for VRAM)
- Persistent workers: Enabled (faster dataloading)
- Frozen modules: VE and S3Gen not trained

### 6. Inference Pipeline (`inference.py`)

**Flow**:
1. Load base TTS engine
2. Create new T3 with custom vocab size
3. Load fine-tuned weights from `.safetensors`
4. Split text into sentences
5. Generate audio per sentence
6. Trim silence using Silero VAD
7. Concatenate with 0.2s pauses
8. Save to `output.wav` (24kHz)

**Key Features**:
- **VAD Integration**: Silero VAD automatically trims silence/hallucinations
- **Sentence Splitting**: Regex-based (`(?<=[.?!])\s+`)
- **Deterministic**: Seed 42 for reproducibility
- **Mode-specific parameters**:
  - Standard: `temperature=0.8, exaggeration=0.5, cfg_weight=0.5, repetition_penalty=1.2`
  - Turbo: `temperature=0.8, exaggeration=0.5, repetition_penalty=1.2`

### 7. Tokenizer System

**Standard Mode Tokenizer**:
- Type: Grapheme (character-level)
- File: `pretrained_models/tokenizer.json`
- Languages: 23 languages pre-supported
- Vocabulary: 2,454 tokens
- Coverage: Latin, Turkish, Eastern European, and more
- Romanian coverage: Likely complete (ă, â, î, ș, ț are common in Latin sets)

**Tokenizer Customization** (if needed):
1. Verify Romanian characters in default tokenizer
2. If missing characters, create custom `tokenizer.json`
3. Count tokens in JSON
4. Update `new_vocab_size` in **both** `src/config.py` and `inference.py`
5. Replace `pretrained_models/tokenizer.json` before training

**Critical**: `new_vocab_size` must match exactly in:
- `src/config.py`: Used for training
- `inference.py`: Used for loading fine-tuned model

### 8. Dataset Requirements

**Format Options**:
1. **LJSpeech** (recommended): `metadata.csv` + `wavs/` directory
2. **File-based**: `.wav` + `.txt` pairs in same directory
3. **JSON**: JSON metadata file

**LJSpeech Format**:
```
MyTTSDataset/
├── metadata.csv        # filename|raw_text|normalized_text
└── wavs/
    ├── file001.wav
    ├── file002.wav
    └── ...
```

**Quality Requirements**:
- Sample rate: Any (resampled to 16kHz)
- Format: WAV (mono or stereo, converted to mono)
- Duration per clip: 3-10 seconds (optimal)
- Total duration: 1+ hour recommended
- Audio quality: Clean, minimal background noise
- Recommended training: 150 epochs or 1000 steps for 1 hour of data

### 9. Multi-GPU Support

**Current Status**: **NO OFFICIAL MULTI-GPU SUPPORT**

**Evidence**:
- No DDP (DistributedDataParallel) imports in `train.py`
- No `local_rank`, `world_size`, or distributed setup
- Uses standard HuggingFace Trainer without distributed flags
- DDP references only in S3Gen model files (frozen component)

**Workarounds**:
1. **HuggingFace Accelerate**: Can enable multi-GPU via launcher
2. **Manual DDP**: Modify `train.py` to add distributed training
3. **Single GPU**: Default mode (what most users do)

**Recommendation**: Start with single-GPU training. Multi-GPU can be added later if needed.

### 10. Dependencies (`requirements.txt`)

```
peft==0.17.1                # Parameter-Efficient Fine-Tuning
torch==2.6.0                # PyTorch
torchaudio==2.6.0           # Audio processing
torchvision==0.21.0         # Vision (dependency)
chatterbox-tts==0.1.2       # Official Chatterbox package
silero-vad==6.2.0           # Voice Activity Detection
librosa==0.11.0             # Audio analysis
soundfile==0.13.1           # Audio I/O
num2words                   # Number normalization
ffmpeg-python               # Audio conversion
tqdm                        # Progress bars
pandas                      # CSV reading
safetensors                 # Model format
tensorboard                 # Training visualization
omegaconf                   # Configuration
hf_transfer                 # HuggingFace downloads
pyloudnorm                  # Audio normalization
gdown                       # Google Drive downloads
```

**System Dependencies**:
- FFmpeg (required for audio processing)
- CUDA 11.8+ (for GPU training)

## Critical Workflows

### Mode Switching (CRITICAL WARNING)

**If switching between Standard and Turbo modes**:
1. **DELETE entire `pretrained_models/` directory**
2. **DELETE preprocessed data directory**
3. Update `is_turbo` in `src/config.py`
4. Run `python setup.py`
5. Update `new_vocab_size` with value from setup output
6. Set `preprocess=True`
7. Run training

**Why**: Tokenizer files are replaced in-place. Switching modes without cleanup causes corruption.

### Setup Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure mode
# Edit src/config.py: set is_turbo = False

# 3. Download models
python setup.py

# 4. Prepare dataset
# Place data in MyTTSDataset/ or FileBasedDataset/

# 5. Train
python train.py

# 6. Inference
# Edit inference.py: set TEXT_TO_SAY and AUDIO_PROMPT
python inference.py
```

## Known Issues and Limitations

### 1. Turbo Mode Issues with New Languages
- Documentation explicitly mentions issues with Turbo on new languages
- Romanian should use Standard mode

### 2. Vocab Size Synchronization
- **Critical**: `new_vocab_size` must match in `src/config.py` AND `inference.py`
- Mismatch causes: `RuntimeError: size mismatch in loading state_dict`

### 3. Memory Requirements
- T3 is VRAM-intensive
- Minimum: 12GB VRAM for `batch_size=4`
- Lower VRAM: Use `batch_size=2` with `grad_accum=32`
- Gradient checkpointing reduces VRAM by ~60% but slows training

### 4. No Multi-GPU Support
- Single GPU only by default
- Would require custom modifications for DDP

### 5. Preprocessing is Mandatory
- Cannot skip preprocessing
- Must rerun if changing modes or dataset

### 6. Audio Format Constraints
- Input must be clean (low background noise)
- Silence trimming happens only at inference (VAD)
- Poor quality audio = poor quality synthesis

## Patches and Modifications Needed for Romanian Adaptation

### 1. Tokenizer Verification (Phase 2, Task 5)
**File**: `pretrained_models/tokenizer.json`

**Action**: Verify Romanian characters are included:
- ă (a-breve)
- â (a-circumflex)
- î (i-circumflex)
- ș (s-comma)
- ț (t-comma)

**If missing**:
- Create custom tokenizer.json
- Update `new_vocab_size` in both config files

### 2. Configuration Adjustments (Phase 2, Task 6)
**File**: `src/config.py`

**Changes needed**:
```python
# Dataset paths
csv_path: str = "./data/swara_converted/metadata.csv"
wav_dir: str = "./data/swara_converted/wavs"
preprocessed_dir: str = "./data/swara_converted/preprocess"

# Training for SWARA dataset (estimate ~1-2 hours)
num_epochs: int = 150  # or 1000 steps
batch_size: int = 4    # adjust based on GPU
grad_accum: int = 8    # effective batch = 32

# Ensure Standard mode
is_turbo: bool = False
```

### 3. Multi-GPU Support (Optional)
**File**: `train.py`

**If needed**:
- Add `torchrun` launcher support
- Initialize process group
- Wrap model in DDP
- Adjust batch size per GPU
- Handle checkpoint saving on rank 0 only

**Estimated effort**: 50-100 lines of code

### 4. Evaluation Integration (Phase 3)
**New file**: `src/evaluation.py`

**Features needed**:
- Load fine-tuned model
- Generate test set samples
- Compute WER, CER, MOS (if possible)
- Save results to JSON

**Estimated effort**: 200-300 lines

### 5. Documentation (Phase 4)
**New file**: `docs/romanian-training-guide.md`

**Contents**:
- Romanian-specific tokenizer setup
- SWARA dataset preparation
- Training hyperparameters for Romanian
- Known issues and workarounds

## Comparison with Official Chatterbox

**What this kit provides**:
- Fine-tuning infrastructure (official Chatterbox has no public fine-tuning code)
- Vocabulary extension for new languages
- Offline preprocessing for speed
- LJSpeech dataset support
- Training loop with checkpointing
- Inference with VAD

**What it does NOT provide**:
- Multi-GPU training (official might have it internally)
- Advanced data augmentation
- Hyperparameter tuning tools
- Evaluation metrics (WER, CER, MOS)
- Production deployment tools

## Recommendations for Romanian Adaptation

### 1. Use Standard Mode
- `is_turbo = False`
- More stable for non-English languages
- Simpler tokenizer management

### 2. Verify Tokenizer First
- Check `tokenizer.json` for Romanian characters
- Create custom tokenizer if needed
- Test tokenization before training

### 3. Start with Small Batch Size
- `batch_size = 4` for initial testing
- Increase if VRAM allows
- Use gradient accumulation for effective larger batches

### 4. Monitor Training Closely
- Use TensorBoard: `tensorboard --logdir chatterbox_output`
- Check loss convergence
- Generate samples periodically

### 5. Plan for Long Training
- 150+ epochs for 1 hour of data
- Expect 10-20 hours on single GPU
- Save checkpoints every 500 steps

### 6. Preprocessing Once
- Preprocessing takes time (1-2 hours for SWARA)
- Set `preprocess = False` after first run
- Keep preprocessed `.pt` files safe

### 7. Test Inference Early
- Run inference after 50 epochs
- Validate audio quality
- Adjust hyperparameters if needed

## Files Requiring Attention

**Immediate**:
- `src/config.py`: Adjust all paths and hyperparameters
- `pretrained_models/tokenizer.json`: Verify Romanian support

**During Training**:
- `train.py`: Monitor for errors
- `chatterbox_output/`: Check checkpoints

**For Inference**:
- `inference.py`: Update text and reference paths
- `speaker_reference/`: Provide clean Romanian reference audio

**For Deployment**:
- `requirements.txt`: Freeze versions for reproducibility
- `setup.py`: Ensure correct model downloads

## Technical Debt and Concerns

### 1. No Version Pinning for Chatterbox
- Uses `chatterbox-tts==0.1.2`
- Upstream changes could break compatibility
- **Mitigation**: Fork and vendor chatterbox package

### 2. Hard-coded Model URLs
- `setup.py` has hard-coded HuggingFace URLs
- Models could move or change
- **Mitigation**: Download models once, commit to LFS

### 3. Mixed Tokenizer Handling
- Standard mode: Custom tokenizer class
- Turbo mode: HuggingFace tokenizer
- Complex switching logic
- **Mitigation**: Use Standard mode only

### 4. No Validation Split
- Training uses entire dataset
- No validation metrics during training
- **Mitigation**: Split SWARA into train/val manually

### 5. Limited Error Handling
- Preprocessing can silently skip files
- No detailed error reporting
- **Mitigation**: Add logging, validate dataset before training

## Summary

The Chatterbox fine-tuning kit is a functional but minimal infrastructure for adapting Chatterbox to new languages. For Romanian adaptation:

**Strengths**:
- Works with multilingual grapheme tokenizer
- Offline preprocessing for speed
- Supports LJSpeech format (matches our conversion plan)
- Mean initialization for new tokens
- VAD for clean inference

**Weaknesses**:
- No multi-GPU support
- No evaluation tools
- No validation split
- Limited documentation for non-English languages
- Requires manual tokenizer verification

**Next Steps**:
1. Verify Romanian tokenizer support (Task 5)
2. Convert SWARA to LJSpeech format (Task 4)
3. Run preprocessing (Task 6)
4. Start training (Task 7)
5. Build evaluation tools (Tasks 9-10)

**Overall Assessment**: **Suitable for Romanian adaptation with minor modifications**. The kit provides a solid foundation. Main gaps are in evaluation and multi-GPU support, which we can add incrementally.
