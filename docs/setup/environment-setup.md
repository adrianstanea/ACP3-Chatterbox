# Environment Setup Documentation

**Last Updated:** February 21, 2026
**Status:** Production-ready

## Overview

This document describes the containerized development and training environment for the Romanian Chatterbox adaptation project. The setup is designed to work both locally (for development) and on the DGX system (for full-scale training).

## Architecture Decision: Container-Based Workflow

### Rationale

**Why Docker Compose + Devcontainer?**

1. **Reproducibility:** Exact same environment across machines
2. **Portability:** Works on local machines and DGX
3. **Isolation:** No conflicts with system Python/CUDA
4. **GPU Access:** Native NVIDIA GPU support
5. **IDE Integration:** VS Code devcontainer support

**Key Design Principle:**
> "Never alter the original dataset. Use read-only mounts and environment variables for paths."

### Container Stack

```
┌─────────────────────────────────────────┐
│  Development Environment                │
│  (VS Code + Devcontainer)               │
├─────────────────────────────────────────┤
│  Docker Container                       │
│  - PyTorch 26.01                        │
│  - Python 3.12                          │
│  - CUDA 13.1                            │
│  - Project dependencies                 │
├─────────────────────────────────────────┤
│  Host System                            │
│  - NVIDIA Driver                        │
│  - Docker Engine                        │
│  - Dataset (read-only mount)            │
└─────────────────────────────────────────┘
```

## Base Image Selection

### Chosen Image: `nvcr.io/nvidia/pytorch:26.01-py3`

**Specifications:**
- CUDA Version: 13.1
- PyTorch Version: 2.10
- Python Version: 3.12
- Base OS: Ubuntu 22.04

### Decision History: PyTorch Version

**Timeline:**

1. **Initial Research (Feb 20, 2026):**
   - Considered PyTorch 24.12 (Python 3.10)
   - Concern: Python 3.12 compatibility with Chatterbox

2. **Investigation (Feb 20, 2026):**
   - Researched Chatterbox dependencies
   - Found all packages compatible with Python 3.12
   - Verified via PyPI package metadata

3. **Final Decision (Feb 21, 2026):**
   - Selected PyTorch 26.01 (Python 3.12)
   - Rationale: Latest CUDA support, better performance
   - Confirmed compatibility with all dependencies

**Research Findings:**

| Package | Python 3.12 Support | Notes |
|---------|---------------------|-------|
| torch 2.6.0 | ✓ Yes | Official support |
| torchaudio 2.6.0 | ✓ Yes | Matches torch version |
| transformers | ✓ Yes | Python 3.8+ supported |
| peft 0.17.1 | ✓ Yes | Modern package |
| chatterbox-tts 0.1.2 | ✓ Yes | Verified compatible |
| soundfile 0.13.1 | ✓ Yes | Pure Python + libsndfile |
| librosa 0.11.0 | ✓ Yes | NumPy 2.0+ compatible |

**Conclusion:** PyTorch 26.01 with Python 3.12 is the optimal choice.

### Alternative Considered

**PyTorch 24.12 (Python 3.10):**
- CUDA 12.6
- PyTorch 2.6
- Python 3.10

**Why Not Chosen:**
- Older Python version (3.10 vs 3.12)
- Slightly older CUDA (12.6 vs 13.1)
- No compatibility advantages
- Less future-proof

## Docker Compose Configuration

### File: `docker-compose.yml`

```yaml
services:
  chatterbox:
    image: nvcr.io/nvidia/pytorch:26.01-py3
    volumes:
      - .:/workspace
      - ${SWARA_PATH}:/data/swara:ro
      - ${OUTPUT_PATH}:/data/output
    working_dir: /workspace
    env_file: .env
    shm_size: '16gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${NUM_GPUS:-1}
              capabilities: [gpu]
    command: bash -c "pip install -r requirements.txt && bash"
```

### Key Configuration Elements

#### Volume Mounts

**1. Project Workspace (`/workspace`)**
```yaml
- .:/workspace
```
- Mounts entire project directory
- Read-write access
- Contains code, scripts, docs

**2. SWARA Dataset (`/data/swara`)**
```yaml
- ${SWARA_PATH}:/data/swara:ro
```
- Read-only mount (`:ro` flag)
- Prevents accidental modification
- Path from environment variable

**3. Output Directory (`/data/output`)**
```yaml
- ${OUTPUT_PATH}:/data/output
```
- Read-write access
- Stores processed data
- Stores model checkpoints

#### GPU Configuration

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: ${NUM_GPUS:-1}
          capabilities: [gpu]
```

**Features:**
- Dynamic GPU count via `NUM_GPUS` environment variable
- Default: 1 GPU (for local development)
- Production: All GPUs on DGX
- Native NVIDIA runtime (no legacy `--gpus` flag)

#### Shared Memory

```yaml
shm_size: '16gb'
```

**Purpose:**
- PyTorch DataLoader uses shared memory for multiprocessing
- Default Docker shm (64MB) causes crashes
- 16GB sufficient for 8-16 workers

**Calculation:**
- Rule of thumb: 1-2GB per DataLoader worker
- 8 workers × 2GB = 16GB

#### Startup Command

```yaml
command: bash -c "pip install -r requirements.txt && bash"
```

**Behavior:**
1. Install Python dependencies from `requirements.txt`
2. Drop into interactive bash shell
3. Manual control for development

**Production Alternative:**
```bash
docker compose run --rm chatterbox python vendor/chatterbox-finetuning/train.py
```

## Devcontainer Configuration

### File: `.devcontainer/devcontainer.json`

```json
{
  "name": "Chatterbox Romanian",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "chatterbox",
  "workspaceFolder": "/workspace",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter"
      ]
    }
  }
}
```

### VS Code Integration

**Features:**
- Uses existing Docker Compose service
- Automatic GPU access in IDE
- Python extension with IntelliSense
- Jupyter notebook support

**Workflow:**
1. Open project in VS Code
2. "Reopen in Container" command
3. VS Code runs inside container
4. Full IDE features with GPU access

## Environment Variables

### File: `.env` (gitignored)

**Template:** `.env.example`

```bash
# Dataset paths
SWARA_PATH=/home/astanea/data/SWARA1.0_22k_noSil
OUTPUT_PATH=/home/astanea/data/chatterbox-output

# GPU configuration
NUM_GPUS=1

# Training configuration (optional, can override in config.py)
BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=1e-5
```

### Environment Variable Design

**Principle:** Path injection, not hard-coding

**Benefits:**
1. **Local Development:**
   - Use small dataset subset
   - Output to local SSD

2. **DGX Production:**
   - Use full SWARA dataset
   - Output to fast NVMe storage

3. **Security:**
   - No paths in git
   - Each user has own `.env`

### Creating `.env` File

**Local Development:**
```bash
cp .env.example .env
# Edit .env with your local paths
```

**DGX Production:**
```bash
cat > .env << 'EOF'
SWARA_PATH=/mnt/data/datasets/SWARA1.0_22k_noSil
OUTPUT_PATH=/mnt/nvme/chatterbox-output
NUM_GPUS=4
EOF
```

## DGX Compatibility

### DGX System Specifications

**Hardware:**
- GPUs: NVIDIA V100 32GB (multiple available)
- Memory: 512GB+ RAM
- Storage: NVMe for fast I/O
- Network: High-speed for dataset access

### DGX-Specific Considerations

#### 1. GPU Precision

**V100 Constraint:**
- No BF16 (bfloat16) support
- FP16 (float16) required for mixed precision

**Configuration:**
```python
# In training config
TrainingArguments(
    fp16=True,        # ✓ Works on V100
    bf16=False,       # ✗ Not supported on V100
)
```

#### 2. Multi-GPU Training

**Current Status:** Single-GPU only (fine-tuning kit limitation)

**Workaround Options:**

**Option A: HuggingFace Accelerate**
```bash
accelerate launch vendor/chatterbox-finetuning/train.py
```

**Option B: Torchrun**
```bash
torchrun --nproc_per_node=4 vendor/chatterbox-finetuning/train.py
```

**Option C: Manual DDP** (requires code modification)
- See [Technical Decisions: Multi-GPU Training](../technical-decisions.md#multi-gpu-training)

#### 3. Dataset Path

**DGX Paths:**
- Datasets typically in `/mnt/data/` or network storage
- Checkpoints to local NVMe for speed
- Avoid network I/O for preprocessed data

**Example DGX Setup:**
```bash
# Copy dataset to local NVMe for speed
rsync -av /network/storage/SWARA1.0_22k_noSil/ /mnt/nvme/SWARA/

# Set environment variables
export SWARA_PATH=/mnt/nvme/SWARA
export OUTPUT_PATH=/mnt/nvme/chatterbox-output
export NUM_GPUS=4
```

## Dependency Management

### File: `requirements.txt`

```
# Core dependencies
chatterbox-tts>=0.1.2
torch>=2.6.0
torchaudio>=2.6.0

# Analysis and preprocessing
soundfile>=0.13.1
librosa>=0.11.0
numpy>=1.24.0
pandas>=2.0.0

# Future: add evaluation dependencies
```

### Installation

**Inside Container:**
```bash
pip install -r requirements.txt
```

**Verification:**
```bash
python -c "import chatterbox; import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Vendor Code (Git Submodule)

**Location:** `vendor/chatterbox-finetuning/`

**Installation:**
```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>

# Or initialize submodules later
git submodule update --init --recursive
```

**Why Submodule?**
- Separate version control for vendor code
- Track our fork with modifications
- Clean separation from project code

## System Dependencies

### Installed in Base Image

**CUDA Toolkit:**
- Version: 13.1
- NVCC compiler included
- cuDNN pre-installed

**Python:**
- Version: 3.12
- pip, setuptools, wheel

**System Libraries:**
- libsndfile (for soundfile)
- ffmpeg (for audio conversion)
- apt-get packages pre-configured

### Additional Requirements (if needed)

**espeak-ng** (for phonemization):
```bash
apt-get update && apt-get install -y espeak-ng
```

**Romanian language data:**
```bash
# espeak-ng includes Romanian support by default
espeak-ng -v ro "Bună ziua" --stdout > test.wav
```

## Usage Workflows

### Local Development

**1. Start Container:**
```bash
# Using Docker Compose
docker compose up -d

# Or using VS Code
# File > Reopen in Container
```

**2. Run Analysis:**
```bash
python scripts/analyze_swara.py metadata_SWARA1.0_text.csv --audio-dir /data/swara
```

**3. Test Preprocessing:**
```bash
cd vendor/chatterbox-finetuning
python src/preprocess_ljspeech.py --help
```

### DGX Training

**1. Prepare Environment:**
```bash
# On DGX host
cd /mnt/nvme/projects/chatterbox-romanian
cp .env.example .env
# Edit .env with DGX paths

# Set GPU count
echo "NUM_GPUS=4" >> .env
```

**2. Launch Training:**
```bash
docker compose run --rm chatterbox bash
# Inside container:
cd vendor/chatterbox-finetuning
python train.py
```

**3. Monitor Training:**
```bash
# On host (TensorBoard)
docker compose exec chatterbox tensorboard --logdir /workspace/vendor/chatterbox-finetuning/chatterbox_output
```

### Jupyter Notebook Development

**1. Start Jupyter:**
```bash
docker compose exec chatterbox jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

**2. Access Notebook:**
- Open browser to `http://localhost:8888`
- Token displayed in terminal

**3. Use Case:**
- Interactive data exploration
- Visualization of training curves
- Debugging preprocessing

## Troubleshooting

### GPU Not Detected

**Symptom:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**
1. Verify NVIDIA driver: `nvidia-smi` on host
2. Check Docker GPU runtime: `docker info | grep -i runtime`
3. Verify GPU count in `.env`: `NUM_GPUS=1`
4. Restart Docker daemon: `sudo systemctl restart docker`

### Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `src/config.py`
2. Enable gradient checkpointing
3. Increase gradient accumulation
4. Use smaller GPU count (more memory per GPU)

**Configuration:**
```python
# In src/config.py
batch_size = 2          # Reduced from 4
grad_accum = 16         # Increased to maintain effective batch size
gradient_checkpointing = True
```

### Shared Memory Error

**Symptom:**
```
ERROR: Unexpected bus error encountered in worker
```

**Solution:**
Increase `shm_size` in `docker-compose.yml`:
```yaml
shm_size: '32gb'  # Increased from 16gb
```

### Permission Denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/data/output/...'
```

**Solutions:**
1. Check output path ownership on host
2. Create output directory before mounting:
   ```bash
   mkdir -p $OUTPUT_PATH
   chmod -R 777 $OUTPUT_PATH  # Or proper user permissions
   ```

### Dataset Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/swara/...'
```

**Solutions:**
1. Verify `SWARA_PATH` in `.env`
2. Check mount in container: `docker compose exec chatterbox ls -la /data/swara`
3. Ensure dataset exists on host: `ls -la $SWARA_PATH`

## Performance Optimization

### I/O Optimization

**Preprocessed Data Location:**
- Store on fast SSD/NVMe (not network storage)
- Avoids I/O bottleneck during training
- Processed once, read many times

**DataLoader Workers:**
```python
# In src/config.py
dataloader_num_workers = 8  # Adjust based on CPU cores
```

**Rule of thumb:** 1-2 workers per GPU

### Memory Optimization

**Gradient Checkpointing:**
```python
gradient_checkpointing = True  # Reduces VRAM by ~60%
```

**Trade-off:** Slower training (~20%) but fits larger models

**Mixed Precision:**
```python
fp16 = True  # 2x memory reduction, faster on V100
```

### Multi-GPU Scaling

**Effective Batch Size:**
```
effective_batch = batch_size × grad_accum × num_gpus
```

**Target:** 32 (optimal for Chatterbox)

**Examples:**

| GPUs | Batch/GPU | Grad Accum | Effective Batch |
|------|-----------|------------|-----------------|
| 1 | 4 | 8 | 32 |
| 2 | 4 | 4 | 32 |
| 4 | 4 | 2 | 32 |

## Reproducibility Checklist

**Environment:**
- [ ] Base image version pinned: `nvcr.io/nvidia/pytorch:26.01-py3`
- [ ] Python dependencies frozen: `requirements.txt`
- [ ] Vendor code pinned: Git submodule commit

**Configuration:**
- [ ] `.env` file created and configured
- [ ] Dataset path verified: `SWARA_PATH`
- [ ] Output path created: `OUTPUT_PATH`
- [ ] GPU count set: `NUM_GPUS`

**Verification:**
- [ ] GPU detected: `nvidia-smi` in container
- [ ] PyTorch CUDA: `torch.cuda.is_available()`
- [ ] Dataset accessible: `ls /data/swara`
- [ ] Dependencies installed: `pip list`

## References

### Docker Documentation
- Docker Compose GPU: https://docs.docker.com/compose/gpu-support/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/

### PyTorch Documentation
- PyTorch Containers: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html

### Project Documentation
- [Technical Decisions](../technical-decisions.md)
- [Project Overview](../project-overview.md)
- [Fine-tuning Kit Analysis](../vendor/chatterbox-finetuning-analysis.md)

## Version History

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-02-20 | Initial Docker Compose setup | Containerization requirement |
| 2026-02-20 | PyTorch 24.12 considered | Python 3.12 compatibility concern |
| 2026-02-21 | Switched to PyTorch 26.01 | Confirmed Python 3.12 compatibility |
| 2026-02-21 | Added devcontainer config | VS Code integration |
| 2026-02-21 | Environment variable design | Path injection, portability |
