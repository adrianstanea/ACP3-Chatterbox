# DGX Deployment Guide - Chatterbox Romanian Adaptation

This guide covers deploying the Chatterbox Romanian adaptation project on an NVIDIA DGX system.

## Prerequisites

- NVIDIA DGX system with GPU access
- Docker and Docker Compose installed
- Git installed
- SWARA dataset downloaded and accessible
- Minimum 16GB GPU memory per GPU
- Minimum 32GB system RAM

## Initial Setup

### 1. Clone the Repository

```bash
cd /workspace  # or your preferred directory
git clone https://github.com/adrianstanea/ACP3-Chatterbox.git
cd ACP3-Chatterbox
git worktree add .worktrees/romanian-adaptation romanian-adaptation
cd .worktrees/romanian-adaptation
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Configure Environment

Copy the example environment file and edit it for your DGX paths:

```bash
cp .env.example .env
```

Edit `.env` with your actual paths:

```bash
# DGX paths - adjust these for your environment
SWARA_PATH=/data/datasets/SWARA1.0_22k_noSil
OUTPUT_PATH=/workspace/chatterbox-output

# Training configuration
NUM_GPUS=1  # or 2, 4, 8 depending on your DGX
```

### 4. Verify Dataset Path

Ensure the SWARA dataset is accessible at the path specified in `.env`:

```bash
ls -lh $SWARA_PATH/metadata_SWARA1.0_text.csv
```

Expected output: The metadata CSV file should be present and approximately 2.6MB.

## Building the Container

### First-Time Build

The initial build will take approximately 10-15 minutes as it:
- Pulls the NVIDIA PyTorch base image (~10GB)
- Installs system dependencies (espeak-ng, ffmpeg)
- Installs Python dependencies (chatterbox-tts, librosa, etc.)

```bash
docker compose build
```

### Subsequent Builds

If you modify dependencies or the Dockerfile, rebuild with:

```bash
docker compose build --no-cache  # Force clean rebuild
# or
docker compose build  # Use cache when possible
```

## Running the Container

### Start Container in Background

```bash
docker compose up -d
```

This starts the container with:
- GPU access (configured via NUM_GPUS)
- Volume mounts for code, data, and output
- 16GB shared memory for data loading
- Interactive terminal access

### Enter Container

```bash
docker compose exec chatterbox bash
```

You're now inside the container at `/workspace`.

## Initial Verification (First Run)

Once inside the container, run these commands in order:

### 1. Verify Environment

```bash
./scripts/check_environment.sh
```

This checks:
- CUDA and GPU availability
- Python packages
- System dependencies
- Dataset paths
- Directory structure

### 2. Download Pretrained Models

```bash
cd vendor/chatterbox-finetuning
python setup.py
```

Expected behavior:
- Creates `pretrained_models/` directory
- Downloads model weights (~2-3GB total):
  - ve.safetensors (~500MB) - Voice encoder
  - t3_turbo_v1.safetensors (~1.2GB) - Text-to-token model
  - s3gen_meanflow.safetensors (~500MB) - Speech generator
  - conds.pt - Conditioning vectors
  - Tokenizer files
- Merges tokenizer with custom phoneme vocabulary
- Reports new vocabulary size

**Important**: Note the final vocabulary size reported. You'll need this for the next step.

### 3. Verify Tokenizer (Task 5)

Find the tokenizer location:

```bash
find . -name "tokenizer.json" -type f
```

Expected output:
```
./vendor/chatterbox-finetuning/pretrained_models/tokenizer.json
```

Run tokenizer verification:

```bash
cd /workspace
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/MyTTSDataset/metadata.csv
```

Expected output:
```
=== Tokenizer Verification Report ===
Base tokenizer size: 50257
Extended tokenizer size: [NEW_SIZE]
New tokens added: [COUNT]

Romanian-specific characters coverage:
✓ ă, â, î, ș, ț all supported

Metadata analysis:
- Total entries: [COUNT]
- Successfully tokenized: [COUNT]
- Issues found: [COUNT]

[Detailed analysis follows...]
```

### 4. Update Configuration (if needed)

If the verification script reports that vocabulary extension is needed:

```bash
cd vendor/chatterbox-finetuning
nano src/config.py
```

Find the `new_vocab_size` parameter and update it to match the size reported by setup.py:

```python
new_vocab_size: int = [SIZE_FROM_SETUP]  # Update this value
```

Save and exit (Ctrl+X, Y, Enter).

## Project Structure Inside Container

```
/workspace/
├── data/
│   ├── raw/              # Original SWARA dataset (read-only mount)
│   ├── processed/        # Converted LJSpeech format
│   └── analysis/         # Analysis outputs
├── vendor/
│   └── chatterbox-finetuning/
│       ├── pretrained_models/  # Downloaded after setup.py
│       ├── src/
│       └── train.py
├── scripts/
│   ├── check_environment.sh
│   ├── verify_tokenizer.py
│   └── convert_swara_to_ljspeech.py
├── docs/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Common Issues and Solutions

### Issue: "Cannot connect to Docker daemon"

**Cause**: Docker service not running.

**Solution**:
```bash
sudo systemctl start docker
# or on DGX
sudo service docker start
```

### Issue: "Permission denied" on SWARA_PATH

**Cause**: Docker container cannot access the dataset path.

**Solution**:
```bash
# Check permissions
ls -la /data/datasets/SWARA1.0_22k_noSil

# If needed, adjust permissions (use cautiously)
sudo chmod -R a+rX /data/datasets/SWARA1.0_22k_noSil
```

### Issue: "CUDA out of memory"

**Cause**: Insufficient GPU memory.

**Solution**:
- Reduce batch size in training config
- Use fewer GPUs initially
- Free up GPU memory: `docker compose restart`

### Issue: "No space left on device"

**Cause**: Docker volume or system disk full.

**Solution**:
```bash
# Check disk space
df -h

# Clean Docker cache
docker system prune -a

# Move OUTPUT_PATH to larger disk
```

### Issue: Downloads fail during setup.py

**Cause**: Network issues or HuggingFace Hub connectivity.

**Solution**:
```bash
# Check network
ping huggingface.co

# Retry with timeout
cd vendor/chatterbox-finetuning
python -u setup.py 2>&1 | tee setup.log

# If partial downloads exist, remove and retry
rm -rf pretrained_models/
python setup.py
```

### Issue: Tokenizer verification fails

**Cause**: Incorrect paths or missing data.

**Solution**:
```bash
# Verify metadata exists
ls -la data/processed/MyTTSDataset/metadata.csv

# If not, run conversion first
python scripts/convert_swara_to_ljspeech.py \
  /data/swara \
  data/processed/MyTTSDataset

# Then retry verification
```

## Expected First-Run Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Clone repository | 1-2 min | Depends on network |
| Docker build | 10-15 min | First time only |
| Start container | 30 sec | Subsequent starts: <5 sec |
| setup.py downloads | 5-10 min | ~3GB download |
| Tokenizer verification | 30 sec | Analyzes metadata |
| **Total** | **~20-30 min** | First run only |

Subsequent container starts: **<1 minute**

## Resource Requirements

### Minimum Configuration
- 1x GPU (16GB VRAM)
- 32GB system RAM
- 50GB disk space (models + data + output)

### Recommended Configuration
- 2-4x GPUs (16GB+ VRAM each)
- 64GB+ system RAM
- 200GB disk space
- NVMe storage for dataset

### DGX A100 Optimal
- 4-8x A100 GPUs
- 128GB+ system RAM
- 500GB NVMe for workspace
- 1TB+ for output/checkpoints

## Next Steps

After successful verification:

1. **Run preprocessing** (Task 6):
   ```bash
   python scripts/preprocess_dataset.py
   ```

2. **Start training** (Task 7):
   ```bash
   cd vendor/chatterbox-finetuning
   python train.py
   ```

3. **Monitor training**:
   ```bash
   tensorboard --logdir=checkpoints/ --bind_all
   # Access at http://<dgx-ip>:6006
   ```

## Shutting Down

### Stop Container
```bash
docker compose down
```

### Stop and Remove Everything
```bash
docker compose down -v  # Warning: removes volumes
```

### Keep Container Running
```bash
# Exit without stopping
exit  # Just closes your shell session
```

## Support and Troubleshooting

For issues specific to:
- **Container setup**: Check `docs/DOCKER-SETUP.md`
- **Dataset conversion**: Check `scripts/README.md`
- **Training**: Check `vendor/chatterbox-finetuning/README.md`
- **Tokenizer**: Check verification script output

## Development Workflow

### Making Code Changes

The workspace directory is mounted as a volume, so changes made on the host are immediately visible in the container:

```bash
# On host: Edit code
nano scripts/my_script.py

# In container: Run immediately
python scripts/my_script.py
```

### Installing Additional Dependencies

```bash
# In container:
pip install new-package

# To persist, add to requirements.txt on host:
echo "new-package" >> requirements.txt

# Rebuild container:
docker compose build
```

### Rebuilding After Changes

Only rebuild when you change:
- `Dockerfile`
- `requirements.txt`
- System dependencies

Code changes don't require rebuild due to volume mounts.

## Performance Tuning

### Multi-GPU Training

Edit `.env`:
```bash
NUM_GPUS=4  # Use 4 GPUs
```

Restart container:
```bash
docker compose down
docker compose up -d
```

### Shared Memory

If you see "bus error" during data loading, increase shared memory in `docker-compose.yml`:

```yaml
shm_size: '32gb'  # Increase from 16gb
```

### Data Loading

For faster data loading, ensure:
- SWARA_PATH points to NVMe storage
- OUTPUT_PATH is on fast storage
- Sufficient system RAM for caching

## Monitoring

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Container Resources
```bash
docker stats chatterbox
```

### Disk Space
```bash
df -h /workspace
df -h /data/output
```

---

**Last Updated**: 2026-02-21
**Tested On**: NVIDIA DGX A100, Docker 24.0+, Ubuntu 22.04
