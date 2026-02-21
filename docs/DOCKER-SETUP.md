# Docker Setup Documentation

This document explains the Docker configuration for the Chatterbox Romanian adaptation project.

## Overview

The project uses Docker to provide a consistent, reproducible environment with:
- NVIDIA PyTorch base image (26.01-py3)
- CUDA support for GPU acceleration
- All required dependencies pre-installed
- Volume mounts for development workflow

## Files

### Core Files

1. **Dockerfile** - Defines the container image
2. **docker-compose.yml** - Orchestrates container configuration
3. **.env** - Environment-specific configuration
4. **.env.example** - Template for environment setup

### Documentation

1. **docs/DGX-DEPLOYMENT.md** - Comprehensive deployment guide for DGX systems
2. **docs/DOCKER-QUICK-REFERENCE.md** - Command reference
3. **docs/DOCKER-SETUP.md** - This file

### Scripts

1. **scripts/docker_setup_and_verify.sh** - Automated setup and verification

## Dockerfile Architecture

### Base Image

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.01-py3
```

Uses NVIDIA's official PyTorch container with:
- Ubuntu 22.04 base
- CUDA 12.3
- cuDNN 8.9
- PyTorch 2.6.1
- Python 3.10

### System Dependencies

```dockerfile
RUN apt-get update && apt-get install -y \
    espeak-ng \    # Phonemization
    ffmpeg \       # Audio processing
    git \          # Version control
    && rm -rf /var/lib/apt/lists/*
```

### Python Dependencies

Installed from `requirements.txt`:
- chatterbox-tts - Core TTS engine
- espeak-phonemize - Phoneme conversion
- librosa - Audio analysis
- soundfile - Audio I/O
- tensorboard - Training visualization
- jiwer, pesq, pystoi - Evaluation metrics

### Working Directory

```dockerfile
WORKDIR /workspace
```

All operations happen in `/workspace`, which is mounted from the host.

### Entry Point

```dockerfile
CMD ["/bin/bash"]
```

Provides interactive shell access by default.

## Docker Compose Configuration

### Service Definition

```yaml
services:
  chatterbox:
    build:
      context: .
      dockerfile: Dockerfile
```

Builds from local Dockerfile rather than pulling pre-built image.

### Volume Mounts

```yaml
volumes:
  - .:/workspace                    # Project code
  - ${SWARA_PATH}:/data/swara:ro   # Dataset (read-only)
  - ${OUTPUT_PATH}:/data/output    # Training outputs
```

**Why these mounts?**
- `.:/workspace` - Edit code on host, run in container
- Dataset read-only - Prevent accidental modification
- Output writable - Store checkpoints and results

### GPU Configuration

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: ${NUM_GPUS:-1}
          capabilities: [gpu]
```

Exposes GPUs to container. Count controlled via `.env`.

### Shared Memory

```yaml
shm_size: '16gb'
```

Required for PyTorch DataLoader with multiple workers. Prevents "bus error" during training.

### Interactive Mode

```yaml
stdin_open: true
tty: true
```

Enables interactive terminal access.

## Environment Variables

Configured in `.env`:

```bash
# Dataset location on host
SWARA_PATH=/data/swara

# Output directory (created if missing)
OUTPUT_PATH=/data/output

# Number of GPUs to use
NUM_GPUS=1
```

### DGX Configuration

For DGX systems, adjust paths:

```bash
SWARA_PATH=/data/datasets/SWARA1.0_22k_noSil
OUTPUT_PATH=/workspace/chatterbox-output
NUM_GPUS=4
```

## Build Process

### What Happens During Build

1. **Base image pull** (~10GB, one-time)
   - NVIDIA PyTorch image from NGC registry
   - Includes CUDA, cuDNN, PyTorch

2. **System package installation**
   - espeak-ng for phonemization
   - ffmpeg for audio processing
   - git for submodules

3. **Python package installation**
   - Installs from requirements.txt
   - Downloads and compiles native extensions

4. **Code copy**
   - Copies workspace into image
   - Note: Volume mount overrides this at runtime

### Build Time

- First build: 10-15 minutes
- Subsequent builds: 1-2 minutes (with cache)
- No-cache rebuild: Same as first build

### Build Optimization

The Dockerfile is ordered for optimal caching:
1. Install system deps (rarely changes)
2. Copy requirements.txt (changes occasionally)
3. Install Python deps (changes occasionally)
4. Copy code (changes frequently, but overridden by volume mount)

## Runtime Behavior

### Container Startup

```bash
docker compose up -d
```

1. Checks if image exists (builds if not)
2. Creates container from image
3. Mounts volumes
4. Configures GPU access
5. Starts in background

### Entering Container

```bash
docker compose exec chatterbox bash
```

1. Attaches to running container
2. Opens bash shell in `/workspace`
3. GPU, Python, and all deps available

### File Persistence

| Location | Persistence | Notes |
|----------|-------------|-------|
| `/workspace` | Volume mount | Changes visible on host |
| `/data/swara` | Volume mount | Read-only dataset |
| `/data/output` | Volume mount | Training outputs |
| `/root` | Container only | Lost on rebuild |
| `/tmp` | Container only | Lost on restart |

## Development Workflow

### Typical Session

```bash
# Start container (if not running)
docker compose up -d

# Enter for work
docker compose exec chatterbox bash

# Inside container: run commands
cd vendor/chatterbox-finetuning
python train.py

# Exit shell (container keeps running)
exit

# Stop container when done
docker compose down
```

### Making Changes

**Code changes** - Edit on host, run in container:
```bash
# Host
nano scripts/my_script.py

# Container (no rebuild needed)
python scripts/my_script.py
```

**Dependency changes** - Rebuild required:
```bash
# Host
echo "new-package" >> requirements.txt

# Rebuild
docker compose build

# Restart
docker compose up -d
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

**Symptom**: `nvidia-smi` fails or shows no GPUs

**Check**:
```bash
# On host
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Fix**: Install/update NVIDIA Container Toolkit

#### 2. Permission Denied on Volumes

**Symptom**: Cannot read/write mounted directories

**Check**:
```bash
ls -la $SWARA_PATH
ls -la $OUTPUT_PATH
```

**Fix**:
```bash
sudo chown -R $(id -u):$(id -g) $OUTPUT_PATH
```

#### 3. Out of Shared Memory

**Symptom**: "bus error" during DataLoader

**Fix**: Increase `shm_size` in docker-compose.yml:
```yaml
shm_size: '32gb'  # Increase from 16gb
```

#### 4. Build Fails

**Symptom**: Error during `docker compose build`

**Fix**:
```bash
# Clean build
docker compose build --no-cache

# If still fails, check logs
docker compose build 2>&1 | tee build.log
```

## Performance Considerations

### Storage

- Use NVMe for `SWARA_PATH` and `OUTPUT_PATH`
- Avoid network mounts for dataset
- Minimum 50GB free space

### Memory

- 32GB RAM minimum for single GPU
- 64GB+ recommended for multi-GPU
- 16-32GB shared memory for DataLoader

### GPU

- 16GB VRAM minimum per GPU
- Multi-GPU: adjust NUM_GPUS in .env
- Monitor: `watch -n 1 nvidia-smi`

## Security Notes

### Volume Mounts

- Dataset mounted read-only (`:ro`) - prevents accidental modification
- Output writable - but isolated to designated directory
- Workspace writable - necessary for development

### Network

- Container uses host network by default
- No exposed ports unless specified
- Outbound: HuggingFace Hub for model downloads

### User Permissions

- Container runs as root by default
- Files created in volumes owned by root
- Can be changed with `user:` directive if needed

## Advanced Configuration

### Custom Base Image

To use a different PyTorch version:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.12-py3
```

Update tag in Dockerfile, then rebuild.

### Additional System Packages

Add to Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
    git \
    vim \          # Add your package
    htop \         # Add your package
    && rm -rf /var/lib/apt/lists/*
```

### Python Version

Base image provides Python 3.10. To use different version, choose appropriate base image.

### Multi-Stage Builds

For production deployment, consider multi-stage build:

```dockerfile
# Build stage
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder
# ... build steps ...

# Runtime stage
FROM nvcr.io/nvidia/pytorch:26.01-py3
COPY --from=builder /workspace /workspace
```

## Comparison: Dev vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Base image | Full PyTorch | Same (GPU required) |
| Volume mounts | Yes (live editing) | No (baked in) |
| Shared memory | 16GB | Tune per workload |
| Rebuild frequency | Rarely | Per deploy |
| Entry point | bash | train.py |

## Testing the Setup

Use the automated verification script:

```bash
./scripts/docker_setup_and_verify.sh
```

This will:
1. Build the image
2. Start the container
3. Verify environment
4. Download models
5. Run tokenizer verification

Or test manually:

```bash
# Build
docker compose build

# Start
docker compose up -d

# Test GPU
docker compose exec chatterbox nvidia-smi

# Test Python
docker compose exec chatterbox python -c "import torch; print(torch.cuda.is_available())"

# Test dependencies
docker compose exec chatterbox python -c "import chatterbox"
```

## Maintenance

### Updating Dependencies

1. Edit `requirements.txt`
2. Rebuild: `docker compose build`
3. Test: `docker compose up -d && docker compose exec chatterbox bash`

### Updating Base Image

1. Check for new PyTorch releases: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
2. Update `FROM` in Dockerfile
3. Rebuild with `--no-cache`
4. Test thoroughly

### Cleaning Up

```bash
# Remove stopped containers
docker system prune

# Remove unused images
docker image prune -a

# Remove unused volumes (WARNING: data loss)
docker volume prune
```

## Resources

- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [PyTorch Docker Guide](https://github.com/pytorch/pytorch#docker-image)

---

**Last Updated**: 2026-02-21
**Docker Version**: 24.0+
**Docker Compose Version**: 2.0+
