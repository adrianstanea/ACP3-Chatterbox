# Docker Setup Guide

This guide explains the Docker-based development setup for the Romanian Chatterbox fine-tuning project.

## Architecture Overview

The setup uses a **devcontainer approach** with clear separation of concerns:

1. **Dockerfile**: Clean base image with only system dependencies (espeak-ng, ffmpeg, git)
2. **docker-compose.yml**: Volume mounts and runtime configuration
3. **post-create.sh**: Environment setup (Python deps, git submodules, vendor setup)

## Quick Start

### Local Development (VS Code)

1. Open workspace in VS Code
2. When prompted, click "Reopen in Container"
3. Wait for post-create script to complete (~5 min)
4. Verify: `./scripts/check_environment.sh`

### Manual Container Build

```bash
# Build and start container
docker compose build
docker compose up -d

# Run setup inside container
docker compose exec chatterbox bash .devcontainer/post-create.sh

# Verify installation
docker compose exec chatterbox ./scripts/check_environment.sh
```

### DGX Deployment

1. Clone repository on DGX
2. Build container: `docker compose build`
3. Start container: `docker compose up -d`
4. Run setup: `docker compose exec chatterbox bash .devcontainer/post-create.sh`
5. Work inside: `docker compose exec chatterbox bash`

## Key Design Principles

### 1. Clean Base Image

The Dockerfile contains **only system dependencies**:
- No workspace files copied
- No Python dependencies installed
- Cacheable and portable

### 2. Mounted Workspace

All workspace files are mounted as volumes:
- Changes reflected immediately
- No rebuild needed for code changes
- Same workflow for local dev and DGX

### 3. Idempotent Setup Script

`post-create.sh` can be run multiple times safely:
- Installs Python dependencies
- Initializes git submodules
- Sets up vendor code
- Downloads pretrained models (optional)

## File Roles

### Dockerfile
- **Purpose**: Define clean base environment
- **Contains**: System packages only (espeak-ng, ffmpeg, git)
- **Rebuild**: Only when system dependencies change

### docker-compose.yml
- **Purpose**: Runtime configuration
- **Contains**: Volume mounts, GPU config, environment
- **Modify**: When changing mount points or GPU settings

### .devcontainer/post-create.sh
- **Purpose**: Environment setup and configuration
- **Contains**: Python deps, git submodules, vendor setup
- **Run**: After container creation or when dependencies change
- **Portable**: Same script works in devcontainer or DGX

### .devcontainer/devcontainer.json
- **Purpose**: VS Code integration
- **Contains**: Editor settings, extensions, post-create hook

## Common Workflows

### Adding Python Dependencies

```bash
# 1. Add to requirements.txt
echo "new-package==1.0.0" >> requirements.txt

# 2. Install in running container
docker compose exec chatterbox pip install new-package==1.0.0

# 3. For persistent setup, rebuild or re-run post-create:
docker compose exec chatterbox bash .devcontainer/post-create.sh
```

### Updating Vendor Code

```bash
# Pull latest vendor changes
docker compose exec chatterbox git submodule update --remote

# Reinstall vendor dependencies
docker compose exec chatterbox pip install -r vendor/chatterbox-finetuning/requirements.txt
```

### Download Models (3GB)

```bash
docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && python setup.py"
```

### Clean Rebuild

```bash
# Stop and remove container
docker compose down

# Rebuild image
docker compose build --no-cache

# Start and setup
docker compose up -d
docker compose exec chatterbox bash .devcontainer/post-create.sh
```

## Environment Variables

Configure in `.env`:

```bash
# Dataset path (host machine)
SWARA_PATH=/home/user/data/SWARA1.0_22k_noSil

# Output path (host machine)
OUTPUT_PATH=/home/user/outputs

# GPU configuration
NUM_GPUS=1
```

## Mounted Volumes

| Host | Container | Purpose |
|------|-----------|---------|
| `.` | `/workspace` | Workspace files |
| `$SWARA_PATH` | `/data/swara` | SWARA dataset (read-only) |
| `$OUTPUT_PATH` | `/data/output` | Training outputs |

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Post-create fails

```bash
# Run interactively to see errors
docker compose exec chatterbox bash
bash .devcontainer/post-create.sh
```

### Submodule issues

```bash
# Manually initialize
docker compose exec chatterbox git submodule update --init --recursive
```

## Next Steps

After successful setup:

1. **Verify installation**: `./scripts/check_environment.sh`
2. **Download models**: `cd vendor/chatterbox-finetuning && python setup.py`
3. **Run Task 5**: Verify tokenizer (see Task 5 documentation)
4. **Preprocessing**: Convert dataset (Task 6)
5. **Training**: Fine-tune model (Task 7)
