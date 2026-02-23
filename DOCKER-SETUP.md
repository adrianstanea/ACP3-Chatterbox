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

### Updating Vendor Code

```bash
# Pull latest vendor changes
docker compose exec chatterbox git submodule update --remote

# Reinstall vendor dependencies
docker compose exec chatterbox pip install -r vendor/chatterbox-finetuning/requirements.txt
```

### Download Models

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

| Host           | Container      | Purpose                   |
| -------------- | -------------- | ------------------------- |
| `.`            | `/workspace`   | Workspace files           |
| `$SWARA_PATH`  | `/data/swara`  | SWARA dataset (read-only) |
| `$OUTPUT_PATH` | `/data/output` | Training outputs          |

## Next Steps

After successful setup:

1. **Verify installation**: `./scripts/check_environment.sh`
2. **Download models**: `cd vendor/chatterbox-finetuning && python setup.py`
