# Docker Setup - Quick Start

This project is configured to run in a Docker container with full GPU support.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- NVIDIA GPU with 16GB+ VRAM

## Documentation

- **[DGX-DEPLOYMENT.md](docs/DGX-DEPLOYMENT.md)** - Complete deployment guide for DGX systems
- **[DOCKER-SETUP.md](docs/DOCKER-SETUP.md)** - Detailed Docker configuration explanation
- **[DOCKER-QUICK-REFERENCE.md](docs/DOCKER-QUICK-REFERENCE.md)** - Command reference

## Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit paths for your system
nano .env
```

Update these paths in `.env`:

```bash
SWARA_PATH=/path/to/SWARA1.0_22k_noSil  # Your dataset location
OUTPUT_PATH=/path/to/output              # Where to save results
NUM_GPUS=1                                # Number of GPUs to use
```

### 2. Automated Setup (Optional)

Run the automated setup script:

```bash
./scripts/docker_setup_and_verify.sh
```

### 3. Manual Setup (Alternative)

If you prefer manual control:

```bash
# Build image
docker compose build

# Start container
docker compose up -d

# Enter container
docker compose exec chatterbox bash

# Inside container:
./scripts/check_environment.sh
cd vendor/chatterbox-finetuning
python setup.py
cd /workspace
python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/MyTTSDataset/metadata.csv
```
