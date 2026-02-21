# Docker Setup - Quick Start

This project is configured to run in a Docker container with full GPU support.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- NVIDIA GPU with 16GB+ VRAM

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

### 2. Automated Setup (Recommended)

Run the automated setup script:

```bash
./scripts/docker_setup_and_verify.sh
```

This will:
- Build the Docker image (~15 minutes first time)
- Start the container
- Verify environment
- Download pretrained models (~3GB, ~10 minutes)
- Run tokenizer verification (Task 5)

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

## Daily Usage

```bash
# Start container
docker compose up -d

# Enter container for work
docker compose exec chatterbox bash

# When done
docker compose down  # or leave running
```

## What's Inside?

The container includes:
- NVIDIA PyTorch 2.6.1 with CUDA 12.3
- Python 3.10
- espeak-ng for phonemization
- ffmpeg for audio processing
- All Python dependencies from requirements.txt
- Full GPU access

## Documentation

- **[DGX-DEPLOYMENT.md](docs/DGX-DEPLOYMENT.md)** - Complete deployment guide for DGX systems
- **[DOCKER-SETUP.md](docs/DOCKER-SETUP.md)** - Detailed Docker configuration explanation
- **[DOCKER-QUICK-REFERENCE.md](docs/DOCKER-QUICK-REFERENCE.md)** - Command reference

## Common Commands

```bash
# Build
docker compose build

# Start
docker compose up -d

# Enter
docker compose exec chatterbox bash

# Stop
docker compose down

# Logs
docker compose logs -f

# GPU check
docker compose exec chatterbox nvidia-smi

# Rebuild
docker compose build --no-cache
```

## Next Steps After Setup

Once setup completes successfully:

1. Review tokenizer verification output
2. Update `vendor/chatterbox-finetuning/src/config.py` with new vocab size if needed
3. Run preprocessing (Task 6)
4. Start training (Task 7)

## Troubleshooting

See [DGX-DEPLOYMENT.md](docs/DGX-DEPLOYMENT.md#common-issues-and-solutions) for detailed troubleshooting.

Quick fixes:

- **GPU not found**: `nvidia-smi` on host, restart Docker daemon
- **Permission denied**: Check `.env` paths and file permissions
- **Build fails**: `docker compose build --no-cache`
- **Out of memory**: Reduce batch size or increase shared memory

## File Structure

```
.
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Service orchestration
├── .env                            # Your configuration
├── .env.example                    # Template
├── DOCKER-README.md                # This file
├── requirements.txt                # Python dependencies
├── docs/
│   ├── DGX-DEPLOYMENT.md          # Full deployment guide
│   ├── DOCKER-SETUP.md            # Technical details
│   └── DOCKER-QUICK-REFERENCE.md  # Command reference
└── scripts/
    └── docker_setup_and_verify.sh # Automated setup
```

## Support

For issues:
1. Check logs: `docker compose logs`
2. Verify GPU: `nvidia-smi`
3. Check environment: Inside container run `./scripts/check_environment.sh`
4. See troubleshooting guide in DGX-DEPLOYMENT.md

---

**Ready to start?** Run `./scripts/docker_setup_and_verify.sh`
