# Docker CLI Commands (Alternative to docker-compose)

This guide provides Docker CLI equivalents to the docker-compose.yml configuration for systems without docker-compose or sudo access.

## Quick Start

### Option 1: Use the Helper Script (Recommended)
```bash
# First time setup
./docker-cli-commands.sh full-setup

# Enter container
./docker-cli-commands.sh exec

# Stop container
./docker-cli-commands.sh stop

# Start again
./docker-cli-commands.sh start
./docker-cli-commands.sh exec
```

### Option 2: Manual Commands

#### Step 1: Build the Image
```bash
docker build -t chatterbox:latest .
```

**Equivalent to**: `docker compose build`

#### Step 2: Create and Run Container
```bash
# Load environment variables
export $(grep -v '^#' .env | xargs)

# Run container
docker run -d \
  --name chatterbox \
  --gpus "device=0-$((NUM_GPUS-1))" \
  --shm-size 16g \
  -v "$(pwd):/workspace" \
  -v "${SWARA_PATH}:/data/swara:ro" \
  -v "${OUTPUT_PATH}:/data/output" \
  -w /workspace \
  --env-file .env \
  -it \
  chatterbox:latest
```

**Equivalent to**: `docker compose up -d`

**Explanation of flags**:
- `-d`: Run in detached mode (background)
- `--name chatterbox`: Name the container
- `--gpus "device=0-7"`: Use GPUs 0-7 (adjust based on NUM_GPUS in .env)
- `--shm-size 16g`: Set shared memory to 16GB
- `-v "$(pwd):/workspace"`: Mount current directory to /workspace
- `-v "${SWARA_PATH}:/data/swara:ro"`: Mount SWARA dataset (read-only)
- `-v "${OUTPUT_PATH}:/data/output"`: Mount output directory
- `-w /workspace`: Set working directory
- `--env-file .env`: Load environment variables from .env
- `-it`: Interactive with TTY

#### Step 3: Run Post-Create Setup
```bash
docker exec chatterbox bash /workspace/.devcontainer/post-create.sh
```

**Equivalent to**: `docker compose exec chatterbox bash .devcontainer/post-create.sh`

#### Step 4: Enter Container
```bash
docker exec -it chatterbox bash
```

**Equivalent to**: `docker compose exec chatterbox bash`

## Common Operations

### Start Existing Container
```bash
docker start chatterbox
```

**Equivalent to**: `docker compose start`

### Stop Container
```bash
docker stop chatterbox
```

**Equivalent to**: `docker compose stop`

### View Logs
```bash
docker logs -f chatterbox
```

**Equivalent to**: `docker compose logs -f`

### Remove Container
```bash
docker rm -f chatterbox
```

**Equivalent to**: `docker compose down`

### Rebuild and Restart
```bash
# Remove old container
docker rm -f chatterbox

# Rebuild image
docker build -t chatterbox:latest .

# Run new container (use command from Step 2 above)
docker run -d --name chatterbox ...
```

**Equivalent to**: `docker compose up -d --build`

## GPU Configuration

### Using Specific GPUs

**Single GPU (GPU 0)**:
```bash
docker run ... --gpus device=0 ...
```

**Multiple GPUs (0-3)**:
```bash
docker run ... --gpus device=0-3 ...
```

**All GPUs**:
```bash
docker run ... --gpus all ...
```

**From .env file** (default):
```bash
export $(grep -v '^#' .env | xargs)
docker run ... --gpus "device=0-$((NUM_GPUS-1))" ...
```

## Troubleshooting

### Container Already Exists
```bash
# Error: Conflict. The container name "/chatterbox" is already in use.
# Solution: Remove the old container first
docker rm -f chatterbox
```

### GPU Not Available
```bash
# Error: could not select device driver "" with capabilities: [[gpu]]
# Check: Is NVIDIA Docker runtime installed?
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Alternative: Try without docker-compose GPU syntax
docker run ... --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 ...
```

### Environment Variables Not Loading
```bash
# Option 1: Load from .env automatically
docker run ... --env-file .env ...

# Option 2: Export and use individual variables
export $(grep -v '^#' .env | xargs)
docker run ... -e SWARA_PATH -e OUTPUT_PATH -e NUM_GPUS ...

# Option 3: Set manually
docker run ... \
  -e SWARA_PATH=/path/to/swara \
  -e OUTPUT_PATH=/path/to/output \
  -e NUM_GPUS=8 \
  ...
```

### Cannot Access .env File
```bash
# If .env doesn't exist, create it:
cat > .env << 'EOF'
# Host paths (used for Docker volume mounting)
HOST_SWARA_PATH=/mnt/QNAP/staria/SWARA/SWARA1.0_22k_noSil
HOST_OUTPUT_PATH=/mnt/QNAP/staria/Bogdan/ACP/ACP3-logs

# Container paths (what scripts see inside the container)
SWARA_PATH=/data/swara
OUTPUT_PATH=/data/output

# Processed dataset paths (inside workspace)
DATASET_DIR=/workspace/data/processed/MyTTSDataset
PREPROCESS_DIR=/workspace/data/processed/MyTTSDataset/preprocess

# Training
NUM_GPUS=4
CUDA_VISIBLE_DEVICES=4,5,6,7
EOF
```

## Training Workflow

### Start Training (Inside Container)
```bash
# Enter container
docker exec -it chatterbox bash

# Navigate to vendor directory
cd vendor/chatterbox-finetuning

# Start training
python train.py 2>&1 | tee training.log
```

### Monitor Training (Outside Container)
```bash
# Watch logs
docker exec chatterbox tail -f /workspace/vendor/chatterbox-finetuning/training.log

# Check GPU usage
docker exec chatterbox nvidia-smi
```

### Run in Background with Screen/Tmux
```bash
# Enter container
docker exec -it chatterbox bash

# Start screen session
screen -S training

# Start training
cd vendor/chatterbox-finetuning
python train.py

# Detach: Ctrl+A, then D

# Reattach later:
docker exec -it chatterbox bash
screen -r training
```

## Comparison: docker-compose vs Docker CLI

| Operation | docker-compose | Docker CLI | Helper Script |
|-----------|----------------|------------|---------------|
| Build | `docker compose build` | `docker build -t chatterbox:latest .` | `./docker-cli-commands.sh build` |
| Start | `docker compose up -d` | `docker run -d --name chatterbox ...` | `./docker-cli-commands.sh run` |
| Enter | `docker compose exec chatterbox bash` | `docker exec -it chatterbox bash` | `./docker-cli-commands.sh exec` |
| Stop | `docker compose stop` | `docker stop chatterbox` | `./docker-cli-commands.sh stop` |
| Remove | `docker compose down` | `docker rm -f chatterbox` | `./docker-cli-commands.sh remove` |
| Logs | `docker compose logs -f` | `docker logs -f chatterbox` | `./docker-cli-commands.sh logs` |

## Advanced: Customizing the Run Command

### Using Different Paths
```bash
docker run -d \
  --name chatterbox \
  --gpus device=0-7 \
  --shm-size 16g \
  -v "$(pwd):/workspace" \
  -v "/custom/path/to/swara:/data/swara:ro" \
  -v "/custom/path/to/output:/data/output" \
  -w /workspace \
  -e SWARA_PATH=/custom/path/to/swara \
  -e OUTPUT_PATH=/custom/path/to/output \
  -e NUM_GPUS=8 \
  -it \
  chatterbox:latest
```

### Using Fewer GPUs
```bash
# Use only 2 GPUs (0 and 1)
docker run ... --gpus device=0-1 ... -e NUM_GPUS=2 ...
```

### Running Without GPUs (CPU-only testing)
```bash
# Remove --gpus flag entirely
docker run -d \
  --name chatterbox \
  --shm-size 16g \
  -v "$(pwd):/workspace" \
  -v "${SWARA_PATH}:/data/swara:ro" \
  -v "${OUTPUT_PATH}:/data/output" \
  -w /workspace \
  --env-file .env \
  -it \
  chatterbox:latest
```

## Notes

1. **Persistent Containers**: The docker CLI approach creates a persistent named container (`chatterbox`) that you can start/stop repeatedly, just like docker-compose.

2. **Environment Variables**: The `.env` file uses a dual-path system:
   - `HOST_*_PATH` variables: Host machine paths used for Docker volume mounting
   - `SWARA_PATH`/`OUTPUT_PATH`: Container paths that scripts see inside the container
   - This ensures scripts work correctly even after `load_dotenv()` calls

3. **GPU Indexing**: GPU device numbers start at 0. For 8 GPUs, use `device=0-7`.

4. **Shared Memory**: The `--shm-size 16g` flag is important for PyTorch DataLoader with multiple workers.

5. **Read-Only Volumes**: The `:ro` flag on `/data/swara` ensures the dataset cannot be accidentally modified.

6. **Helper Script**: The `docker-cli-commands.sh` script automates all these commands and is the recommended approach.

## See Also

- [README.md](README.md) - Project overview and training guide
- [DGX-DEPLOYMENT.md](docs/DGX-DEPLOYMENT.md) - Complete deployment guide
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues and solutions

---

**Pro Tip**: Use the helper script (`docker-cli-commands.sh`) for the best experience. It handles all the complexity automatically.
