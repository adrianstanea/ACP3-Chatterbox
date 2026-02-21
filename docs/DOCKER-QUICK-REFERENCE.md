# Docker Quick Reference - Chatterbox Romanian

Quick command reference for common Docker operations.

## Container Lifecycle

```bash
# Build image from Dockerfile
docker compose build

# Start container (detached)
docker compose up -d

# Stop container
docker compose down

# Restart container
docker compose restart

# View container status
docker compose ps

# View logs
docker compose logs -f
```

## Entering the Container

```bash
# Main method - interactive bash
docker compose exec chatterbox bash

# Alternative - run single command
docker compose exec chatterbox python scripts/check_environment.sh

# Run as different user
docker compose exec -u root chatterbox bash
```

## File Operations

```bash
# Copy file FROM container TO host
docker compose cp chatterbox:/workspace/output.txt ./

# Copy file FROM host TO container
docker compose cp ./input.txt chatterbox:/workspace/

# View files without entering
docker compose exec chatterbox ls -la /workspace
```

## Debugging

```bash
# View container logs
docker compose logs chatterbox

# Follow logs in real-time
docker compose logs -f chatterbox

# Check container processes
docker compose exec chatterbox ps aux

# Check GPU access
docker compose exec chatterbox nvidia-smi

# Check disk space
docker compose exec chatterbox df -h

# Check environment variables
docker compose exec chatterbox env
```

## Resource Management

```bash
# View resource usage
docker stats

# Clean up stopped containers
docker system prune

# Remove all images and cache (WARNING: destructive)
docker system prune -a

# View disk usage
docker system df
```

## Rebuilding

```bash
# Rebuild with cache
docker compose build

# Rebuild without cache (clean build)
docker compose build --no-cache

# Rebuild and restart
docker compose up -d --build

# Pull latest base image
docker compose pull
docker compose build --no-cache
```

## Volume Management

```bash
# List volumes
docker volume ls

# Inspect specific volume
docker volume inspect romanian-adaptation_data

# Clean unused volumes (WARNING: data loss)
docker volume prune
```

## Network Troubleshooting

```bash
# Test network from container
docker compose exec chatterbox ping google.com
docker compose exec chatterbox ping huggingface.co

# Check DNS
docker compose exec chatterbox cat /etc/resolv.conf

# View network configuration
docker network ls
docker network inspect romanian-adaptation_default
```

## Multi-GPU Configuration

```bash
# Check available GPUs
nvidia-smi

# Edit .env file for GPU count
echo "NUM_GPUS=4" >> .env

# Restart with new GPU count
docker compose down
docker compose up -d

# Verify GPU access in container
docker compose exec chatterbox nvidia-smi
```

## Common Workflows

### First-Time Setup
```bash
# 1. Build container
docker compose build

# 2. Start container
docker compose up -d

# 3. Enter container
docker compose exec chatterbox bash

# 4. Run environment check
./scripts/check_environment.sh

# 5. Download models
cd vendor/chatterbox-finetuning
python setup.py
```

### Daily Development
```bash
# Start container (if not running)
docker compose up -d

# Enter for work
docker compose exec chatterbox bash

# When done, keep running or stop
docker compose down  # or leave running
```

### Troubleshooting Issues
```bash
# Check logs
docker compose logs -f

# Restart container
docker compose restart

# Clean rebuild
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Performance Testing
```bash
# Monitor GPU during training
watch -n 1 nvidia-smi

# Monitor container resources
docker stats chatterbox

# Check disk I/O
docker compose exec chatterbox iostat -x 1
```

## Environment Variables

View all environment variables from `.env`:

```bash
docker compose config
```

Override environment variable temporarily:

```bash
NUM_GPUS=2 docker compose up -d
```

## Cleanup Commands

```bash
# Stop and remove container (keeps volumes)
docker compose down

# Stop, remove container AND volumes (WARNING: data loss)
docker compose down -v

# Remove dangling images
docker image prune

# Full cleanup (WARNING: removes all unused Docker data)
docker system prune -a --volumes
```

## Tips

1. **Always check container is running** before exec:
   ```bash
   docker compose ps
   ```

2. **Keep container running** during development - volume mounts let you edit on host.

3. **Rebuild only when needed**:
   - Changed `Dockerfile` → rebuild
   - Changed `requirements.txt` → rebuild
   - Changed Python code → NO rebuild needed (volume mount)

4. **Save work before `down -v`** - this removes volumes!

5. **Use logs for debugging**:
   ```bash
   docker compose logs -f chatterbox | tee debug.log
   ```

6. **Test GPU access immediately**:
   ```bash
   docker compose exec chatterbox nvidia-smi
   ```

## Emergency Recovery

### Container won't start
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose logs -f
```

### Out of disk space
```bash
docker system prune -a
docker volume prune
```

### Permission issues
```bash
# Check volume mount permissions
ls -la /data/swara
ls -la /data/output

# Fix if needed (be careful with production data)
sudo chown -R $(id -u):$(id -g) /data/output
```

### GPU not detected
```bash
# Check host GPU
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

---

**Last Updated**: 2026-02-21
