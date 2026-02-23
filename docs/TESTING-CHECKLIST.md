# Docker Setup Testing Checklist

This checklist validates the Docker setup and Task 5 completion when Docker becomes available.

## Pre-Testing Requirements

- [ ] Docker Engine installed and running
- [ ] NVIDIA Container Toolkit installed
- [ ] `nvidia-smi` works on host
- [ ] `.env` file configured with valid paths
- [ ] SWARA dataset accessible at `$SWARA_PATH`

## Quick Validation

```bash
# Check Docker
docker --version
docker compose version

# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check environment
cat .env
ls -la $SWARA_PATH/metadata_SWARA1.0_text.csv
```

Expected:
- Docker 24.0+
- Docker Compose 2.0+
- nvidia-smi output showing GPU(s)
- .env exists with valid paths
- Dataset metadata file exists

## Automated Testing

### Option 1: Full Automated Test

```bash
./scripts/docker_setup_and_verify.sh
```

**Expected behavior:**
1. Builds image (10-15 min first time)
2. Starts container
3. Runs environment check
4. Downloads models (~3GB, 5-10 min)
5. Verifies tokenizer
6. Prints success message

**Success criteria:**
- All steps complete without errors
- Final message shows "Docker setup completed successfully!"
- Container remains running

### Option 2: Step-by-Step Manual Test

Follow this checklist:

## Manual Testing Checklist

### 1. Build Image

```bash
docker compose build
```

**Expected:**
- [ ] Base image pulls successfully (~10GB)
- [ ] System dependencies install (espeak-ng, ffmpeg, git)
- [ ] Python dependencies install from requirements.txt
- [ ] Build completes without errors
- [ ] Image appears in `docker images`

**Verify:**
```bash
docker images | grep chatterbox
```

**Timing:** 10-15 minutes (first time), 1-2 minutes (cached)

### 2. Start Container

```bash
docker compose up -d
```

**Expected:**
- [ ] Container starts in detached mode
- [ ] No error messages
- [ ] Container shows as "Up" in `docker compose ps`

**Verify:**
```bash
docker compose ps
```

**Expected output:**
```
NAME                COMMAND             SERVICE             STATUS
chatterbox          "/bin/bash"         chatterbox          Up
```

### 3. GPU Access

```bash
docker compose exec chatterbox nvidia-smi
```

**Expected:**
- [ ] nvidia-smi runs without errors
- [ ] Correct number of GPUs shown (matches NUM_GPUS in .env)
- [ ] GPUs have available memory
- [ ] CUDA version shown (12.3+)

### 4. Environment Verification

```bash
docker compose exec chatterbox ./scripts/check_environment.sh
```

**Expected checks:**
- [ ] CUDA available: ✓
- [ ] GPU count matches NUM_GPUS
- [ ] PyTorch can access GPU(s)
- [ ] Python packages installed:
  - [ ] chatterbox
  - [ ] espeak_phonemizer
  - [ ] librosa
  - [ ] soundfile
  - [ ] tensorboard
- [ ] System commands available:
  - [ ] espeak-ng
  - [ ] ffmpeg
- [ ] Dataset path accessible
- [ ] Metadata file found
- [ ] Directory structure correct

### 5. Download Pretrained Models

```bash
docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && python setup.py"
```

**Expected:**
- [ ] Creates `pretrained_models/` directory
- [ ] Downloads files with progress bars:
  - [ ] ve.safetensors (~500MB)
  - [ ] t3_turbo_v1.safetensors (~1.2GB)
  - [ ] s3gen_meanflow.safetensors (~500MB)
  - [ ] conds.pt
  - [ ] vocab.json
  - [ ] added_tokens.json
  - [ ] special_tokens_map.json
  - [ ] tokenizer_config.json
  - [ ] merges.txt
  - [ ] grapheme_mtl_merged_expanded_v1.json
- [ ] Merges tokenizer with custom vocab
- [ ] Reports final vocab size
- [ ] Prints "INSTALLATION COMPLETE" message
- [ ] No download errors

**Record the vocab size:**
```
New vocab size: ___________
```

**Verify files:**
```bash
docker compose exec chatterbox ls -lh vendor/chatterbox-finetuning/pretrained_models/
```

**Expected:**
- [ ] All model files present
- [ ] tokenizer.json exists
- [ ] Total size ~2-3GB

### 6. Find Tokenizer

```bash
docker compose exec chatterbox find vendor/chatterbox-finetuning -name "tokenizer.json" -type f
```

**Expected:**
- [ ] Finds `vendor/chatterbox-finetuning/pretrained_models/tokenizer.json`
- [ ] Only one match

### 7. Verify Tokenizer (Task 5)

```bash
docker compose exec chatterbox python scripts/verify_tokenizer.py \
  vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
  data/processed/MyTTSDataset/metadata.csv
```

**Expected output sections:**

#### A. Tokenizer Info
- [ ] Base tokenizer size: 50257
- [ ] Extended tokenizer size: (should match setup.py output)
- [ ] New tokens added: (difference)

#### B. Romanian Character Coverage
- [ ] ă - supported
- [ ] â - supported
- [ ] î - supported
- [ ] ș - supported
- [ ] ț - supported

#### C. Metadata Analysis
- [ ] Total entries counted
- [ ] Successfully tokenized count
- [ ] Issues count (should be 0 or low)

#### D. Sample Verification
- [ ] Shows sample Romanian text
- [ ] Shows tokenization result
- [ ] No encoding errors

**Record results:**
```
Tokenizer size: ___________
Romanian chars: OK / FAILED
Metadata entries: ___________
Issues found: ___________
```

### 8. Config Update (if needed)

If tokenizer was extended, update config:

```bash
docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && grep new_vocab_size src/config.py"
```

**Expected:**
- [ ] Shows current `new_vocab_size` value
- [ ] If different from setup.py output, update needed

**If update needed:**
```bash
docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && nano src/config.py"
```

Update line to match setup.py output:
```python
new_vocab_size: int = [SIZE_FROM_SETUP]
```

Verify:
```bash
docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && grep new_vocab_size src/config.py"
```

- [ ] Value matches setup.py output

### 9. Volume Mount Verification

Test that code changes on host are visible in container:

```bash
# On host
echo "# Test comment" >> test_mount.py

# In container
docker compose exec chatterbox ls -la test_mount.py
docker compose exec chatterbox cat test_mount.py

# Cleanup
rm test_mount.py
```

**Expected:**
- [ ] File visible in container
- [ ] Content matches
- [ ] Delete on host removes from container

### 10. Persistence Test

Test that output persists across container restarts:

```bash
# Create test file
docker compose exec chatterbox bash -c "echo 'test' > /data/output/test.txt"

# Restart container
docker compose down
docker compose up -d

# Check file still exists
docker compose exec chatterbox cat /data/output/test.txt

# Cleanup
docker compose exec chatterbox rm /data/output/test.txt
```

**Expected:**
- [ ] File persists across restart
- [ ] Content unchanged

## Integration Tests

### Test 1: Python Environment

```bash
docker compose exec chatterbox python -c "
import torch
import chatterbox
import librosa
import soundfile
import tensorboard
print('All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

**Expected:**
- [ ] All imports succeed
- [ ] PyTorch version shown
- [ ] CUDA available: True
- [ ] GPU count matches NUM_GPUS

### Test 2: Audio Processing

```bash
docker compose exec chatterbox python -c "
import numpy as np
import soundfile as sf
# Create test audio
audio = np.random.randn(16000).astype(np.float32)
sf.write('/tmp/test.wav', audio, 16000)
# Read it back
audio2, sr = sf.read('/tmp/test.wav')
print(f'Audio shape: {audio2.shape}')
print(f'Sample rate: {sr}')
print('Audio I/O working')
"
```

**Expected:**
- [ ] No errors
- [ ] Audio shape: (16000,)
- [ ] Sample rate: 16000
- [ ] "Audio I/O working" printed

### Test 3: Phonemization

```bash
docker compose exec chatterbox python -c "
from espeak_phonemizer import Phonemizer
ph = Phonemizer(language='ro')
result = ph('Salut, cum ești?')
print(f'Romanian phonemization: {result}')
print('Phonemization working')
"
```

**Expected:**
- [ ] No errors
- [ ] Phonemization result shown
- [ ] "Phonemization working" printed

## Performance Tests

### GPU Memory Test

```bash
docker compose exec chatterbox python -c "
import torch
print('Testing GPU memory allocation...')
for i in range(5):
    tensor = torch.randn(1000, 1000).cuda()
    print(f'Allocated {i+1}GB on GPU')
print('GPU memory test passed')
"
```

**Expected:**
- [ ] No CUDA out of memory errors
- [ ] All allocations succeed

### Data Loading Test

```bash
docker compose exec chatterbox python -c "
from torch.utils.data import DataLoader, TensorDataset
import torch
dataset = TensorDataset(torch.randn(100, 10))
loader = DataLoader(dataset, batch_size=10, num_workers=4)
for batch in loader:
    pass
print('DataLoader test passed')
"
```

**Expected:**
- [ ] No bus errors (indicates shm_size adequate)
- [ ] "DataLoader test passed" printed

## Cleanup Tests

### Test Container Lifecycle

```bash
# Stop container
docker compose down
docker compose ps  # Should show nothing

# Start again
docker compose up -d
docker compose ps  # Should show running

# Logs
docker compose logs | head -20

# Stop
docker compose down
```

**Expected:**
- [ ] Clean shutdown
- [ ] Clean startup
- [ ] Logs show no errors

## Success Criteria

All of the following must pass:

- [ ] Image builds successfully
- [ ] Container starts and runs
- [ ] GPU access works
- [ ] All environment checks pass
- [ ] Models download successfully
- [ ] Tokenizer verification completes
- [ ] Romanian characters supported
- [ ] No critical errors in logs
- [ ] Volume mounts work correctly
- [ ] Container survives restart

## Task 5 Completion Criteria

Specifically for Task 5 (Verify and Extend Tokenizer):

- [ ] Tokenizer located at correct path
- [ ] Romanian diacritics (ă, â, î, ș, ț) fully supported
- [ ] Metadata can be tokenized without errors
- [ ] Vocab size documented
- [ ] config.py updated if needed
- [ ] Verification report saved/documented

## Documentation Checklist

After testing:

- [ ] Record final vocab size
- [ ] Note any issues encountered
- [ ] Document workarounds used
- [ ] Update .env with working paths
- [ ] Save verification output
- [ ] Commit all changes

## Troubleshooting Notes

Record any issues and solutions here:

| Issue | Solution | Notes |
|-------|----------|-------|
| | | |
| | | |

## Sign-off

Date tested: ___________
Tested by: ___________
Docker version: ___________
GPU model: ___________
All tests passed: YES / NO

Notes:
```
[Any additional notes about the setup]
```

## Next Steps After Successful Testing

1. [ ] Commit Dockerfile and docker-compose.yml
2. [ ] Commit documentation
3. [ ] Tag this working configuration
4. [ ] Proceed to Task 6 (preprocessing)
5. [ ] Update project README with Docker instructions

---

**This checklist should result in a fully working Docker environment ready for Task 6 and beyond.**
