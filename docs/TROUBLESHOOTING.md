# Troubleshooting Guide

This document captures solutions to issues encountered during Romanian Chatterbox adaptation.

## Environment Setup Issues

### 1. Perth Watermarker - "No module named 'pkg_resources'"

**Problem**: After installing `resemble-perth`, importing fails with:
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause**: Perth package uses deprecated `pkg_resources` module which was removed from Python 3.12+ setuptools.

**Solution**: Install setuptools 69.5.1 which still includes `pkg_resources`:
```bash
pip install setuptools==69.5.1
```

**Reference**: [Perth Issue #7](https://github.com/resemble-ai/Perth/issues/7)

**Status**: Fixed in `.devcontainer/post-create.sh` (Step 1)

---

### 2. Flash-Attention ABI Compatibility

**Problem**: Training fails with:
```
ImportError: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Cause**: Pre-compiled flash-attention CUDA extensions have ABI incompatibility with PyTorch 2.6.0 in the container.

**Solution**: Uninstall flash-attention entirely:
```bash
pip uninstall -y flash-attn
```

**Impact**: Transformers library falls back to standard attention implementation. Slightly slower but fully functional.

**Note**: Rebuilding from source didn't resolve the issue due to complex C++ ABI incompatibilities.

**Status**: Fixed in `.devcontainer/post-create.sh` (Step 2)

---

### 3. Dataset Symlinks Point to Host Paths

**Problem**: Preprocessing fails with "No such file or directory" when loading WAV files.

**Cause**: Symlinks in `data/processed/MyTTSDataset/wavs/` point to host filesystem paths (`/home/astanea/data/SWARA1.0_22k_noSil/`) which don't exist inside the container.

**Solution**: Recreate symlinks to point to the mounted dataset path inside container:
```bash
cd /workspace/data/processed/MyTTSDataset/wavs
rm *.wav
for file in /data/swara/*.wav; do
    ln -s "$file" .
done
```

**Root Cause**: Dataset conversion script (`scripts/convert_swara_to_ljspeech.py`) creates symlinks using absolute host paths instead of relative paths.

**Status**:
- **Immediate fix**: Manual symlink recreation in container
- **Automated**: Added to `.devcontainer/post-create.sh` (Step 5)
- **TODO**: Update `convert_swara_to_ljspeech.py` to use relative symlinks

---

## PyTorch Version Compatibility

**Base Image**: `nvcr.io/nvidia/pytorch:24.11-py3`
- PyTorch: 2.6.0
- Python: 3.12
- CUDA: 12.4

**Key Dependencies** (from vendor/chatterbox-finetuning/requirements.txt):
- torch==2.6.0 ✅
- transformers==4.46.3 ✅
- chatterbox-tts==0.1.2 ✅
- diffusers==0.29.0 ✅

**Known Issues**:
- flash-attention: Pre-built binaries incompatible (removed, see #2)
- setuptools 82.0.0: Missing pkg_resources (downgraded to 69.5.1, see #1)

---

## SSL Certificate Issues (Corporate Networks)

**Problem**: SSL verification errors when downloading packages/models in corporate networks with self-signed certificates.

**Solution**: Copy organization's CA certificate bundle to image during build:
```dockerfile
# Optional - build succeeds without certs/ directory
COPY cert[s]/ /tmp/certs/
RUN if [ -f /tmp/certs/nscacert_combined.crt ]; then \
    cp /tmp/certs/nscacert_combined.crt /usr/local/share/ca-certificates/nscacert_combined.crt && \
    update-ca-certificates; \
    fi
```

**CRITICAL**: Add `certs/` to `.gitignore` - never commit certificates!

**Status**: Documented in `DOCKER-SETUP.md`

---

## Performance Notes

### Preprocessing Speed
- **Observed**: ~7 files/second
- **Total files**: 21,304
- **Estimated time**: 50-60 minutes
- **Hardware**: Single GPU (container config: NUM_GPUS=1)

**Bottleneck**: Voice encoder + S3 tokenizer inference on GPU

### Disk Space Requirements
- Pretrained models: ~3 GB
- Preprocessed .pt files: ~15-20 GB (estimated)
- SWARA dataset: ~2 GB
- **Total**: ~25 GB minimum

---

## Quick Fixes Summary

```bash
# Fix Perth watermarker
pip install setuptools==69.5.1

# Remove problematic flash-attention
pip uninstall -y flash-attn

# Fix dataset symlinks (run inside container)
cd /workspace/data/processed/MyTTSDataset/wavs
rm *.wav
for file in /data/swara/*.wav; do ln -s "$file" .; done
```

---

## Getting Help

1. **Check logs**: All preprocessing logs visible in train.py output
2. **Container logs**: `docker compose logs chatterbox`
3. **Interactive debugging**: `docker compose exec chatterbox bash`
4. **Python environment**: `pip list | grep -E 'torch|flash|perth|setuptools'`

---

**Last Updated**: 2026-02-21
**Applies to**: PyTorch 24.11 container, Python 3.12, Chatterbox Multilingual fine-tuning
