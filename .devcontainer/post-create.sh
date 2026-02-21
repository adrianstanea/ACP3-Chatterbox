#!/bin/bash
# Post-create script for devcontainer
# This script sets up the development environment after container creation
# Can be run multiple times safely (idempotent where possible)

set -e  # Exit on error

echo "========================================"
echo "Running post-create setup..."
echo "========================================"

# Step 1: Upgrade pip and install compatible setuptools (Python 3.12 compatibility)
echo ""
echo "[1/6] Upgrading pip and installing compatible setuptools..."
pip install --upgrade pip wheel
# Install setuptools 69.5.1 for pkg_resources (needed by Perth watermarker)
# See: https://github.com/resemble-ai/Perth/issues/7
pip install setuptools==69.5.1
echo "✓ pip, wheel, and setuptools 69.5.1 installed"

# Step 2: Install vendor dependencies first (exact versions)
echo ""
echo "[2/6] Installing vendor dependencies..."
if [ -f vendor/chatterbox-finetuning/requirements.txt ]; then
    # Install exact versions from vendor to avoid conflicts
    pip install --no-cache-dir -r vendor/chatterbox-finetuning/requirements.txt

    # Uninstall flash-attn if present (ABI compatibility issues with PyTorch 2.6.0)
    # Transformers will fall back to standard attention (slightly slower but stable)
    pip uninstall -y flash-attn 2>/dev/null || true

    echo "✓ Vendor dependencies installed"
else
    echo "⚠ Vendor requirements not found, skipping"
fi

# Step 2b: Install additional workspace dependencies
echo ""
echo "[2b/6] Installing workspace dependencies..."
if [ -f requirements.txt ]; then
    # Install remaining dependencies (jiwer, pesq, pystoi for evaluation)
    pip install --no-cache-dir jiwer pesq pystoi
    echo "✓ Workspace dependencies installed"
else
    echo "⚠ requirements.txt not found, skipping"
fi

# Step 3: Initialize git submodules
echo ""
echo "[3/6] Initializing git submodules..."
if [ -f .gitmodules ]; then
    # Check if we're in a worktree or regular repo
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git submodule update --init --recursive
        echo "✓ Git submodules initialized"
    else
        echo "⊘ Running in container - git submodules should be initialized on host"
        # Check if vendor directory exists (already initialized on host)
        if [ -d vendor/chatterbox-finetuning/.git ]; then
            echo "✓ Vendor submodule already present"
        else
            echo "⚠ Vendor submodule not found - run 'git submodule update --init' on host"
        fi
    fi
else
    echo "⚠ No git submodules found, skipping"
fi

# Step 4: Download pretrained models and extend tokenizer
echo ""
echo "[4/6] Pretrained models setup..."
if [ -d vendor/chatterbox-finetuning ]; then
    if [ ! -d vendor/chatterbox-finetuning/pretrained_models ]; then
        echo "Downloading pretrained models (~3GB, 5-10 minutes)..."
        cd vendor/chatterbox-finetuning
        python setup.py
        cd /workspace
        echo "✓ Pretrained models downloaded"
    else
        echo "✓ Pretrained models already present"
    fi

    # Extend tokenizer for Romanian
    echo ""
    echo "[4b/6] Extending tokenizer for Romanian..."
    if [ -f scripts/extend_tokenizer.py ]; then
        python scripts/extend_tokenizer.py --language romanian
        echo "✓ Tokenizer extended for Romanian (2454 → 2459 tokens)"
    else
        echo "⚠ Tokenizer extension script not found"
    fi
else
    echo "⚠ Vendor directory not found, skipping"
fi

echo ""
echo "[5/6] Fixing dataset symlinks for container paths..."
if [ -d /workspace/data/processed/MyTTSDataset/wavs ]; then
    # Recreate symlinks to point to container mount path /data/swara
    # (original symlinks point to host paths that don't exist in container)
    cd /workspace/data/processed/MyTTSDataset/wavs
    rm -f *.wav 2>/dev/null || true
    if [ -d /data/swara ]; then
        for file in /data/swara/*.wav; do
            ln -s "$file" . 2>/dev/null || true
        done
        echo "✓ Dataset symlinks fixed for container paths"
    else
        echo "⚠ SWARA dataset not mounted at /data/swara"
    fi
else
    echo "⊘ Dataset directory not found, skipping symlink fix"
fi

echo ""
echo "========================================"
echo "✓ Post-create setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Verify tokenizer: python scripts/verify_tokenizer.py \\"
echo "       vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \\"
echo "       data/processed/MyTTSDataset/metadata.csv"
echo "  2. Start training: cd vendor/chatterbox-finetuning && python train.py"
echo ""
