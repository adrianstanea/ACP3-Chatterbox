#!/bin/bash
# Post-create script for devcontainer
# This script sets up the development environment after container creation
# Can be run multiple times safely (idempotent where possible)

set -e  # Exit on error

echo "========================================"
echo "Running post-create setup..."
echo "========================================"

# Step 1: Upgrade pip and setuptools (Python 3.12 compatibility)
echo ""
echo "[1/5] Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel
echo "✓ pip, setuptools, and wheel upgraded"

# Step 2: Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
if [ -f requirements.txt ]; then
    # Install chatterbox-tts without dependencies (base image has compatible versions)
    # This avoids Python 3.12 incompatibility with old numpy versions
    pip install --no-cache-dir --no-deps chatterbox-tts

    # Install remaining dependencies normally
    pip install --no-cache-dir librosa soundfile tensorboard jiwer pesq pystoi

    echo "✓ Python dependencies installed"
else
    echo "⚠ requirements.txt not found, skipping"
fi

# Step 3: Initialize git submodules
echo ""
echo "[3/5] Initializing git submodules..."
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

# Step 4: Install vendor dependencies
echo ""
echo "[4/5] Installing vendor dependencies..."
if [ -f vendor/chatterbox-finetuning/requirements.txt ]; then
    pip install --no-cache-dir -r vendor/chatterbox-finetuning/requirements.txt
    echo "✓ Vendor dependencies installed"
else
    echo "⚠ Vendor requirements not found, skipping"
fi

# Step 5: Download pretrained models (optional - can be run later)
echo ""
echo "[5/5] Pretrained models setup..."
if [ -d vendor/chatterbox-finetuning ]; then
    echo "To download pretrained models (~3GB), run:"
    echo "  cd vendor/chatterbox-finetuning && python setup.py"
    echo "This is optional and can be run when needed."
else
    echo "⚠ Vendor directory not found, skipping"
fi

echo ""
echo "========================================"
echo "✓ Post-create setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Verify environment: ./scripts/check_environment.sh"
echo "  2. Download models: cd vendor/chatterbox-finetuning && python setup.py"
echo "  3. Run Task 5: python scripts/verify_tokenizer.py ..."
echo ""
