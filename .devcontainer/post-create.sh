#!/bin/bash
# Post-create script for devcontainer
# This script sets up the development environment after container creation
# Can be run multiple times safely (idempotent where possible)

set -e  # Exit on error

echo "========================================"
echo "Running post-create setup..."
echo "========================================"

# Step 1: Install Python dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "⚠ requirements.txt not found, skipping"
fi

# Step 2: Initialize git submodules
echo ""
echo "[2/4] Initializing git submodules..."
if [ -f .gitmodules ]; then
    git submodule update --init --recursive
    echo "✓ Git submodules initialized"
else
    echo "⚠ No git submodules found, skipping"
fi

# Step 3: Install vendor dependencies
echo ""
echo "[3/4] Installing vendor dependencies..."
if [ -f vendor/chatterbox-finetuning/requirements.txt ]; then
    pip install --no-cache-dir -r vendor/chatterbox-finetuning/requirements.txt
    echo "✓ Vendor dependencies installed"
else
    echo "⚠ Vendor requirements not found, skipping"
fi

# Step 4: Download pretrained models (optional - can be run later)
echo ""
echo "[4/4] Pretrained models setup..."
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
