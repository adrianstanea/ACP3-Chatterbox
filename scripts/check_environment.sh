#!/bin/bash
#
# Environment Check Script for Tokenizer Verification
#
# This script checks if the environment is ready for tokenizer verification
# and provides helpful feedback on what's missing.
#

set -e

echo "======================================================================"
echo "Environment Check for Tokenizer Verification"
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_READY=true

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ALL_READY=false
    fi
}

check_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Check Docker daemon
echo "1. Docker Daemon Status"
if systemctl is-active --quiet docker 2>/dev/null || service docker status >/dev/null 2>&1; then
    check_status 0 "Docker daemon is running"
else
    check_status 1 "Docker daemon is not running"
    echo "   Fix: sudo service docker start"
fi
echo ""

# 2. Check .env file
echo "2. Environment Configuration"
if [ -f ".env" ]; then
    check_status 0 ".env file exists"

    # Check required variables
    if grep -q "^SWARA_PATH=" .env; then
        SWARA_PATH=$(grep "^SWARA_PATH=" .env | cut -d'=' -f2)
        echo "   SWARA_PATH: $SWARA_PATH"
    else
        check_warning "SWARA_PATH not set in .env"
    fi

    if grep -q "^OUTPUT_PATH=" .env; then
        OUTPUT_PATH=$(grep "^OUTPUT_PATH=" .env | cut -d'=' -f2)
        echo "   OUTPUT_PATH: $OUTPUT_PATH"
    else
        check_warning "OUTPUT_PATH not set in .env"
    fi
else
    check_status 1 ".env file does not exist"
    echo "   Fix: cp .env.example .env && nano .env"
fi
echo ""

# 3. Check verification script
echo "3. Verification Script"
if [ -f "scripts/verify_tokenizer.py" ]; then
    check_status 0 "verify_tokenizer.py exists"
    if [ -x "scripts/verify_tokenizer.py" ]; then
        check_status 0 "verify_tokenizer.py is executable"
    else
        check_status 1 "verify_tokenizer.py is not executable"
        echo "   Fix: chmod +x scripts/verify_tokenizer.py"
    fi
else
    check_status 1 "verify_tokenizer.py not found"
fi
echo ""

# 4. Check metadata file (converted dataset)
echo "4. Dataset Status"
if [ -f "data/processed/MyTTSDataset/metadata.csv" ]; then
    check_status 0 "metadata.csv exists"
    LINE_COUNT=$(wc -l < data/processed/MyTTSDataset/metadata.csv)
    echo "   Lines in metadata: $LINE_COUNT"
else
    check_status 1 "metadata.csv not found"
    echo "   Note: Run convert_swara_to_ljspeech.py first"
fi
echo ""

# 5. Check if container is running
echo "5. Docker Container Status"
if docker compose ps | grep -q "chatterbox.*running"; then
    check_status 0 "Container is running"
else
    check_status 1 "Container is not running"
    echo "   Fix: docker compose up -d chatterbox"
fi
echo ""

# 6. Check if setup.py has been run (inside container)
echo "6. Pretrained Models Status"
if [ -d "vendor/chatterbox-finetuning/pretrained_models" ]; then
    check_status 0 "pretrained_models directory exists"

    if [ -f "vendor/chatterbox-finetuning/pretrained_models/tokenizer.json" ]; then
        check_status 0 "tokenizer.json exists"
        FILE_SIZE=$(stat -f%z "vendor/chatterbox-finetuning/pretrained_models/tokenizer.json" 2>/dev/null || stat -c%s "vendor/chatterbox-finetuning/pretrained_models/tokenizer.json" 2>/dev/null)
        echo "   Tokenizer size: $FILE_SIZE bytes"
    else
        check_status 1 "tokenizer.json not found"
        echo "   Note: Run setup.py inside container"
    fi

    # Count model files
    MODEL_COUNT=$(find vendor/chatterbox-finetuning/pretrained_models -type f | wc -l)
    echo "   Model files found: $MODEL_COUNT"
    if [ "$MODEL_COUNT" -lt 5 ]; then
        check_warning "Expected 5 model files, found $MODEL_COUNT"
        echo "   Expected: tokenizer.json, ve.safetensors, t3_cfg.safetensors, s3gen.safetensors, conds.pt"
    fi
else
    check_status 1 "pretrained_models directory does not exist"
    echo "   Note: Run setup.py inside container to download models"
fi
echo ""

# Summary
echo "======================================================================"
echo "Summary"
echo "======================================================================"

if [ "$ALL_READY" = true ]; then
    echo -e "${GREEN}✓ Environment is ready for tokenizer verification!${NC}"
    echo ""
    echo "Next step:"
    echo "  docker compose exec chatterbox bash"
    echo "  cd /workspace"
    echo "  python scripts/verify_tokenizer.py \\"
    echo "      vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \\"
    echo "      data/processed/MyTTSDataset/metadata.csv"
else
    echo -e "${RED}✗ Environment is not ready. Please address the issues above.${NC}"
    echo ""
    echo "Quick setup:"
    echo "  1. sudo service docker start"
    echo "  2. docker compose up -d chatterbox"
    echo "  3. docker compose exec chatterbox bash"
    echo "  4. Inside container: cd /workspace && pip install -r requirements.txt"
    echo "  5. Inside container: cd vendor/chatterbox-finetuning && python setup.py"
    echo "  6. Exit container and re-run this script"
fi

echo "======================================================================"
echo ""

# Documentation links
echo "Documentation:"
echo "  - docs/TOKENIZER-SETUP-GUIDE.md (quick reference)"
echo "  - docs/tokenizer-verification.md (full documentation)"
echo "  - docs/EXECUTION-STATUS.md (current status)"
echo ""
