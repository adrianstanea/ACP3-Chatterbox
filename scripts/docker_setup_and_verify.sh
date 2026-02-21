#!/bin/bash
# docker_setup_and_verify.sh
# Complete Docker setup and Task 5 verification workflow
# This script builds the container and runs all verification steps

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    log_error "docker-compose.yml not found. Please run from workspace root."
    exit 1
fi

# Check if .env exists
if [[ ! -f ".env" ]]; then
    log_warning ".env not found. Creating from .env.example..."
    cp .env.example .env
    log_info "Please edit .env with your actual paths before continuing."
    exit 1
fi

log_info "Starting Docker setup and verification workflow..."
echo ""

# Step 1: Build Docker image
log_info "Step 1/6: Building Docker image (this may take 10-15 minutes on first run)..."
if docker compose build; then
    log_success "Docker image built successfully"
else
    log_error "Docker build failed"
    exit 1
fi
echo ""

# Step 2: Start container
log_info "Step 2/6: Starting container..."
if docker compose up -d; then
    log_success "Container started"
else
    log_error "Failed to start container"
    exit 1
fi

# Wait for container to be ready
sleep 3
echo ""

# Step 3: Verify environment
log_info "Step 3/6: Verifying environment..."
if docker compose exec chatterbox ./scripts/check_environment.sh; then
    log_success "Environment verification passed"
else
    log_error "Environment verification failed"
    docker compose logs
    exit 1
fi
echo ""

# Step 4: Download pretrained models
log_info "Step 4/6: Downloading pretrained models..."
log_warning "This will download ~3GB of data and may take 5-10 minutes..."
if docker compose exec chatterbox bash -c "cd vendor/chatterbox-finetuning && python setup.py"; then
    log_success "Models downloaded successfully"
else
    log_error "Model download failed"
    exit 1
fi
echo ""

# Step 5: Find tokenizer
log_info "Step 5/6: Locating tokenizer..."
TOKENIZER_PATH=$(docker compose exec chatterbox find vendor/chatterbox-finetuning -name "tokenizer.json" -type f | tr -d '\r')
if [[ -n "$TOKENIZER_PATH" ]]; then
    log_success "Tokenizer found at: $TOKENIZER_PATH"
else
    log_error "Tokenizer not found"
    exit 1
fi
echo ""

# Step 6: Run tokenizer verification (Task 5)
log_info "Step 6/6: Running tokenizer verification (Task 5)..."
if docker compose exec chatterbox python scripts/verify_tokenizer.py \
    "$TOKENIZER_PATH" \
    "data/processed/MyTTSDataset/metadata.csv"; then
    log_success "Tokenizer verification completed"
else
    log_error "Tokenizer verification failed"
    exit 1
fi
echo ""

# Final summary
log_success "=================================="
log_success "Docker setup completed successfully!"
log_success "=================================="
echo ""
log_info "Next steps:"
echo "  1. Review the tokenizer verification output above"
echo "  2. If vocab extension is needed, update src/config.py with new vocab size"
echo "  3. Enter container: docker compose exec chatterbox bash"
echo "  4. Proceed to Task 6 (preprocessing)"
echo ""
log_info "Container is running. Use 'docker compose down' to stop."
