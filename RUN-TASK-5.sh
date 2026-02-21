#!/bin/bash
# Execute Task 5: Tokenizer Verification
# Run this script to complete Task 5 validation

set -e

echo "==================================================================="
echo "  TASK 5: Tokenizer Verification - Execution Script"
echo "==================================================================="
echo ""

# # Step 1: Start Docker
# echo "Step 1: Starting Docker daemon..."
# echo "  Running: sudo service docker start"
# sudo service docker start || sudo systemctl start docker
# echo "  ✓ Docker daemon started"
# echo ""

# Wait for Docker to be ready
# echo "  Waiting for Docker to be ready..."
# sleep 3
# echo ""

# Step 2: Build and start container
echo "Step 2: Building and starting container..."
cd "$(dirname "$0")"
./scripts/docker_setup_and_verify.sh
echo ""

echo "==================================================================="
echo "  Task 5 execution complete!"
echo "  Check output above for tokenizer verification results."
echo "==================================================================="
