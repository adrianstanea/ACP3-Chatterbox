#!/bin/bash
# Docker CLI Alternative to docker-compose
# Use these commands when docker-compose is not available

set -e  # Exit on error

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using defaults."
    HOST_SWARA_PATH=${HOST_SWARA_PATH:-/path/to/swara}
    HOST_OUTPUT_PATH=${HOST_OUTPUT_PATH:-/path/to/output}
    NUM_GPUS=${NUM_GPUS:-1}
fi

# Image and container names
IMAGE_NAME="chatterbox:latest"
CONTAINER_NAME="chatterbox"

# Function: Build the Docker image
build() {
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" .
    echo "✓ Build complete"
}

# Function: Run the container
run() {
    echo "Starting container: $CONTAINER_NAME"

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container $CONTAINER_NAME already exists. Use 'start' or 'remove' it first."
        exit 1
    fi

    echo "Using host paths for mounting:"
    echo "  SWARA: ${HOST_SWARA_PATH} -> /data/swara"
    echo "  OUTPUT: ${HOST_OUTPUT_PATH} -> /data/output"

    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --shm-size 16g \
        -v "$(pwd):/workspace" \
        -v "${HOST_SWARA_PATH}:/data/swara:ro" \
        -v "${HOST_OUTPUT_PATH}:/data/output" \
        -w /workspace \
        --env-file .env \
        -it \
        "$IMAGE_NAME"

    echo "✓ Container started: $CONTAINER_NAME"
    echo "  To enter: docker exec -it $CONTAINER_NAME bash"
    echo "  Inside container:"
    echo "    SWARA_PATH=/data/swara"
    echo "    OUTPUT_PATH=/data/output"
}

# Function: Start existing container
start() {
    echo "Starting existing container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME"
    echo "✓ Container started"
}

# Function: Stop container
stop() {
    echo "Stopping container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    echo "✓ Container stopped"
}

# Function: Execute bash in container
exec_bash() {
    echo "Entering container: $CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" bash
}

# Function: Run post-create setup
setup() {
    echo "Running post-create setup..."
    docker exec "$CONTAINER_NAME" bash /workspace/.devcontainer/post-create.sh
    echo "✓ Setup complete"
}

# Function: Remove container
remove() {
    echo "Removing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo "✓ Container removed"
}

# Function: Show logs
logs() {
    docker logs -f "$CONTAINER_NAME"
}

# Function: Show usage
usage() {
    cat << EOF
Docker CLI Alternative to docker-compose

Usage: $0 <command>

Commands:
  build        Build the Docker image
  run          Create and start a new container
  start        Start existing container
  stop         Stop running container
  exec         Execute bash in container
  setup        Run post-create setup script
  remove       Remove container (use before 'run' again)
  logs         Show container logs

  build-and-run    Build image and run container
  full-setup       Build, run, and setup (complete initialization)

Environment Variables (from .env):
  HOST_SWARA_PATH=$HOST_SWARA_PATH
  HOST_OUTPUT_PATH=$HOST_OUTPUT_PATH
  NUM_GPUS=$NUM_GPUS

Examples:
  # First time setup
  $0 full-setup

  # Or step by step:
  $0 build
  $0 run
  $0 setup
  $0 exec

  # Regular usage:
  $0 start
  $0 exec

EOF
}

# Function: Build and run
build_and_run() {
    build
    run
}

# Function: Full setup (build, run, setup)
full_setup() {
    build
    run
    setup
    echo ""
    echo "✓ Full setup complete!"
    echo "  Enter container: $0 exec"
}

# Main command dispatcher
case "${1:-}" in
    build)
        build
        ;;
    run)
        run
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    exec)
        exec_bash
        ;;
    setup)
        setup
        ;;
    remove)
        remove
        ;;
    logs)
        logs
        ;;
    build-and-run)
        build_and_run
        ;;
    full-setup)
        full_setup
        ;;
    *)
        usage
        exit 1
        ;;
esac
