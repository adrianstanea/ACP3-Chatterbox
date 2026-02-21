# Clean base image for Romanian Chatterbox fine-tuning
# This image contains only system dependencies
# Python dependencies and project setup are handled by post-create.sh

FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install system dependencies only
# These rarely change and should be baked into the image
RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Set working directory (workspace will be mounted here)
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (interactive bash)
CMD ["/bin/bash"]
