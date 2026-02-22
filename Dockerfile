# Clean base image for Romanian Chatterbox fine-tuning
# This image contains only system dependencies
# Python dependencies and project setup are handled by post-create.sh
# Using 24.11 for PyTorch 2.6.0 compatibility

FROM nvcr.io/nvidia/pytorch:24.11-py3

# Handle CA certificates (optional - build succeeds without certs/)
# COPY cert[s]/ /tmp/certs/
# RUN if [ -f /tmp/certs/nscacert_combined.crt ]; then \
#     cp /tmp/certs/nscacert_combined.crt /usr/local/share/ca-certificates/nscacert_combined.crt && \
#     update-ca-certificates; \
#     fi
ENV DEBIAN_FRONTEND=noninteractive

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
