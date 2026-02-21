FROM nvcr.io/nvidia/pytorch:26.01-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (interactive bash)
CMD ["/bin/bash"]
