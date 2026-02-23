FROM nvcr.io/nvidia/pytorch:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
