# Devcontainer Support Requirements

## Objective
This repository should support **devcontainers** for local development and DGX training workflows. The goal is to:
- Develop locally with a **local dataset copy**.
- Move the same repo to the DGX system with **minimal config changes**.
- Run training inside a container on DGX with a **consistent environment**.

## Requirements
- Provide a `devcontainer.json` with GPU support (CUDA runtime) and Python tooling.
- Keep configuration minimal and portable between local and DGX.
- Allow dataset path to be injected via environment variables or a small config file (not hard-coded).
- Ensure training scripts can run in-container with mounted dataset paths.

## Expected Outcome
- Local: run preprocessing and small-scale tests with local SWARA copy.
- DGX: run the same container image with mounted dataset and execute full training.

## Notes
- If the fine-tuning kit repo is forked, devcontainer configs should live in the fork and be kept lightweight.
