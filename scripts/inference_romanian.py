#!/usr/bin/env python3
"""
Romanian Chatterbox TTS Inference Script

Synthesizes Romanian speech using the finetuned Chatterbox model.
Supports single text and batch processing modes.

Usage:
    # Single text
    python scripts/inference_romanian.py --text "Bună ziua" --reference ref.wav --output out.wav

    # Batch mode
    python scripts/inference_romanian.py --batch input.csv --reference ref.wav --output-dir outputs/
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from tqdm import tqdm

# Add vendor to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "chatterbox-finetuning"))

from src.romanian_preprocessor import preprocess_and_normalize
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference-romanian")


class RomanianTTSEngine:
    """Romanian TTS engine using finetuned Chatterbox checkpoint."""

    VOCAB_SIZE = 2454  # Original vocab, no extension
    SAMPLE_RATE = 24000

    def __init__(
        self,
        checkpoint_dir: Path,
        pretrained_dir: Path,
        device: str = "cuda",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pretrained_dir = Path(pretrained_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.engine = None

        logger.info(f"Initializing RomanianTTSEngine on {self.device}")

    def load_model(self):
        """Load pretrained Chatterbox and replace T3 with finetuned weights."""
        logger.info(f"Loading base model from {self.pretrained_dir}")

        # Load base Chatterbox (on CPU first to manage memory)
        self.engine = ChatterboxTTS.from_local(str(self.pretrained_dir), device="cpu")

        # Create new T3 with original vocab size
        logger.info(f"Creating T3 with vocab size {self.VOCAB_SIZE}")
        t3_config = self.engine.t3.hp
        t3_config.text_tokens_dict_size = self.VOCAB_SIZE
        new_t3 = T3(hp=t3_config)

        # Load finetuned weights
        weights_path = self.checkpoint_dir / "model.safetensors"
        logger.info(f"Loading finetuned weights from {weights_path}")
        state_dict = load_file(str(weights_path), device="cpu")
        new_t3.load_state_dict(state_dict, strict=True)
        logger.info("Finetuned weights loaded successfully")

        # Replace T3 in engine
        self.engine.t3 = new_t3

        # Move to device
        self.engine.t3.to(self.device).eval()
        self.engine.s3gen.to(self.device).eval()
        self.engine.ve.to(self.device).eval()
        self.engine.device = self.device

        logger.info("Model loaded and ready")

    def preprocess_text(self, text: str) -> str:
        """Apply Romanian preprocessing: ș→sh, ț→ts, keep ă/â/î."""
        preprocessed = preprocess_and_normalize(text, lowercase=True, mode="phoneme")
        logger.debug(f"Preprocessed: '{text}' → '{preprocessed}'")
        return preprocessed

    def synthesize(
        self,
        text: str,
        reference_path: Path,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """
        Synthesize speech from Romanian text.

        Args:
            text: Romanian text to synthesize
            reference_path: Path to speaker reference audio
            cfg_weight: Classifier-free guidance weight
            exaggeration: Prosodic expressiveness
            temperature: Sampling temperature

        Returns:
            Audio waveform as numpy array (24kHz)
        """
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess Romanian text
        processed_text = self.preprocess_text(text)

        # Generate audio
        logger.debug(f"Generating audio for: {processed_text[:50]}...")
        wav_tensor = self.engine.generate(
            text=processed_text,
            audio_prompt_path=str(reference_path),
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
        )

        # Convert to numpy
        wav_np = wav_tensor.squeeze().cpu().numpy()

        return wav_np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Romanian Chatterbox TTS Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Romanian text to synthesize (single mode)",
    )
    input_group.add_argument(
        "--batch",
        type=Path,
        help="CSV file with id,text columns (batch mode)",
    )

    # Required
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Speaker reference audio (WAV, 3-10 seconds)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WAV path (single mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory (batch mode, default: ./outputs)",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/chatterbox_output"),
        help="Checkpoint directory containing model.safetensors",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/pretrained_models"),
        help="Directory with pretrained Chatterbox components",
    )

    # Generation parameters
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Classifier-free guidance weight (default: 0.5)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Prosodic expressiveness (default: 0.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations."""
    errors = []

    # Single mode requires --output
    if args.text and not args.output:
        errors.append("--output is required when using --text")

    # Batch mode: --output not allowed
    if args.batch and args.output:
        errors.append("Use --output-dir instead of --output for batch mode")

    # Reference must exist
    if not args.reference.exists():
        errors.append(f"Reference audio not found: {args.reference}")

    # Batch file must exist
    if args.batch and not args.batch.exists():
        errors.append(f"Batch CSV not found: {args.batch}")

    # Checkpoint must have model.safetensors
    model_path = args.checkpoint / "model.safetensors"
    if not model_path.exists():
        errors.append(f"Model weights not found: {model_path}")

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validate_args(args)

    logger.info("Arguments validated successfully")
    logger.info(f"Mode: {'single' if args.text else 'batch'}")
    logger.info(f"Reference: {args.reference}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Parameters: cfg_weight={args.cfg_weight}, exaggeration={args.exaggeration}")

    # TODO: Load model and synthesize
    logger.info("Model loading not yet implemented")


if __name__ == "__main__":
    main()
