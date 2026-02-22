"""
ChatterboxRomanian: HuggingFace-compatible Romanian TTS Wrapper

This class wraps the Chatterbox TTS engine with Romanian-specific adaptations,
loading a fine-tuned T3 model while using pretrained components (VoiceEncoder,
S3Gen, tokenizer) from the base multilingual model.

Key features:
- Automatic Romanian text preprocessing (ș→sh, ț→ts)
- Loads fine-tuned T3 weights with prefix stripping
- Uses original vocab size (2454) - no tokenizer extension needed
- HuggingFace-style API with from_local() and from_pretrained()

Usage:
    from chatterbox_ro import ChatterboxRomanian

    model = ChatterboxRomanian.from_local(
        checkpoint_dir="/path/to/finetuned/checkpoint-1000",
        pretrained_dir="/path/to/pretrained_models",
        device="cuda"
    )

    wav = model.generate(
        text="Buna ziua, ce mai faci?",
        reference_audio="/path/to/reference.wav"
    )

Author: Adrian Stanea / ACP3
License: Apache 2.0
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file

# Add vendor path for chatterbox imports
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "chatterbox-finetuning"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

# Import from vendor (no naming conflict since our package is chatterbox_ro, not src)
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.models.t3.modules.t3_config import T3Config
from src.chatterbox_.models.voice_encoder import VoiceEncoder
from src.chatterbox_.models.s3gen import S3Gen
from src.chatterbox_.models.tokenizers import EnTokenizer
from src.romanian_preprocessor import preprocess_and_normalize


class ChatterboxRomanian:
    """
    HuggingFace-compatible Romanian TTS class.

    Wraps ChatterboxTTS with Romanian-specific preprocessing and
    fine-tuned T3 model loading.

    Class Constants:
        VOCAB_SIZE: Original multilingual vocab size (2454)
        SAMPLE_RATE: Output audio sample rate (24000 Hz)
    """

    VOCAB_SIZE = 2454  # Original multilingual vocab, no extension
    SAMPLE_RATE = 24000  # S3Gen output sample rate

    def __init__(self, engine: ChatterboxTTS):
        """
        Initialize with a configured ChatterboxTTS engine.

        Use from_local() or from_pretrained() class methods instead
        of calling __init__ directly.

        Args:
            engine: Configured ChatterboxTTS engine with fine-tuned T3
        """
        self._engine = engine

    @property
    def sr(self) -> int:
        """Sample rate of generated audio."""
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        """Device the model is running on."""
        return str(self._engine.device)

    @classmethod
    def from_local(
        cls,
        checkpoint_dir: Union[str, Path],
        pretrained_dir: Union[str, Path],
        device: str = "cuda",
    ) -> "ChatterboxRomanian":
        """
        Load Romanian TTS model from local directories.

        Loads the fine-tuned T3 model from checkpoint_dir and combines it
        with pretrained components (VE, S3Gen, tokenizer) from pretrained_dir.

        Args:
            checkpoint_dir: Directory containing model.safetensors (fine-tuned T3)
            pretrained_dir: Directory containing pretrained model files:
                - ve.safetensors (VoiceEncoder)
                - s3gen.safetensors (S3Gen decoder)
                - tokenizer.json (text tokenizer)
            device: Device to load model on ("cuda", "cpu", "mps")

        Returns:
            Configured ChatterboxRomanian instance

        Raises:
            FileNotFoundError: If required model files are missing

        Example:
            >>> model = ChatterboxRomanian.from_local(
            ...     checkpoint_dir="outputs/checkpoint-1000",
            ...     pretrained_dir="vendor/chatterbox-finetuning/pretrained_models",
            ...     device="cuda"
            ... )
        """
        checkpoint_dir = Path(checkpoint_dir)
        pretrained_dir = Path(pretrained_dir)

        # Determine map_location for non-CUDA devices
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        # ─────────────────────────────────────────────────────────────────
        # Load pretrained components from pretrained_dir
        # ─────────────────────────────────────────────────────────────────

        # VoiceEncoder
        ve_path = pretrained_dir / "ve.safetensors"
        if not ve_path.exists():
            raise FileNotFoundError(f"VoiceEncoder not found: {ve_path}")
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ve_path))
        ve.to(device).eval()

        # S3Gen
        s3gen_path = pretrained_dir / "s3gen.safetensors"
        if not s3gen_path.exists():
            raise FileNotFoundError(f"S3Gen not found: {s3gen_path}")
        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(s3gen_path), strict=False)
        s3gen.to(device).eval()

        # Tokenizer
        tokenizer_path = pretrained_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        tokenizer = EnTokenizer(str(tokenizer_path))

        # ─────────────────────────────────────────────────────────────────
        # Create T3 with multilingual config and load fine-tuned weights
        # ─────────────────────────────────────────────────────────────────

        t3_config = T3Config.multilingual()
        t3_config.text_tokens_dict_size = cls.VOCAB_SIZE
        t3 = T3(hp=t3_config)

        # Load fine-tuned checkpoint
        checkpoint_path = checkpoint_dir / "model.safetensors"
        if not checkpoint_path.exists():
            # Try pytorch_model.bin as fallback
            checkpoint_path = checkpoint_dir / "pytorch_model.bin"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_dir}. "
                    f"Expected model.safetensors or pytorch_model.bin"
                )

        # Load state dict
        if str(checkpoint_path).endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Strip "t3." prefix from checkpoint keys
        # Training saves as: {"t3.tfmr.layers.0.self_attn...": tensor}
        # T3 expects:        {"tfmr.layers.0.self_attn...": tensor}
        clean_state_dict = {}
        for key, value in state_dict.items():
            # Remove module. prefix (from DDP training)
            key = key.replace("module.", "")

            # Strip t3. prefix
            if key.startswith("t3."):
                clean_key = key[3:]  # Remove "t3."
            else:
                # Skip non-T3 keys (s3gen, ve, tokenizer embeddings)
                if any(x in key for x in ["s3gen", "ve.", "tokenizer"]):
                    continue
                clean_key = key

            clean_state_dict[clean_key] = value

        # Load weights into T3
        missing_keys, unexpected_keys = t3.load_state_dict(clean_state_dict, strict=False)

        # Check for critical missing keys (transformer layers)
        critical_missing = [k for k in missing_keys if "tfmr.layers" in k]
        if critical_missing:
            raise RuntimeError(
                f"Critical T3 weights missing! Found {len(critical_missing)} missing layer keys. "
                f"First few: {critical_missing[:3]}"
            )

        t3.to(device).eval()

        # ─────────────────────────────────────────────────────────────────
        # Assemble ChatterboxTTS engine
        # ─────────────────────────────────────────────────────────────────

        engine = ChatterboxTTS(
            t3=t3,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            conds=None,  # Will be set by prepare_conditionals
        )

        return cls(engine)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "adrianstanea/chatterbox-romanian",
        device: str = "cuda",
    ) -> "ChatterboxRomanian":
        """
        Load Romanian TTS model from HuggingFace Hub.

        This is a placeholder for future HuggingFace Hub release.
        Currently raises NotImplementedError.

        Args:
            repo_id: HuggingFace repository ID
            device: Device to load model on

        Raises:
            NotImplementedError: Hub release not yet available
        """
        raise NotImplementedError(
            f"HuggingFace Hub release not yet available. "
            f"Please use from_local() with your trained checkpoint. "
            f"Future release planned at: https://huggingface.co/{repo_id}"
        )

    def generate(
        self,
        text: str,
        reference_audio: Union[str, Path],
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate Romanian speech from text.

        Automatically applies Romanian preprocessing (ș→sh, ț→ts) before
        synthesis. The reference audio is used for voice cloning.

        Args:
            text: Romanian text to synthesize (normal Romanian with diacritics)
            reference_audio: Path to reference audio for voice cloning (WAV)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
                Higher = more adherence to text, less naturalness
            exaggeration: Emotion/prosody exaggeration (0.0-2.0)
                Higher = more expressive speech
            temperature: Sampling temperature (0.0-1.0)
                Higher = more random, lower = more deterministic
            repetition_penalty: Penalty for repeated tokens (1.0-2.0)
            **kwargs: Additional arguments passed to engine.generate()

        Returns:
            Audio tensor of shape (1, num_samples) at 24000 Hz

        Example:
            >>> model = ChatterboxRomanian.from_local(...)
            >>> wav = model.generate(
            ...     text="Buna ziua, cum te cheama?",
            ...     reference_audio="reference.wav",
            ...     cfg_weight=0.3,
            ...     exaggeration=0.8
            ... )
            >>> import soundfile as sf
            >>> sf.write("output.wav", wav.squeeze().numpy(), model.sr)
        """
        # Apply Romanian preprocessing
        # Maps: ș→sh, ț→ts, keeps ă/â/î, lowercases
        processed_text = preprocess_and_normalize(
            text,
            lowercase=True,
            mode="phoneme",
        )

        # Generate audio using underlying engine
        with torch.inference_mode():
            wav = self._engine.generate(
                text=processed_text,
                audio_prompt_path=str(reference_audio),
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

        return wav

    def __repr__(self) -> str:
        return (
            f"ChatterboxRomanian("
            f"vocab_size={self.VOCAB_SIZE}, "
            f"sample_rate={self.SAMPLE_RATE}, "
            f"device='{self.device}')"
        )
