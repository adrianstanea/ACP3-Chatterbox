# Romanian Inference Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a CLI inference script that synthesizes Romanian speech using the finetuned Chatterbox checkpoint-5300.

**Architecture:** Standalone script at `scripts/inference_romanian.py` with a reusable `RomanianTTSEngine` class. Imports Romanian preprocessing from vendor, loads finetuned T3 weights, exposes cfg_weight/exaggeration via CLI flags. Supports single text and batch modes.

**Tech Stack:** Python 3.12, PyTorch, argparse, tqdm, soundfile, safetensors

---

## Task 1: Create Script Skeleton with Argument Parsing

**Files:**
- Create: `scripts/inference_romanian.py`

**Step 1: Create the argument parser**

```python
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
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference-romanian")


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
```

**Step 2: Test argument parsing works**

Run:
```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation
python scripts/inference_romanian.py --help
```

Expected: Help text with all arguments displayed

**Step 3: Test validation catches missing reference**

Run:
```bash
python scripts/inference_romanian.py --text "Test" --reference nonexistent.wav --output out.wav
```

Expected: Error "Reference audio not found: nonexistent.wav"

**Step 4: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: add Romanian inference script skeleton with arg parsing"
```

---

## Task 2: Implement RomanianTTSEngine Class

**Files:**
- Modify: `scripts/inference_romanian.py`

**Step 1: Add imports and engine class**

Add after the existing imports:

```python
import os
import random
import numpy as np
import torch
import soundfile as sf
from safetensors.torch import load_file

# Add vendor to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "chatterbox-finetuning"))

from src.romanian_preprocessor import preprocess_and_normalize
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3


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
```

**Step 2: Test model loading (requires Docker environment)**

This step requires the Docker environment with GPU. Skip if testing locally without GPU.

**Step 3: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: add RomanianTTSEngine class with model loading"
```

---

## Task 3: Implement Single Text Mode

**Files:**
- Modify: `scripts/inference_romanian.py`

**Step 1: Add seed setting utility**

Add before the `main()` function:

```python
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
```

**Step 2: Implement single text synthesis in main()**

Replace the TODO section in `main()` with:

```python
def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validate_args(args)

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize engine
    engine = RomanianTTSEngine(
        checkpoint_dir=args.checkpoint,
        pretrained_dir=args.pretrained_dir,
        device="cuda",
    )
    engine.load_model()

    # Generation parameters
    gen_params = {
        "cfg_weight": args.cfg_weight,
        "exaggeration": args.exaggeration,
        "temperature": args.temperature,
    }

    if args.text:
        # Single text mode
        run_single_mode(engine, args, gen_params)
    else:
        # Batch mode
        run_batch_mode(engine, args, gen_params)


def run_single_mode(engine: RomanianTTSEngine, args, gen_params: dict):
    """Synthesize a single text."""
    logger.info(f"Synthesizing: {args.text[:50]}{'...' if len(args.text) > 50 else ''}")

    wav = engine.synthesize(
        text=args.text,
        reference_path=args.reference,
        **gen_params,
    )

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), wav, engine.SAMPLE_RATE)

    duration = len(wav) / engine.SAMPLE_RATE
    logger.info(f"Saved {duration:.2f}s audio to {args.output}")
```

**Step 3: Add placeholder for batch mode**

```python
def run_batch_mode(engine: RomanianTTSEngine, args, gen_params: dict):
    """Synthesize from a batch CSV file."""
    logger.info("Batch mode not yet implemented")
    sys.exit(1)
```

**Step 4: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: implement single text synthesis mode"
```

---

## Task 4: Implement Batch Mode

**Files:**
- Modify: `scripts/inference_romanian.py`

**Step 1: Add CSV reading and batch synthesis**

Replace the `run_batch_mode` placeholder:

```python
import csv
from tqdm import tqdm


def run_batch_mode(engine: RomanianTTSEngine, args, gen_params: dict):
    """Synthesize from a batch CSV file."""
    # Read batch file
    rows = []
    with open(args.batch, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate columns
        if "id" not in reader.fieldnames or "text" not in reader.fieldnames:
            logger.error("Batch CSV must have 'id' and 'text' columns")
            sys.exit(1)

        for row in reader:
            if row["text"].strip():  # Skip empty rows
                rows.append(row)

    logger.info(f"Loaded {len(rows)} utterances from {args.batch}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Synthesize each row
    failed = []
    for row in tqdm(rows, desc="Synthesizing", unit="utterance"):
        try:
            wav = engine.synthesize(
                text=row["text"],
                reference_path=args.reference,
                **gen_params,
            )

            output_path = args.output_dir / f"{row['id']}.wav"
            sf.write(str(output_path), wav, engine.SAMPLE_RATE)

        except Exception as e:
            logger.error(f"Failed {row['id']}: {e}")
            failed.append(row["id"])

    # Summary
    success_count = len(rows) - len(failed)
    logger.info(f"Completed: {success_count}/{len(rows)}")

    if failed:
        logger.warning(f"Failed utterances: {failed}")
```

**Step 2: Move import to top of file**

Add `csv` to imports and ensure `tqdm` is imported:

```python
import csv
from tqdm import tqdm
```

**Step 3: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: implement batch synthesis mode with tqdm progress"
```

---

## Task 5: Create Test Batch CSV

**Files:**
- Create: `data/eval/test_sentences.csv`

**Step 1: Create test CSV with Romanian diacritics**

```csv
id,text
test_01,Bună ziua, ce mai faci?
test_02,Știința și înțelepciunea sunt importante.
test_03,Țara mea frumoasă are munți înalți.
test_04,De asemenea, contează și dacă imobilul este la stradă.
test_05,Ăsta e un test pentru vocala ă.
test_06,Î vine de la început și sfârșit.
test_07,Câinele aleargă prin pădure.
test_08,Aceasta este o propoziție simplă fără diacritice speciale.
```

**Step 2: Create directory if needed**

```bash
mkdir -p data/eval
```

**Step 3: Commit**

```bash
git add data/eval/test_sentences.csv
git commit -m "test: add Romanian test sentences for inference validation"
```

---

## Task 6: Integration Test in Docker

**Files:**
- None (testing only)

**Step 1: Start Docker container**

```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation
docker compose up -d
docker compose exec chatterbox bash
```

**Step 2: Run single text test**

Inside container:
```bash
cd /workspace
python scripts/inference_romanian.py \
    --text "Bună ziua, mă numesc Maria." \
    --reference vendor/chatterbox-finetuning/speaker_reference/bas_rnd1_011.wav \
    --output test_single.wav \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output \
    --verbose
```

Expected:
- Model loads without error
- Preprocessing logs show `ș→sh` etc.
- Output file `test_single.wav` created
- Audio is intelligible Romanian

**Step 3: Run batch test**

```bash
python scripts/inference_romanian.py \
    --batch data/eval/test_sentences.csv \
    --reference vendor/chatterbox-finetuning/speaker_reference/bas_rnd1_011.wav \
    --output-dir outputs/test_batch/ \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output
```

Expected:
- Progress bar shows 8 utterances
- All files created in `outputs/test_batch/`
- Summary shows "Completed: 8/8"

**Step 4: Listen to outputs (on host)**

Copy files to host and play:
```bash
# From host
docker cp chatterbox:/workspace/test_single.wav ./
docker cp chatterbox:/workspace/outputs/test_batch/ ./test_batch/
aplay test_single.wav
```

---

## Task 7: Add Error Handling Refinements

**Files:**
- Modify: `scripts/inference_romanian.py`

**Step 1: Add text length warning**

In `RomanianTTSEngine.synthesize()`, add before generation:

```python
    # Warn if text is very long
    MAX_CHARS = 500  # Approximate limit before quality degrades
    if len(text) > MAX_CHARS:
        logger.warning(
            f"Text is {len(text)} chars (>{MAX_CHARS}). "
            "Consider splitting into sentences for better quality."
        )
```

**Step 2: Add reference audio validation**

In `RomanianTTSEngine.synthesize()`, add:

```python
    # Validate reference audio
    import torchaudio
    info = torchaudio.info(str(reference_path))
    duration_sec = info.num_frames / info.sample_rate
    if duration_sec < 1.0:
        logger.warning(f"Reference audio is only {duration_sec:.1f}s (recommend 3-10s)")
```

**Step 3: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: add text length and reference audio validation"
```

---

## Task 8: Final Cleanup and Documentation

**Files:**
- Modify: `scripts/inference_romanian.py`
- Modify: `README.md`

**Step 1: Add module docstring examples**

Update the module docstring at the top of the script to include full examples:

```python
"""
Romanian Chatterbox TTS Inference Script

Synthesizes Romanian speech using the finetuned Chatterbox model (checkpoint-5300).
Applies automatic Romanian text preprocessing: ș→sh, ț→ts, keeps ă/â/î.

Examples:
    # Single text synthesis
    python scripts/inference_romanian.py \\
        --text "Bună ziua, ce mai faci?" \\
        --reference speaker.wav \\
        --output output.wav

    # Batch synthesis from CSV
    python scripts/inference_romanian.py \\
        --batch sentences.csv \\
        --reference speaker.wav \\
        --output-dir outputs/

    # With custom parameters
    python scripts/inference_romanian.py \\
        --text "Aceasta este un test." \\
        --reference speaker.wav \\
        --output output.wav \\
        --cfg-weight 0.3 \\
        --exaggeration 0.7 \\
        --seed 42

CSV format (batch mode):
    id,text
    001,Prima propoziție.
    002,A doua propoziție.
"""
```

**Step 2: Add inference section to README.md**

Add after the training section in README.md:

```markdown
## Inference

After training, generate Romanian speech using the inference script:

### Single Text

```bash
python scripts/inference_romanian.py \
    --text "Bună ziua, mă numesc Maria." \
    --reference path/to/speaker.wav \
    --output output.wav
```

### Batch Processing

Create a CSV file with `id,text` columns:

```csv
id,text
001,Prima propoziție.
002,A doua propoziție.
```

Then run:

```bash
python scripts/inference_romanian.py \
    --batch sentences.csv \
    --reference speaker.wav \
    --output-dir outputs/
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cfg-weight` | 0.5 | Classifier-free guidance (lower = faster speech) |
| `--exaggeration` | 0.5 | Expressiveness (higher = more dramatic) |
| `--temperature` | 0.8 | Sampling randomness |
| `--seed` | None | Random seed for reproducibility |
```

**Step 3: Commit**

```bash
git add scripts/inference_romanian.py README.md
git commit -m "docs: add inference examples and README section"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Script skeleton with arg parsing | `scripts/inference_romanian.py` |
| 2 | RomanianTTSEngine class | `scripts/inference_romanian.py` |
| 3 | Single text mode | `scripts/inference_romanian.py` |
| 4 | Batch mode with tqdm | `scripts/inference_romanian.py` |
| 5 | Test sentences CSV | `data/eval/test_sentences.csv` |
| 6 | Integration test in Docker | (testing) |
| 7 | Error handling refinements | `scripts/inference_romanian.py` |
| 8 | Documentation | `scripts/inference_romanian.py`, `README.md` |

**Total estimated time:** 45-60 minutes

**Dependencies:**
- Docker container running with GPU
- Checkpoint-5300 downloaded to `vendor/chatterbox-finetuning/chatterbox_output/`
- Reference audio available (e.g., `bas_rnd1_011.wav`)
