# Romanian Chatterbox Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build HF-compatible Romanian TTS class, experiment runner, metrics integration, and GitHub Pages gallery for systematic parameter evaluation.

**Architecture:** Modular approach with ChatterboxRomanian class as core, experiment runner generating samples across parameter matrix, integration with external TTS-Evaluation-Pipeline, and static HTML gallery generator.

**Tech Stack:** Python 3.12, PyTorch, safetensors, tqdm, Jinja2 (templates), existing TTS-Evaluation-Pipeline

---

## Task 1: Create ChatterboxRomanian Class

**Files:**
- Create: `src/chatterbox_ro.py`
- Create: `src/__init__.py`

**Step 1: Create src directory and __init__.py**

```bash
mkdir -p src
touch src/__init__.py
```

**Step 2: Create ChatterboxRomanian class skeleton**

Create `src/chatterbox_ro.py`:

```python
"""
Romanian-adapted Chatterbox TTS with HuggingFace-style loading.

Usage:
    from src.chatterbox_ro import ChatterboxRomanian

    tts = ChatterboxRomanian.from_local(
        checkpoint_dir="vendor/chatterbox-finetuning/chatterbox_output",
        pretrained_dir="vendor/chatterbox-finetuning/pretrained_models",
        device="cuda"
    )
    wav = tts.generate("Bună ziua!", reference_audio="speaker.wav")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from safetensors.torch import load_file

# Add vendor to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "chatterbox-finetuning"))

from src.romanian_preprocessor import preprocess_and_normalize
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3

logger = logging.getLogger(__name__)


class ChatterboxRomanian:
    """
    Romanian-adapted Chatterbox TTS with HuggingFace-style loading.

    Automatically applies Romanian text preprocessing (ș→sh, ț→ts).
    Loads finetuned T3 weights while using pretrained ve, s3gen, tokenizer.
    """

    VOCAB_SIZE = 2454  # Original vocab, no extension
    SAMPLE_RATE = 24000

    def __init__(
        self,
        engine: ChatterboxTTS,
        device: str = "cuda",
    ):
        self.engine = engine
        self.device = device
        logger.info(f"ChatterboxRomanian initialized on {device}")

    @classmethod
    def from_local(
        cls,
        checkpoint_dir: Union[str, Path],
        pretrained_dir: Union[str, Path],
        device: str = "cuda",
    ) -> "ChatterboxRomanian":
        """
        Load Romanian TTS from local checkpoint.

        Args:
            checkpoint_dir: Directory containing finetuned model.safetensors
            pretrained_dir: Directory containing pretrained ve, s3gen, tokenizer
            device: Device to load model on ("cuda" or "cpu")

        Returns:
            Configured ChatterboxRomanian instance
        """
        checkpoint_dir = Path(checkpoint_dir)
        pretrained_dir = Path(pretrained_dir)

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        logger.info(f"Loading base model from {pretrained_dir}")
        engine = ChatterboxTTS.from_local(str(pretrained_dir), device="cpu")

        # Create new T3 with original vocab size
        logger.info(f"Creating T3 with vocab size {cls.VOCAB_SIZE}")
        t3_config = engine.t3.hp
        t3_config.text_tokens_dict_size = cls.VOCAB_SIZE
        new_t3 = T3(hp=t3_config)

        # Load finetuned weights
        weights_path = checkpoint_dir / "model.safetensors"
        logger.info(f"Loading finetuned weights from {weights_path}")
        state_dict = load_file(str(weights_path), device="cpu")

        # Strip "t3." prefix from keys if present
        if any(k.startswith("t3.") for k in state_dict.keys()):
            logger.info("Stripping 't3.' prefix from checkpoint keys")
            state_dict = {k.replace("t3.", "", 1): v for k, v in state_dict.items()}

        new_t3.load_state_dict(state_dict, strict=True)
        logger.info("Finetuned weights loaded successfully")

        # Replace T3 in engine
        engine.t3 = new_t3

        # Move to device
        engine.t3.to(device).eval()
        engine.s3gen.to(device).eval()
        engine.ve.to(device).eval()
        engine.device = device

        return cls(engine=engine, device=device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "adrianstanea/chatterbox-romanian",
        device: str = "cuda",
    ) -> "ChatterboxRomanian":
        """
        Load from HuggingFace Hub.

        Note: Not yet implemented. Model release pending.
        """
        raise NotImplementedError(
            f"HuggingFace release pending. Use from_local() for now.\n"
            f"Example:\n"
            f"  tts = ChatterboxRomanian.from_local(\n"
            f"      checkpoint_dir='vendor/chatterbox-finetuning/chatterbox_output',\n"
            f"      pretrained_dir='vendor/chatterbox-finetuning/pretrained_models',\n"
            f"      device='cuda'\n"
            f"  )"
        )

    def _preprocess_text(self, text: str) -> str:
        """Apply Romanian preprocessing: ș→sh, ț→ts, keep ă/â/î."""
        return preprocess_and_normalize(text, lowercase=True, mode="phoneme")

    def generate(
        self,
        text: str,
        reference_audio: Union[str, Path],
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
    ) -> np.ndarray:
        """
        Generate speech from Romanian text.

        Args:
            text: Romanian text to synthesize
            reference_audio: Path to speaker reference audio (3-10 seconds)
            cfg_weight: Classifier-free guidance (0.3=slower, 0.5=natural)
            exaggeration: Prosodic expressiveness (0.3=calm, 0.7=dramatic)
            temperature: Sampling temperature

        Returns:
            Audio waveform as numpy array (24kHz)
        """
        # Preprocess Romanian text
        processed_text = self._preprocess_text(text)
        logger.debug(f"Preprocessed: '{text}' → '{processed_text}'")

        # Generate audio
        wav_tensor = self.engine.generate(
            text=processed_text,
            audio_prompt_path=str(reference_audio),
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
        )

        return wav_tensor.squeeze().cpu().numpy()
```

**Step 3: Verify imports work**

Run inside Docker:
```bash
docker compose exec chatterbox python -c "from src.chatterbox_ro import ChatterboxRomanian; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/
git commit -m "feat: add HF-compatible ChatterboxRomanian class"
```

---

## Task 2: Create Experiment Runner Script

**Files:**
- Create: `scripts/run_experiment.py`

**Step 1: Create experiment runner**

Create `scripts/run_experiment.py`:

```python
#!/usr/bin/env python3
"""
Romanian Chatterbox Experiment Runner

Generates samples across parameter matrix for systematic evaluation.

Usage:
    python scripts/run_experiment.py \
        --checkpoint vendor/chatterbox-finetuning/chatterbox_output \
        --holdout-csv data/processed/MyTTSDataset/splits/test_holdout.csv \
        --swara-dir /data/swara \
        --output-dir experiments/2026-02-22 \
        --samples-per-condition 50
"""

import argparse
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatterbox_ro import ChatterboxRomanian

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    speakers: List[str]
    exaggeration_values: List[float]
    cfg_values: List[float]
    samples_per_condition: int
    seed: int
    timestamp: str

    def to_dict(self):
        return {
            "speakers": self.speakers,
            "exaggeration_values": self.exaggeration_values,
            "cfg_values": self.cfg_values,
            "samples_per_condition": self.samples_per_condition,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "total_conditions": len(self.speakers) * len(self.exaggeration_values) * len(self.cfg_values),
            "total_samples": self.samples_per_condition * len(self.speakers) * len(self.exaggeration_values) * len(self.cfg_values),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Romanian TTS experiment")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/chatterbox_output"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("vendor/chatterbox-finetuning/pretrained_models"),
        help="Pretrained models directory",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        required=True,
        help="Path to holdout CSV (id|raw_text|norm_text)",
    )
    parser.add_argument(
        "--swara-dir",
        type=Path,
        required=True,
        help="Path to SWARA dataset (for reference audio)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for experiment",
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=50,
        help="Number of samples per condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def load_holdout_samples(csv_path: Path) -> dict:
    """
    Load holdout samples grouped by speaker.

    Returns:
        Dict mapping speaker prefix (bas, sgs) to list of (id, text) tuples
    """
    samples = {"bas": [], "sgs": []}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                sample_id, text = row[0], row[1]
                speaker = sample_id.split("_")[0].lower()
                if speaker in samples:
                    samples[speaker].append((sample_id, text))

    logger.info(f"Loaded holdout: BAS={len(samples['bas'])}, SGS={len(samples['sgs'])}")
    return samples


def get_reference_audio(speaker: str, swara_dir: Path) -> Path:
    """Get reference audio path for speaker."""
    ref_file = swara_dir / f"{speaker}_rnd1_011.wav"
    if not ref_file.exists():
        # Fallback to first available
        candidates = list(swara_dir.glob(f"{speaker}_*.wav"))
        if candidates:
            ref_file = candidates[0]
        else:
            raise FileNotFoundError(f"No reference audio found for {speaker}")
    return ref_file


def run_experiment(args):
    """Run the experiment."""
    random.seed(args.seed)

    # Experiment configuration
    config = ExperimentConfig(
        speakers=["bas", "sgs"],
        exaggeration_values=[0.3, 0.7],
        cfg_values=[0.3, 0.5],
        samples_per_condition=args.samples_per_condition,
        seed=args.seed,
        timestamp=datetime.now().isoformat(),
    )

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filelists_dir = args.output_dir / "filelists"
    filelists_dir.mkdir(exist_ok=True)

    # Save config
    with open(args.output_dir / "experiment_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Load model
    logger.info("Loading ChatterboxRomanian model...")
    tts = ChatterboxRomanian.from_local(
        checkpoint_dir=args.checkpoint,
        pretrained_dir=args.pretrained,
        device=args.device,
    )

    # Load holdout samples
    holdout = load_holdout_samples(args.holdout_csv)

    # Run experiments
    for speaker in config.speakers:
        # Get reference audio
        ref_audio = get_reference_audio(speaker, args.swara_dir)
        logger.info(f"Using reference audio: {ref_audio}")

        # Select samples for this speaker
        speaker_samples = holdout[speaker]
        if len(speaker_samples) < args.samples_per_condition:
            logger.warning(
                f"Only {len(speaker_samples)} samples for {speaker}, "
                f"requested {args.samples_per_condition}"
            )
        selected = random.sample(
            speaker_samples,
            min(args.samples_per_condition, len(speaker_samples))
        )

        for exag in config.exaggeration_values:
            for cfg in config.cfg_values:
                condition = f"{speaker}_exag{exag}_cfg{cfg}"
                condition_dir = args.output_dir / condition
                condition_dir.mkdir(exist_ok=True)

                # Prepare filelist
                filelist_rows = [["base_file", "transcript_original"]]

                logger.info(f"Generating {condition}...")
                for sample_id, text in tqdm(selected, desc=condition):
                    try:
                        wav = tts.generate(
                            text=text,
                            reference_audio=ref_audio,
                            cfg_weight=cfg,
                            exaggeration=exag,
                        )

                        output_path = condition_dir / f"{sample_id}.wav"
                        sf.write(output_path, wav, ChatterboxRomanian.SAMPLE_RATE)
                        filelist_rows.append([f"{sample_id}.wav", text])

                    except Exception as e:
                        logger.error(f"Failed {sample_id}: {e}")

                # Write filelist
                filelist_path = filelists_dir / f"{condition}.csv"
                with open(filelist_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="|")
                    writer.writerows(filelist_rows)

                logger.info(f"Completed {condition}: {len(filelist_rows)-1} samples")

    logger.info(f"Experiment complete. Output: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
```

**Step 2: Verify script runs (dry run check)**

```bash
python scripts/run_experiment.py --help
```

Expected: Help message with all arguments

**Step 3: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat: add experiment runner for parameter evaluation"
```

---

## Task 3: Create Metrics Aggregation Script

**Files:**
- Create: `scripts/aggregate_metrics.py`

**Step 1: Create aggregation script**

Create `scripts/aggregate_metrics.py`:

```python
#!/usr/bin/env python3
"""
Aggregate metrics from TTS evaluation pipeline results.

Reads per-sample metrics CSVs and computes mean/std per condition.

Usage:
    python scripts/aggregate_metrics.py --experiment-dir experiments/2026-02-22
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory containing metrics/ folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: experiment-dir/results_summary.csv)",
    )
    return parser.parse_args()


def aggregate_metrics(experiment_dir: Path, output_path: Path):
    """Aggregate metrics from all condition CSVs."""
    metrics_dir = experiment_dir / "metrics"

    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        print("Run the TTS evaluation pipeline first.")
        return

    results = []
    metric_columns = ["CER", "WER", "UTMOS_Synthesized", "Log_F0_RMSE", "MCD", "Speaker_Cosine_Similarity"]

    for csv_file in sorted(metrics_dir.glob("*.csv")):
        condition = csv_file.stem
        try:
            df = pd.read_csv(csv_file, delimiter="|")

            row = {"condition": condition}
            for col in metric_columns:
                if col in df.columns:
                    row[f"{col}_mean"] = df[col].mean()
                    row[f"{col}_std"] = df[col].std()

            results.append(row)
            print(f"Processed: {condition}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        print(summary_df.to_string())
    else:
        print("No metrics found to aggregate.")


if __name__ == "__main__":
    args = parse_args()
    output = args.output or (args.experiment_dir / "results_summary.csv")
    aggregate_metrics(args.experiment_dir, output)
```

**Step 2: Commit**

```bash
git add scripts/aggregate_metrics.py
git commit -m "feat: add metrics aggregation script"
```

---

## Task 4: Create Metrics Pipeline Wrapper

**Files:**
- Create: `scripts/run_metrics.sh`

**Step 1: Create wrapper script**

Create `scripts/run_metrics.sh`:

```bash
#!/bin/bash
#
# Run TTS evaluation pipeline on experiment results
#
# Usage:
#   ./scripts/run_metrics.sh experiments/2026-02-22 /path/to/swara
#

set -e

EXPERIMENT_DIR=${1:?Usage: $0 <experiment-dir> <swara-dir>}
SWARA_DIR=${2:?Usage: $0 <experiment-dir> <swara-dir>}
PIPELINE_DIR=${PIPELINE_DIR:-/home/astanea/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline}

echo "========================================"
echo "Running TTS Evaluation Pipeline"
echo "Experiment: $EXPERIMENT_DIR"
echo "SWARA: $SWARA_DIR"
echo "Pipeline: $PIPELINE_DIR"
echo "========================================"

# Create output directories
mkdir -p "$EXPERIMENT_DIR/transcriptions"
mkdir -p "$EXPERIMENT_DIR/metrics"

# Process each condition
for filelist in "$EXPERIMENT_DIR"/filelists/*.csv; do
    condition=$(basename "$filelist" .csv)
    echo ""
    echo "Processing: $condition"
    echo "----------------------------------------"

    # Step 1: Transcribe synthesized audio with Whisper
    echo "Step 1: Transcribing..."
    python "$PIPELINE_DIR/transcription.py" \
        --input_filelist "$filelist" \
        --base_dir_synthesized "$EXPERIMENT_DIR/$condition" \
        --generator_model "chatterbox-ro-$condition" \
        || { echo "Transcription failed for $condition"; continue; }

    # The transcription script outputs to a specific location
    # Move/copy the output to our experiment dir
    TRANS_OUTPUT="$PIPELINE_DIR/chatterbox-ro-$condition.csv"
    if [ -f "$TRANS_OUTPUT" ]; then
        mv "$TRANS_OUTPUT" "$EXPERIMENT_DIR/transcriptions/$condition.csv"
    fi

    # Step 2: Compute metrics
    echo "Step 2: Computing metrics..."
    python "$PIPELINE_DIR/metrics.py" \
        --input_filelist "$EXPERIMENT_DIR/transcriptions/$condition.csv" \
        --base_dir_synthesized "$EXPERIMENT_DIR/$condition" \
        --base_dir_original "$SWARA_DIR" \
        --output_csv "$condition.csv" \
        || { echo "Metrics failed for $condition"; continue; }

    # Move metrics output
    if [ -f "$PIPELINE_DIR/results/$condition.csv" ]; then
        mv "$PIPELINE_DIR/results/$condition.csv" "$EXPERIMENT_DIR/metrics/"
    fi

    echo "Completed: $condition"
done

# Step 3: Aggregate results
echo ""
echo "========================================"
echo "Aggregating results..."
echo "========================================"
python scripts/aggregate_metrics.py --experiment-dir "$EXPERIMENT_DIR"

echo ""
echo "Done! Results in: $EXPERIMENT_DIR/results_summary.csv"
```

**Step 2: Make executable**

```bash
chmod +x scripts/run_metrics.sh
```

**Step 3: Commit**

```bash
git add scripts/run_metrics.sh
git commit -m "feat: add metrics pipeline wrapper script"
```

---

## Task 5: Create GitHub Pages Gallery Generator

**Files:**
- Create: `scripts/build_gallery.py`
- Create: `docs/style.css`

**Step 1: Create gallery generator**

Create `scripts/build_gallery.py`:

```python
#!/usr/bin/env python3
"""
Generate static HTML gallery for experiment results.

Creates academic-style GitHub Pages site with audio samples.

Usage:
    python scripts/build_gallery.py \
        --experiment-dir experiments/2026-02-22 \
        --output-dir docs
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Build audio gallery")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Output directory for GitHub Pages",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Max samples to include per condition (for file size)",
    )
    return parser.parse_args()


def get_conditions(experiment_dir: Path) -> List[str]:
    """Get list of condition directories."""
    conditions = []
    for d in sorted(experiment_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(("filelists", "transcriptions", "metrics")):
            conditions.append(d.name)
    return conditions


def load_filelist(experiment_dir: Path, condition: str) -> List[Dict]:
    """Load filelist for a condition."""
    filelist_path = experiment_dir / "filelists" / f"{condition}.csv"
    samples = []
    if filelist_path.exists():
        with open(filelist_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                samples.append(row)
    return samples


def load_results_summary(experiment_dir: Path) -> Dict:
    """Load aggregated results."""
    summary_path = experiment_dir / "results_summary.csv"
    results = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["condition"]] = row
    return results


def generate_style_css() -> str:
    """Generate minimal academic CSS."""
    return """
:root {
    --bg: #fafafa;
    --text: #333;
    --border: #ddd;
    --accent: #2563eb;
}

* { box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: var(--text);
    background: var(--bg);
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
h3 { font-size: 1rem; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

nav { margin-bottom: 2rem; padding: 1rem; background: #fff; border: 1px solid var(--border); }
nav a { margin-right: 1rem; }

.sample {
    background: #fff;
    border: 1px solid var(--border);
    padding: 1rem;
    margin-bottom: 1rem;
}

.sample-id { font-family: monospace; font-size: 0.875rem; color: #666; }
.sample-text { margin: 0.5rem 0; }

audio { width: 100%; margin-top: 0.5rem; }

table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { padding: 0.5rem; text-align: left; border: 1px solid var(--border); }
th { background: #f5f5f5; }
td { font-family: monospace; font-size: 0.875rem; }

.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.875rem; color: #666; }
"""


def generate_nav(conditions: List[str], current: str = "") -> str:
    """Generate navigation HTML."""
    links = ['<a href="index.html">Home</a>', '<a href="results.html">Results</a>']
    for c in conditions:
        cls = ' style="font-weight:bold"' if c == current else ""
        links.append(f'<a href="{c}.html"{cls}>{c}</a>')
    return f'<nav>{"".join(links)}</nav>'


def generate_index_html(conditions: List[str], config: Dict) -> str:
    """Generate index page."""
    nav = generate_nav(conditions)
    conditions_list = "\n".join(f'<li><a href="{c}.html">{c}</a></li>' for c in conditions)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Romanian Chatterbox Evaluation</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Romanian Chatterbox TTS Evaluation</h1>
    <p>Systematic evaluation of generation parameters on Romanian speech synthesis.</p>

    {nav}

    <h2>Experiment Overview</h2>
    <table>
        <tr><th>Parameter</th><th>Values</th></tr>
        <tr><td>Speakers</td><td>{', '.join(config.get('speakers', []))}</td></tr>
        <tr><td>Exaggeration</td><td>{config.get('exaggeration_values', [])}</td></tr>
        <tr><td>CFG</td><td>{config.get('cfg_values', [])}</td></tr>
        <tr><td>Samples per condition</td><td>{config.get('samples_per_condition', 'N/A')}</td></tr>
        <tr><td>Total samples</td><td>{config.get('total_samples', 'N/A')}</td></tr>
    </table>

    <h2>Conditions</h2>
    <ul>
        {conditions_list}
    </ul>

    <h2>Parameter Descriptions</h2>
    <p><strong>Exaggeration:</strong> Controls prosodic expressiveness. Lower values (0.3) produce calm, neutral speech. Higher values (0.7) produce more dramatic, expressive speech.</p>
    <p><strong>CFG (Classifier-Free Guidance):</strong> Controls speaking rate and confidence. Lower values (0.3) produce slower, more deliberate speech. Higher values (0.5) produce more natural pacing.</p>

    <div class="footer">
        <p>Generated from checkpoint trained on SWARA dataset.</p>
    </div>
</body>
</html>
"""


def generate_condition_html(condition: str, samples: List[Dict], conditions: List[str], max_samples: int) -> str:
    """Generate condition page."""
    nav = generate_nav(conditions, condition)

    # Parse condition name
    parts = condition.split("_")
    speaker = parts[0].upper()
    exag = parts[1].replace("exag", "")
    cfg = parts[2].replace("cfg", "")

    samples_html = ""
    for i, sample in enumerate(samples[:max_samples]):
        sample_id = sample.get("base_file", "").replace(".wav", "")
        text = sample.get("transcript_original", "")
        samples_html += f"""
        <div class="sample">
            <div class="sample-id">{sample_id}</div>
            <div class="sample-text">{text}</div>
            <audio controls preload="none">
                <source src="audio/{condition}/{sample.get('base_file', '')}" type="audio/wav">
            </audio>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{condition} - Romanian Chatterbox</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>{speaker}</h1>
    <p>Exaggeration: {exag} | CFG: {cfg}</p>

    {nav}

    <h2>Samples ({min(len(samples), max_samples)} of {len(samples)})</h2>
    {samples_html}

    <div class="footer">
        <a href="index.html">Back to overview</a>
    </div>
</body>
</html>
"""


def generate_results_html(results: Dict, conditions: List[str]) -> str:
    """Generate results summary page."""
    nav = generate_nav(conditions)

    if not results:
        table = "<p>No results available. Run the metrics pipeline first.</p>"
    else:
        # Build table
        headers = ["Condition", "CER", "WER", "UTMOS", "MCD", "Speaker Sim"]
        rows = ""
        for condition in conditions:
            if condition in results:
                r = results[condition]
                rows += f"""
                <tr>
                    <td>{condition}</td>
                    <td>{float(r.get('CER_mean', 0)):.3f} ± {float(r.get('CER_std', 0)):.3f}</td>
                    <td>{float(r.get('WER_mean', 0)):.3f} ± {float(r.get('WER_std', 0)):.3f}</td>
                    <td>{float(r.get('UTMOS_Synthesized_mean', 0)):.2f} ± {float(r.get('UTMOS_Synthesized_std', 0)):.2f}</td>
                    <td>{float(r.get('MCD_mean', 0)):.2f} ± {float(r.get('MCD_std', 0)):.2f}</td>
                    <td>{float(r.get('Speaker_Cosine_Similarity_mean', 0)):.3f} ± {float(r.get('Speaker_Cosine_Similarity_std', 0)):.3f}</td>
                </tr>
                """

        table = f"""
        <table>
            <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
            {rows}
        </table>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Romanian Chatterbox</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Evaluation Results</h1>

    {nav}

    <h2>Metrics Summary</h2>
    <p>Mean ± standard deviation across samples.</p>
    {table}

    <h2>Metric Descriptions</h2>
    <ul>
        <li><strong>CER:</strong> Character Error Rate (lower is better)</li>
        <li><strong>WER:</strong> Word Error Rate (lower is better)</li>
        <li><strong>UTMOS:</strong> Speech quality score (higher is better, max ~5)</li>
        <li><strong>MCD:</strong> Mel Cepstral Distortion (lower is better)</li>
        <li><strong>Speaker Sim:</strong> Speaker similarity to reference (higher is better, max 1.0)</li>
    </ul>

    <div class="footer">
        <a href="index.html">Back to overview</a>
    </div>
</body>
</html>
"""


def build_gallery(args):
    """Build the gallery."""
    # Load experiment config
    config_path = args.experiment_dir / "experiment_config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Get conditions
    conditions = get_conditions(args.experiment_dir)
    if not conditions:
        print(f"No conditions found in {args.experiment_dir}")
        return

    print(f"Found {len(conditions)} conditions")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Write CSS
    with open(args.output_dir / "style.css", "w") as f:
        f.write(generate_style_css())

    # Write index
    with open(args.output_dir / "index.html", "w") as f:
        f.write(generate_index_html(conditions, config))

    # Load results
    results = load_results_summary(args.experiment_dir)

    # Write results page
    with open(args.output_dir / "results.html", "w") as f:
        f.write(generate_results_html(results, conditions))

    # Generate condition pages and copy audio
    for condition in conditions:
        print(f"Processing: {condition}")

        # Load samples
        samples = load_filelist(args.experiment_dir, condition)

        # Write condition page
        with open(args.output_dir / f"{condition}.html", "w") as f:
            f.write(generate_condition_html(condition, samples, conditions, args.max_samples))

        # Copy audio files (limited number)
        condition_audio_dir = audio_dir / condition
        condition_audio_dir.mkdir(exist_ok=True)

        src_dir = args.experiment_dir / condition
        for i, sample in enumerate(samples[:args.max_samples]):
            src_file = src_dir / sample.get("base_file", "")
            if src_file.exists():
                shutil.copy(src_file, condition_audio_dir / src_file.name)

    print(f"\nGallery built: {args.output_dir}")
    print(f"Enable GitHub Pages on docs/ folder to publish.")


if __name__ == "__main__":
    args = parse_args()
    build_gallery(args)
```

**Step 2: Commit**

```bash
git add scripts/build_gallery.py
git commit -m "feat: add GitHub Pages gallery generator"
```

---

## Task 6: Copy SGS Reference Audio

**Files:**
- Copy: `vendor/chatterbox-finetuning/speaker_reference/sgs_rnd1_011.wav`

**Step 1: Copy SGS reference from SWARA**

```bash
cp /home/astanea/data/SWARA1.0_22k_noSil/sgs_rnd1_011.wav \
   vendor/chatterbox-finetuning/speaker_reference/
```

**Step 2: Verify both references exist**

```bash
ls -la vendor/chatterbox-finetuning/speaker_reference/*.wav
```

Expected: Both `bas_rnd1_011.wav` and `sgs_rnd1_011.wav`

**Step 3: Commit**

```bash
git add vendor/chatterbox-finetuning/speaker_reference/sgs_rnd1_011.wav
git commit -m "data: add SGS reference audio for experiments"
```

---

## Task 7: Integration Test

**Step 1: Test ChatterboxRomanian loading**

```bash
docker compose exec chatterbox python -c "
from src.chatterbox_ro import ChatterboxRomanian
tts = ChatterboxRomanian.from_local(
    'vendor/chatterbox-finetuning/chatterbox_output',
    'vendor/chatterbox-finetuning/pretrained_models',
    'cuda'
)
wav = tts.generate('Test.', 'vendor/chatterbox-finetuning/speaker_reference/bas_rnd1_011.wav')
print(f'Generated {len(wav)/24000:.2f}s audio')
"
```

Expected: `Generated X.XX s audio`

**Step 2: Test experiment runner (small run)**

```bash
docker compose exec chatterbox python scripts/run_experiment.py \
    --checkpoint vendor/chatterbox-finetuning/chatterbox_output \
    --holdout-csv data/processed/MyTTSDataset/splits/test_holdout.csv \
    --swara-dir /data/swara \
    --output-dir experiments/test \
    --samples-per-condition 2 \
    --seed 42
```

Expected: Creates `experiments/test/` with 8 condition directories, 2 samples each

**Step 3: Test gallery generator**

```bash
python scripts/build_gallery.py \
    --experiment-dir experiments/test \
    --output-dir docs \
    --max-samples 2
```

Expected: Creates `docs/` with HTML files

**Step 4: Commit test verification**

```bash
git add -A
git commit -m "test: verify experiment pipeline integration"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | ChatterboxRomanian class | `src/chatterbox_ro.py` |
| 2 | Experiment runner | `scripts/run_experiment.py` |
| 3 | Metrics aggregation | `scripts/aggregate_metrics.py` |
| 4 | Metrics wrapper | `scripts/run_metrics.sh` |
| 5 | Gallery generator | `scripts/build_gallery.py` |
| 6 | SGS reference audio | `vendor/.../sgs_rnd1_011.wav` |
| 7 | Integration test | Verify all components |

**Total:** 7 tasks, ~5 new files

**After completion:**
1. Run full experiment: `python scripts/run_experiment.py --samples-per-condition 50`
2. Run metrics: `./scripts/run_metrics.sh experiments/2026-02-22 /data/swara`
3. Build gallery: `python scripts/build_gallery.py --experiment-dir experiments/2026-02-22`
4. Enable GitHub Pages on `docs/` folder
