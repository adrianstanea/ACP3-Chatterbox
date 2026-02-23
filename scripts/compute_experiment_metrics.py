#!/usr/bin/env python3
"""
Compute metrics for Romanian Chatterbox experiment.

Standalone script that computes all objective metrics for synthesized samples:
- CER/WER: Character/Word Error Rate (via Whisper transcription)
- UTMOS: Speech quality (MOS predictor)
- MCD: Mel Cepstral Distortion
- Log F0 RMSE: Prosody similarity
- Speaker Similarity: Cosine similarity of speaker embeddings

Usage:
    python scripts/compute_experiment_metrics.py \
        --experiment-dir experiments/2026-02-22 \
        --swara-dir /data/swara \
        --whisper-model medium
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute experiment metrics")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory containing condition folders and filelists/",
    )
    parser.add_argument(
        "--swara-dir",
        type=Path,
        required=True,
        help="Path to SWARA dataset (original audio files)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conditions that already have metrics computed",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Process only this specific condition (for debugging)",
    )
    return parser.parse_args()


class MetricsComputer:
    """Computes all objective metrics for TTS evaluation."""

    SR = 16_000  # Sample rate for metrics computation

    def __init__(self, whisper_model: str = "medium"):
        logger.info("Initializing metrics computer...")

        # Import heavy dependencies lazily
        import whisper
        from discrete_speech_metrics import UTMOS, LogF0RMSE
        from resemblyzer import VoiceEncoder, preprocess_wav
        import jiwer

        # Add path to DiscreteSpeechMetrics if needed
        pipeline_dir = Path("/home/astanea/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline")
        if pipeline_dir.exists():
            sys.path.insert(0, str(pipeline_dir))
            from DiscreteSpeechMetrics.discrete_speech_metrics.mcd import MCD
            self.mcd_scorer = MCD(sr=self.SR)
        else:
            logger.warning("DiscreteSpeechMetrics not found, MCD will be unavailable")
            self.mcd_scorer = None

        # Initialize Whisper
        logger.info(f"Loading Whisper {whisper_model}...")
        self.whisper = whisper.load_model(whisper_model)
        self.whisper_options = {"language": "ro", "beam_size": 5, "best_of": 5}

        # Initialize other models
        logger.info("Loading UTMOS, F0 RMSE, speaker encoder...")
        self.utmos = UTMOS(sr=self.SR)
        self.log_f0_rmse = LogF0RMSE(sr=self.SR)
        self.voice_encoder = VoiceEncoder()
        self.preprocess_wav = preprocess_wav
        self.jiwer = jiwer

        logger.info("Metrics computer ready")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper."""
        import whisper

        audio = whisper.load_audio(str(audio_path))
        result = self.whisper.transcribe(audio, **self.whisper_options)
        return result["text"].strip()

    def normalize_text(self, text: str) -> str:
        """Normalize text for CER/WER computation."""
        import re
        from num2words import num2words

        chars_to_ignore = r'[,?.!\-\;\:\""%\'"\u2014\u2018\u2026\u2013\u201E]'

        # Convert % to "procente"
        text = text.replace("%", " procente")

        # Lowercase and remove punctuation
        text = re.sub(chars_to_ignore, "", text.lower())

        # Replace numbers with words (Romanian)
        def replace_number(match):
            try:
                return num2words(int(match.group()), lang="ro")
            except Exception:
                return match.group()

        text = re.sub(r"\b\d+\b", replace_number, text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate."""
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)
        return self.jiwer.wer(ref, hyp)

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate."""
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)
        return self.jiwer.cer(ref, hyp)

    def compute_utmos(self, wav: np.ndarray) -> float:
        """Compute UTMOS quality score."""
        return float(self.utmos.score(wav))

    def compute_f0_rmse(self, wav_ref: np.ndarray, wav_synth: np.ndarray) -> float:
        """Compute Log F0 RMSE."""
        return float(self.log_f0_rmse.score(wav_ref, wav_synth))

    def compute_mcd(self, wav_ref: np.ndarray, wav_synth: np.ndarray) -> Optional[float]:
        """Compute Mel Cepstral Distortion."""
        if self.mcd_scorer is None:
            return None
        return float(self.mcd_scorer.score(wav_ref, wav_synth))

    def compute_speaker_similarity(self, wav_ref: np.ndarray, wav_synth: np.ndarray) -> float:
        """Compute speaker embedding cosine similarity."""
        emb_ref = self.voice_encoder.embed_utterance(wav_ref)
        emb_synth = self.voice_encoder.embed_utterance(wav_synth)

        # Cosine similarity
        similarity = np.dot(emb_ref, emb_synth) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_synth)
        )
        return float(similarity)

    def load_wav(self, path: Path) -> np.ndarray:
        """Load and preprocess audio for metrics."""
        return self.preprocess_wav(path, source_sr=self.SR)

    def compute_all_metrics(
        self,
        synth_path: Path,
        orig_path: Path,
        reference_text: str,
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single sample.

        Returns dict with keys:
            - CER, WER
            - UTMOS_Original, UTMOS_Synthesized
            - Log_F0_RMSE, MCD, Speaker_Cosine_Similarity
        """
        results = {}

        # Load audio
        try:
            wav_synth = self.load_wav(synth_path)
            wav_orig = self.load_wav(orig_path)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return {}

        # Transcription and text metrics
        try:
            synth_transcription = self.transcribe(synth_path)
            results["Synthesized_Transcription"] = synth_transcription
            results["CER"] = self.compute_cer(reference_text, synth_transcription)
            results["WER"] = self.compute_wer(reference_text, synth_transcription)
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            results["Synthesized_Transcription"] = ""
            results["CER"] = 1.0
            results["WER"] = 1.0

        # Quality metrics
        try:
            results["UTMOS_Original"] = self.compute_utmos(wav_orig)
            results["UTMOS_Synthesized"] = self.compute_utmos(wav_synth)
        except Exception as e:
            logger.warning(f"UTMOS failed: {e}")
            results["UTMOS_Original"] = 0.0
            results["UTMOS_Synthesized"] = 0.0

        # Prosody metrics
        try:
            results["Log_F0_RMSE"] = self.compute_f0_rmse(wav_orig, wav_synth)
        except Exception as e:
            logger.warning(f"F0 RMSE failed: {e}")
            results["Log_F0_RMSE"] = float("nan")

        # MCD
        try:
            mcd = self.compute_mcd(wav_orig, wav_synth)
            results["MCD"] = mcd if mcd is not None else float("nan")
        except Exception as e:
            logger.warning(f"MCD failed: {e}")
            results["MCD"] = float("nan")

        # Speaker similarity
        try:
            results["Speaker_Cosine_Similarity"] = self.compute_speaker_similarity(
                wav_orig, wav_synth
            )
        except Exception as e:
            logger.warning(f"Speaker similarity failed: {e}")
            results["Speaker_Cosine_Similarity"] = 0.0

        return results


def load_filelist(csv_path: Path) -> List[Tuple[str, str]]:
    """Load filelist (sample_id|text format)."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                sample_id = row[0].strip()
                text = row[1].strip()
                # Ensure .wav extension
                if not sample_id.endswith(".wav"):
                    sample_id = f"{sample_id}.wav"
                samples.append((sample_id, text))
    return samples


def process_condition(
    computer: MetricsComputer,
    condition: str,
    experiment_dir: Path,
    swara_dir: Path,
) -> Optional[Path]:
    """
    Process all samples for a single condition.

    Returns path to output metrics CSV.
    """
    condition_dir = experiment_dir / condition
    filelist_path = experiment_dir / "filelists" / f"{condition}.csv"
    output_dir = experiment_dir / "metrics"
    output_path = output_dir / f"{condition}.csv"

    if not filelist_path.exists():
        logger.error(f"Filelist not found: {filelist_path}")
        return None

    if not condition_dir.exists():
        logger.error(f"Condition directory not found: {condition_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples = load_filelist(filelist_path)
    if not samples:
        logger.error(f"No samples in filelist: {filelist_path}")
        return None

    logger.info(f"Processing {len(samples)} samples for {condition}")

    # CSV header
    header = [
        "Audio_File",
        "Original_Transcription",
        "Synthesized_Transcription",
        "CER",
        "WER",
        "UTMOS_Original",
        "UTMOS_Synthesized",
        "Log_F0_RMSE",
        "MCD",
        "Speaker_Cosine_Similarity",
    ]

    # Process samples
    from tqdm import tqdm

    results_rows = []

    for sample_id, reference_text in tqdm(samples, desc=condition):
        synth_path = condition_dir / sample_id
        orig_path = swara_dir / sample_id

        if not synth_path.exists():
            logger.warning(f"Synthesized audio not found: {synth_path}")
            continue

        if not orig_path.exists():
            logger.warning(f"Original audio not found: {orig_path}")
            continue

        # Compute metrics
        metrics = computer.compute_all_metrics(synth_path, orig_path, reference_text)

        if metrics:
            row = [
                sample_id,
                reference_text,
                metrics.get("Synthesized_Transcription", ""),
                metrics.get("CER", ""),
                metrics.get("WER", ""),
                metrics.get("UTMOS_Original", ""),
                metrics.get("UTMOS_Synthesized", ""),
                metrics.get("Log_F0_RMSE", ""),
                metrics.get("MCD", ""),
                metrics.get("Speaker_Cosine_Similarity", ""),
            ]
            results_rows.append(row)

    # Write results
    if results_rows:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(header)
            writer.writerows(results_rows)

        logger.info(f"Saved {len(results_rows)} results to {output_path}")
        return output_path

    return None


def get_conditions(experiment_dir: Path) -> List[str]:
    """Get list of condition directories."""
    conditions = []
    filelists_dir = experiment_dir / "filelists"

    if not filelists_dir.exists():
        return []

    for csv_file in sorted(filelists_dir.glob("*.csv")):
        condition = csv_file.stem
        condition_dir = experiment_dir / condition
        if condition_dir.exists() and condition_dir.is_dir():
            conditions.append(condition)

    return conditions


def main():
    args = parse_args()

    if not args.experiment_dir.exists():
        logger.error(f"Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)

    if not args.swara_dir.exists():
        logger.error(f"SWARA directory not found: {args.swara_dir}")
        sys.exit(1)

    # Get conditions to process
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = get_conditions(args.experiment_dir)

    if not conditions:
        logger.error("No conditions found to process")
        sys.exit(1)

    logger.info(f"Found {len(conditions)} conditions: {conditions}")

    # Check for existing metrics
    if args.skip_existing:
        metrics_dir = args.experiment_dir / "metrics"
        conditions = [
            c for c in conditions
            if not (metrics_dir / f"{c}.csv").exists()
        ]
        if not conditions:
            logger.info("All conditions already processed")
            sys.exit(0)
        logger.info(f"Processing {len(conditions)} remaining conditions")

    # Initialize metrics computer
    computer = MetricsComputer(whisper_model=args.whisper_model)

    # Process each condition
    processed = 0
    failed = 0

    for condition in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {condition}")
        logger.info(f"{'='*60}")

        try:
            result = process_condition(
                computer=computer,
                condition=condition,
                experiment_dir=args.experiment_dir,
                swara_dir=args.swara_dir,
            )
            if result:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process {condition}: {e}")
            failed += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Processed: {processed} conditions")
    logger.info(f"Failed: {failed} conditions")
    logger.info(f"Metrics saved to: {args.experiment_dir / 'metrics'}")

    # Run aggregation
    if processed > 0:
        logger.info("\nRunning metrics aggregation...")
        import subprocess

        agg_script = Path(__file__).parent / "aggregate_metrics.py"
        if agg_script.exists():
            result = subprocess.run(
                [sys.executable, str(agg_script), "--experiment-dir", str(args.experiment_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Aggregation complete")
            else:
                logger.warning(f"Aggregation failed: {result.stderr}")


if __name__ == "__main__":
    main()
