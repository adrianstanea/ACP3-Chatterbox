#!/usr/bin/env bash
#
# Romanian Chatterbox Full Experiment Pipeline
#
# Single script to run or redo the entire experiment:
#   1. Sample generation across parameter grid
#   2. Transcription (Whisper on original + synthesized audio)
#   3. Metrics computation (CER, WER, UTMOS, etc.)
#   4. Best sample selection
#   5. Report generation (tables + figures)
#   6. Gallery build for GitHub Pages
#
# Usage:
#   ./scripts/run_full_experiment.sh [options]
#
# Options:
#   --skip-synthesis       Skip sample generation (use existing)
#   --skip-transcription   Skip transcription (use existing)
#   --skip-metrics         Skip metrics computation (use existing)
#   --samples N            Samples per condition (default: 50)
#   --gallery-samples N    Best samples per condition for gallery (default: 5)
#   --whisper-model MODEL  Whisper model: tiny|base|small|medium|large (default: medium)
#   --experiment-dir DIR   Output directory (default: experiments/YYYY-MM-DD)
#   --swara-dir DIR        SWARA dataset directory (default: /data/swara)
#   --help                 Show this help
#
# Requirements:
#   - Run inside Docker container with GPU
#   - Trained model in vendor/chatterbox-finetuning/chatterbox_output/
#   - SWARA dataset mounted at /data/swara/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
CHECKPOINT_DIR="vendor/chatterbox-finetuning/chatterbox_output"
PRETRAINED_DIR="vendor/chatterbox-finetuning/pretrained_models"
HOLDOUT_CSV="data/processed/MyTTSDataset/splits/test_holdout.csv"
SWARA_DIR="${SWARA_DIR:-/data/swara}"
EXPERIMENT_DIR=""

# Default parameters
SAMPLES_PER_CONDITION=50
GALLERY_SAMPLES=5
WHISPER_MODEL="medium"
SKIP_SYNTHESIS=false
SKIP_TRANSCRIPTION=false
SKIP_METRICS=false
STATS_ONLY=false

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Romanian Chatterbox Full Experiment Pipeline

Options:
    --skip-synthesis       Skip sample generation (use existing samples)
    --skip-transcription   Skip transcription (use existing transcriptions)
    --skip-metrics         Skip metrics computation (use existing metrics)
    --stats-only           Only compute statistics from existing metrics (skip all other steps)
    --samples N            Samples per condition for synthesis (default: 50)
    --gallery-samples N    Best samples per condition for gallery (default: 5)
    --whisper-model MODEL  Whisper model: tiny|base|small|medium|large (default: medium)
    --experiment-dir DIR   Experiment directory (default: experiments/YYYY-MM-DD)
    --swara-dir DIR        SWARA dataset directory (default: /data/swara)
    --help                 Show this help message

Examples:
    # Full pipeline with defaults
    $(basename "$0")

    # Skip synthesis and redo transcription + metrics (for new Whisper code)
    $(basename "$0") --skip-synthesis

    # Skip synthesis and transcription, only redo metrics
    $(basename "$0") --skip-synthesis --skip-transcription

    # Resume from metrics (keep synthesis + transcription)
    $(basename "$0") --skip-synthesis --skip-transcription --skip-metrics

    # Compute only statistics from existing metrics (after data cleaning)
    $(basename "$0") --experiment-dir experiments/2026-02-22 --stats-only

    # Quick test with fewer samples
    $(basename "$0") --samples 5 --whisper-model tiny

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-synthesis)
            SKIP_SYNTHESIS=true
            shift
            ;;
        --skip-transcription)
            SKIP_TRANSCRIPTION=true
            shift
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        --stats-only)
            STATS_ONLY=true
            SKIP_SYNTHESIS=true
            SKIP_TRANSCRIPTION=true
            SKIP_METRICS=true
            shift
            ;;
        --samples)
            SAMPLES_PER_CONDITION="$2"
            shift 2
            ;;
        --gallery-samples)
            GALLERY_SAMPLES="$2"
            shift 2
            ;;
        --whisper-model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --experiment-dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --swara-dir)
            SWARA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

# Set default experiment directory if not specified
if [[ -z "$EXPERIMENT_DIR" ]]; then
    EXPERIMENT_DIR="experiments/$(date +%Y-%m-%d)"
fi

# ============================================================================
# Validation
# ============================================================================

cd "$PROJECT_DIR"

echo "========================================================================"
echo "Romanian Chatterbox Full Experiment Pipeline"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Project dir:        $PROJECT_DIR"
echo "  Experiment dir:     $EXPERIMENT_DIR"
echo "  SWARA dir:          $SWARA_DIR"
echo "  Checkpoint:         $CHECKPOINT_DIR"
echo "  Samples/condition:  $SAMPLES_PER_CONDITION"
echo "  Gallery samples:    $GALLERY_SAMPLES"
echo "  Whisper model:      $WHISPER_MODEL"
echo "  Skip synthesis:     $SKIP_SYNTHESIS"
echo "  Skip transcription: $SKIP_TRANSCRIPTION"
echo "  Skip metrics:       $SKIP_METRICS"
echo "  Stats only:         $STATS_ONLY"
echo ""

# Check prerequisites
ERRORS=0

# If stats-only mode, only need metrics directory
if [[ "$STATS_ONLY" == "true" ]]; then
    if [[ ! -d "$EXPERIMENT_DIR/metrics" ]]; then
        echo "ERROR: Metrics directory not found: $EXPERIMENT_DIR/metrics"
        echo "       Cannot compute statistics without existing metrics."
        ERRORS=$((ERRORS + 1))
    fi
else
    if [[ ! -d "$SWARA_DIR" ]]; then
        echo "ERROR: SWARA directory not found: $SWARA_DIR"
        ERRORS=$((ERRORS + 1))
    fi

    if [[ "$SKIP_SYNTHESIS" == "false" ]]; then
        if [[ ! -d "$CHECKPOINT_DIR" ]]; then
            echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
            echo "       Train the model first or use --skip-synthesis"
            ERRORS=$((ERRORS + 1))
        fi

        if [[ ! -f "$HOLDOUT_CSV" ]]; then
            echo "ERROR: Holdout CSV not found: $HOLDOUT_CSV"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "Found $ERRORS error(s). Fix them and try again."
    exit 1
fi

echo "Validation passed. Starting pipeline..."
echo ""

# ============================================================================
# Step 1: Sample Generation
# ============================================================================

if [[ "$SKIP_SYNTHESIS" == "false" ]]; then
    echo "========================================================================"
    echo "STEP 1/6: Sample Generation"
    echo "========================================================================"
    echo "Generating $SAMPLES_PER_CONDITION samples per condition..."
    echo ""

    python3 scripts/run_experiment.py \
        --checkpoint "$CHECKPOINT_DIR" \
        --pretrained "$PRETRAINED_DIR" \
        --holdout-csv "$HOLDOUT_CSV" \
        --swara-dir "$SWARA_DIR" \
        --output-dir "$EXPERIMENT_DIR" \
        --samples-per-condition "$SAMPLES_PER_CONDITION" \
        --seed 42

    echo ""
    echo "Sample generation complete."
    echo ""
else
    echo "========================================================================"
    echo "STEP 1/6: Sample Generation (SKIPPED)"
    echo "========================================================================"
    echo "Using existing samples in $EXPERIMENT_DIR"
    echo ""

    if [[ ! -d "$EXPERIMENT_DIR" ]]; then
        echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
        echo "       Cannot skip synthesis without existing samples."
        exit 1
    fi
fi

# ============================================================================
# Step 2: Transcription (Whisper on original + synthesized audio)
# ============================================================================

if [[ "$SKIP_TRANSCRIPTION" == "false" ]]; then
    echo "========================================================================"
    echo "STEP 2/6: Transcription"
    echo "========================================================================"
    echo "Transcribing original and synthesized audio with Whisper..."
    echo ""

    python3 scripts/compute_experiment_metrics.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --swara-dir "$SWARA_DIR" \
        --step transcribe \
        --whisper-model "$WHISPER_MODEL"

    echo ""
    echo "Transcription complete."
    echo ""
else
    echo "========================================================================"
    echo "STEP 2/6: Transcription (SKIPPED)"
    echo "========================================================================"
    echo "Using existing transcriptions in $EXPERIMENT_DIR/transcriptions/"
    echo ""

    if [[ ! -d "$EXPERIMENT_DIR/transcriptions" ]]; then
        echo "ERROR: Transcriptions directory not found: $EXPERIMENT_DIR/transcriptions"
        echo "       Cannot skip transcription without existing data."
        exit 1
    fi
fi

# ============================================================================
# Step 3: Metrics Computation
# ============================================================================

if [[ "$SKIP_METRICS" == "false" ]]; then
    echo "========================================================================"
    echo "STEP 3/6: Metrics Computation"
    echo "========================================================================"
    echo "Computing CER, WER, UTMOS, MCD, F0 RMSE, Speaker Similarity..."
    echo ""

    python3 scripts/compute_experiment_metrics.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --swara-dir "$SWARA_DIR" \
        --step compute-metrics

    echo ""
    echo "Metrics computation complete."
    echo ""
else
    echo "========================================================================"
    echo "STEP 3/6: Metrics Computation (SKIPPED)"
    echo "========================================================================"
    echo "Using existing metrics in $EXPERIMENT_DIR/metrics/"
    echo ""

    if [[ ! -d "$EXPERIMENT_DIR/metrics" ]]; then
        echo "ERROR: Metrics directory not found: $EXPERIMENT_DIR/metrics"
        echo "       Cannot skip metrics without existing data."
        exit 1
    fi
fi

# ============================================================================
# Step 3b: Statistics Aggregation (STATS-ONLY MODE)
# ============================================================================

if [[ "$STATS_ONLY" == "true" ]]; then
    echo "========================================================================="
    echo "STEP 3b/7: Statistics Aggregation (Stats-Only Mode)"
    echo "========================================================================="
    echo "Computing statistics from existing metrics..."
    echo ""
    
    python3 scripts/aggregate_metrics.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --stats-only

    echo ""
    echo "Statistics computation complete."
    echo "Continuing with sample selection, reports, and gallery build..."
    echo ""
fi

# ============================================================================
# Step 4: Best Sample Selection
# ============================================================================

echo "========================================================================"
echo "STEP 4/6: Best Sample Selection"
echo "========================================================================"
echo "Selecting top $GALLERY_SAMPLES samples per condition..."
echo ""

python3 scripts/select_best_samples.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --samples-per-condition "$GALLERY_SAMPLES" \
    --output "$EXPERIMENT_DIR/selected_samples.json"

echo ""
echo "Sample selection complete."
echo ""

# ============================================================================
# Step 5: Report Generation
# ============================================================================

echo "========================================================================"
echo "STEP 5/6: Report Generation"
echo "========================================================================"
echo "Generating tables and figures..."
echo ""

python3 scripts/generate_report_figures.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --output-dir "$EXPERIMENT_DIR/reports"

echo ""
echo "Report generation complete."
echo ""

# ============================================================================
# Step 6: Gallery Build
# ============================================================================

echo "========================================================================"
echo "STEP 6/6: Gallery Build"
echo "========================================================================"
echo "Building GitHub Pages gallery..."
echo ""

python3 scripts/build_gallery.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --output-dir docs \
    --max-samples "$GALLERY_SAMPLES" \
    --selected-samples "$EXPERIMENT_DIR/selected_samples.json"

echo ""
echo "Gallery build complete."
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Outputs:"
echo ""
echo "  Experiment directory: $EXPERIMENT_DIR/"
echo "    ├── {condition}/           # Synthesized audio (50 samples each)"
echo "    ├── filelists/             # Sample manifests"
echo "    ├── transcriptions/        # Whisper transcriptions (original + synthesized)"
echo "    ├── metrics/               # Per-sample metrics CSVs"
echo "    ├── results_summary.csv    # Aggregated mean ± std"
echo "    ├── selected_samples.json  # Best $((GALLERY_SAMPLES * 8)) samples"
echo "    └── reports/               # Tables and figures"
echo ""
echo "  Gallery: docs/"
echo "    ├── index.html             # Overview"
echo "    ├── results.html           # Metrics table"
echo "    ├── {condition}.html       # Audio players"
echo "    └── audio/                 # Best samples"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review reports:"
echo "     ls $EXPERIMENT_DIR/reports/"
echo ""
echo "  2. Preview gallery locally:"
echo "     cd docs && python3 -m http.server 8000"
echo "     # Open http://localhost:8000"
echo ""
echo "  3. Deploy to GitHub Pages:"
echo "     git add docs/"
echo "     git commit -m 'Update experiment gallery'"
echo "     git push"
echo "     # Enable Pages in repo settings -> Source: main branch, /docs folder"
echo ""
echo "========================================================================"
