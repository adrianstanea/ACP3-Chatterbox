#!/usr/bin/env bash
#
# Romanian Chatterbox Full Experiment Pipeline
#
# Single script to run or redo the entire experiment:
#   1. Sample generation across parameter grid
#   2. Metrics computation (CER, WER, UTMOS, etc.)
#   3. Best sample selection
#   4. Report generation (tables + figures)
#   5. Gallery build for GitHub Pages
#
# Usage:
#   ./scripts/run_full_experiment.sh [options]
#
# Options:
#   --skip-synthesis       Skip sample generation (use existing)
#   --skip-metrics         Skip metrics computation (use existing)
#   --samples N            Samples per condition (default: 50)
#   --gallery-samples N    Best samples per condition for gallery (default: 5)
#   --whisper-model MODEL  Whisper model: tiny|base|small|medium|large (default: medium)
#   --experiment-dir DIR   Output directory (default: experiments/YYYY-MM-DD)
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
SKIP_METRICS=false

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Romanian Chatterbox Full Experiment Pipeline

Options:
    --skip-synthesis       Skip sample generation (use existing samples)
    --skip-metrics         Skip metrics computation (use existing metrics)
    --samples N            Samples per condition for synthesis (default: 50)
    --gallery-samples N    Best samples per condition for gallery (default: 5)
    --whisper-model MODEL  Whisper model: tiny|base|small|medium|large (default: medium)
    --experiment-dir DIR   Experiment directory (default: experiments/YYYY-MM-DD)
    --swara-dir DIR        SWARA dataset directory (default: /data/swara)
    --help                 Show this help message

Examples:
    # Full pipeline with defaults
    $(basename "$0")

    # Redo only metrics and reports (keep existing samples)
    $(basename "$0") --skip-synthesis

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
        --skip-metrics)
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
echo "  Skip metrics:       $SKIP_METRICS"
echo ""

# Check prerequisites
ERRORS=0

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
    echo "STEP 1/5: Sample Generation"
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
    echo "STEP 1/5: Sample Generation (SKIPPED)"
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
# Step 2: Metrics Computation
# ============================================================================

if [[ "$SKIP_METRICS" == "false" ]]; then
    echo "========================================================================"
    echo "STEP 2/5: Metrics Computation"
    echo "========================================================================"
    echo "Computing CER, WER, UTMOS, MCD, F0 RMSE, Speaker Similarity..."
    echo ""

    python3 scripts/compute_experiment_metrics.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --swara-dir "$SWARA_DIR" \
        --whisper-model "$WHISPER_MODEL"

    echo ""
    echo "Metrics computation complete."
    echo ""
else
    echo "========================================================================"
    echo "STEP 2/5: Metrics Computation (SKIPPED)"
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
# Step 3: Best Sample Selection
# ============================================================================

echo "========================================================================"
echo "STEP 3/5: Best Sample Selection"
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
# Step 4: Report Generation
# ============================================================================

echo "========================================================================"
echo "STEP 4/5: Report Generation"
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
# Step 5: Gallery Build
# ============================================================================

echo "========================================================================"
echo "STEP 5/5: Gallery Build"
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
