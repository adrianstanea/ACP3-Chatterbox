#!/usr/bin/env bash
#
# Romanian Chatterbox Evaluation Pipeline
#
# Complete pipeline: Metrics computation -> Sample selection -> Report generation -> Gallery
#
# Usage:
#   ./scripts/run_evaluation_pipeline.sh <experiment_dir> <swara_dir>
#
# Example:
#   ./scripts/run_evaluation_pipeline.sh experiments/2026-02-22 /data/swara
#
# Requirements:
#   - Python 3.12+ with required packages (whisper, jiwer, discrete_speech_metrics, etc.)
#   - Run inside Docker container for GPU support
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================================
# Parse Arguments
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") <experiment_dir> <swara_dir> [options]

Arguments:
    experiment_dir    Path to experiment directory (e.g., experiments/2026-02-22)
    swara_dir         Path to SWARA dataset directory

Options:
    --samples-per-condition N   Number of best samples per condition (default: 5)
    --whisper-model MODEL       Whisper model size: tiny|base|small|medium|large (default: medium)
    --skip-metrics              Skip metrics computation (use existing)
    --no-figures                Skip figure generation
    --help                      Show this help message

Example:
    $(basename "$0") experiments/2026-02-22 /data/swara --samples-per-condition 5

EOF
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

EXPERIMENT_DIR="$1"
SWARA_DIR="$2"
shift 2

# Defaults
SAMPLES_PER_CONDITION=5
WHISPER_MODEL="medium"
SKIP_METRICS=false
NO_FIGURES=false

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples-per-condition)
            SAMPLES_PER_CONDITION="$2"
            shift 2
            ;;
        --whisper-model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        --no-figures)
            NO_FIGURES=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

cd "$PROJECT_DIR"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

if [[ ! -d "$SWARA_DIR" ]]; then
    echo "ERROR: SWARA directory not found: $SWARA_DIR"
    exit 1
fi

echo "========================================================================"
echo "Romanian Chatterbox Evaluation Pipeline"
echo "========================================================================"
echo "Experiment:       $EXPERIMENT_DIR"
echo "SWARA:            $SWARA_DIR"
echo "Samples/cond:     $SAMPLES_PER_CONDITION"
echo "Whisper model:    $WHISPER_MODEL"
echo "Skip metrics:     $SKIP_METRICS"
echo "No figures:       $NO_FIGURES"
echo "========================================================================"
echo ""

# ============================================================================
# Step 1: Compute Metrics
# ============================================================================

if [[ "$SKIP_METRICS" == "false" ]]; then
    echo "========================================================================"
    echo "STEP 1: Computing Metrics"
    echo "========================================================================"

    python3 scripts/compute_experiment_metrics.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --swara-dir "$SWARA_DIR" \
        --whisper-model "$WHISPER_MODEL"

    echo ""
else
    echo "========================================================================"
    echo "STEP 1: Skipping metrics computation (using existing)"
    echo "========================================================================"
fi

# ============================================================================
# Step 2: Select Best Samples
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Selecting Best Samples"
echo "========================================================================"

python3 scripts/select_best_samples.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --samples-per-condition "$SAMPLES_PER_CONDITION" \
    --output "$EXPERIMENT_DIR/selected_samples.json"

echo ""

# ============================================================================
# Step 3: Generate Report Figures
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Generating Report Figures and Tables"
echo "========================================================================"

FIGURES_ARGS="--experiment-dir $EXPERIMENT_DIR --output-dir $EXPERIMENT_DIR/reports"
if [[ "$NO_FIGURES" == "true" ]]; then
    FIGURES_ARGS="$FIGURES_ARGS --no-figures"
fi

python3 scripts/generate_report_figures.py $FIGURES_ARGS

echo ""

# ============================================================================
# Step 4: Build Gallery
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 4: Building GitHub Pages Gallery"
echo "========================================================================"

python3 scripts/build_gallery.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --output-dir docs \
    --max-samples "$SAMPLES_PER_CONDITION" \
    --selected-samples "$EXPERIMENT_DIR/selected_samples.json"

echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Outputs:"
echo "  Metrics:          $EXPERIMENT_DIR/metrics/"
echo "  Results summary:  $EXPERIMENT_DIR/results_summary.csv"
echo "  Selected samples: $EXPERIMENT_DIR/selected_samples.json"
echo "  Reports:          $EXPERIMENT_DIR/reports/"
echo "  Gallery:          docs/"
echo ""
echo "Next steps:"
echo "  1. Review reports in $EXPERIMENT_DIR/reports/"
echo "  2. Preview gallery: cd docs && python3 -m http.server 8000"
echo "  3. Commit and push to deploy GitHub Pages"
echo ""
