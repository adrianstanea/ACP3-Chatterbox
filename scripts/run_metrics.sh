#!/usr/bin/env bash
#
# Metrics Pipeline Wrapper
# Chains TTS-Evaluation-Pipeline steps for all filelists in an experiment.
#
# Usage:
#   ./run_metrics.sh <experiment_dir> <swara_dir>
#
# Example:
#   ./run_metrics.sh ./experiments/exp001 /data/swara/
#
# Environment Variables:
#   PIPELINE_DIR - Path to TTS-Evaluation-Pipeline (default: ~/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline)
#   WHISPER_MODEL - Whisper model size (default: medium)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

PIPELINE_DIR="${PIPELINE_DIR:-/home/astanea/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Usage & Argument Parsing
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") <experiment_dir> <swara_dir>

Arguments:
    experiment_dir    Path to experiment directory (must contain filelists/ and audio/)
    swara_dir         Path to SWARA dataset directory (original audio files)

Environment Variables:
    PIPELINE_DIR      Path to TTS-Evaluation-Pipeline
                      (default: /home/astanea/git-repos/adrianstanea/OpenTTS/TTS-Evaluation-Pipeline)

Example:
    $(basename "$0") ./experiments/exp001 /data/swara/

EOF
    exit 1
}

if [[ $# -ne 2 ]]; then
    echo "ERROR: Expected 2 arguments, got $#"
    usage
fi

EXPERIMENT_DIR="$1"
SWARA_DIR="$2"

# ============================================================================
# Validation
# ============================================================================

# Convert to absolute paths
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" 2>/dev/null && pwd)" || {
    echo "ERROR: Experiment directory does not exist: $1"
    exit 1
}

SWARA_DIR="$(cd "$SWARA_DIR" 2>/dev/null && pwd)" || {
    echo "ERROR: SWARA directory does not exist: $2"
    exit 1
}

# Check pipeline directory
if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "ERROR: Pipeline directory not found: $PIPELINE_DIR"
    echo "Set PIPELINE_DIR environment variable to correct path"
    exit 1
fi

# Check for pipeline scripts
if [[ ! -f "$PIPELINE_DIR/transcription.py" ]]; then
    echo "ERROR: transcription.py not found in $PIPELINE_DIR"
    exit 1
fi

if [[ ! -f "$PIPELINE_DIR/metrics.py" ]]; then
    echo "ERROR: metrics.py not found in $PIPELINE_DIR"
    exit 1
fi

# Check experiment directory structure
FILELISTS_DIR="$EXPERIMENT_DIR/filelists"
AUDIO_DIR="$EXPERIMENT_DIR/audio"

if [[ ! -d "$FILELISTS_DIR" ]]; then
    echo "ERROR: filelists directory not found: $FILELISTS_DIR"
    exit 1
fi

if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "ERROR: audio directory not found: $AUDIO_DIR"
    exit 1
fi

# ============================================================================
# Setup Output Directories
# ============================================================================

TRANSCRIPTIONS_DIR="$EXPERIMENT_DIR/transcriptions"
METRICS_DIR="$EXPERIMENT_DIR/metrics"

mkdir -p "$TRANSCRIPTIONS_DIR"
mkdir -p "$METRICS_DIR"

echo "============================================================"
echo "Metrics Pipeline"
echo "============================================================"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "SWARA directory:      $SWARA_DIR"
echo "Pipeline directory:   $PIPELINE_DIR"
echo "Transcriptions:       $TRANSCRIPTIONS_DIR"
echo "Metrics:              $METRICS_DIR"
echo "============================================================"
echo ""

# ============================================================================
# Process Each Filelist
# ============================================================================

# Count filelists
FILELIST_COUNT=$(find "$FILELISTS_DIR" -name "*.csv" -type f | wc -l)
if [[ $FILELIST_COUNT -eq 0 ]]; then
    echo "ERROR: No CSV filelists found in $FILELISTS_DIR"
    exit 1
fi

echo "Found $FILELIST_COUNT filelist(s) to process"
echo ""

PROCESSED=0
FAILED=0

for FILELIST in "$FILELISTS_DIR"/*.csv; do
    # Extract condition name from filename (without path and extension)
    CONDITION=$(basename "$FILELIST" .csv)

    echo "------------------------------------------------------------"
    echo "Processing condition: $CONDITION"
    echo "------------------------------------------------------------"

    # Determine audio subdirectory for this condition
    CONDITION_AUDIO_DIR="$AUDIO_DIR/$CONDITION"

    if [[ ! -d "$CONDITION_AUDIO_DIR" ]]; then
        echo "WARNING: Audio directory not found for condition: $CONDITION_AUDIO_DIR"
        echo "Skipping..."
        ((FAILED++)) || true
        continue
    fi

    # ========================================================================
    # Step 1: Transcription (Whisper)
    # ========================================================================

    TRANSCRIPTION_OUTPUT="$TRANSCRIPTIONS_DIR/${CONDITION}.csv"

    echo ""
    echo "Step 1: Running transcription with Whisper..."
    echo "  Input filelist: $FILELIST"
    echo "  Synthesized audio: $CONDITION_AUDIO_DIR"
    echo "  Output: $TRANSCRIPTION_OUTPUT"

    # Change to pipeline directory for proper imports
    pushd "$PIPELINE_DIR" > /dev/null

    # Run transcription
    # Note: transcription.py outputs to CWD with generator_model name
    if python transcription.py \
        --input_filelist "$FILELIST" \
        --base_dir_synthesized "$CONDITION_AUDIO_DIR" \
        --generator_model "$CONDITION"; then

        # Move output to transcriptions directory
        if [[ -f "${CONDITION}.csv" ]]; then
            mv "${CONDITION}.csv" "$TRANSCRIPTION_OUTPUT"
            echo "  Transcription complete: $TRANSCRIPTION_OUTPUT"
        else
            echo "WARNING: Transcription output not found"
        fi
    else
        echo "ERROR: Transcription failed for $CONDITION"
        popd > /dev/null
        ((FAILED++)) || true
        continue
    fi

    popd > /dev/null

    # ========================================================================
    # Step 2: Compute Metrics
    # ========================================================================

    METRICS_OUTPUT="$METRICS_DIR/${CONDITION}.csv"

    echo ""
    echo "Step 2: Computing metrics..."
    echo "  Input filelist: $TRANSCRIPTION_OUTPUT"
    echo "  Synthesized audio: $CONDITION_AUDIO_DIR"
    echo "  Original audio: $SWARA_DIR"
    echo "  Output: $METRICS_OUTPUT"

    # Change to pipeline directory for proper imports
    pushd "$PIPELINE_DIR" > /dev/null

    # Run metrics computation
    # Note: metrics.py creates output in SAVE_DIR="results" subdirectory
    if python metrics.py \
        --input_filelist "$TRANSCRIPTION_OUTPUT" \
        --base_dir_synthesized "$CONDITION_AUDIO_DIR" \
        --base_dir_original "$SWARA_DIR" \
        --output_csv "${CONDITION}.csv"; then

        # Move output from results/ to metrics directory
        if [[ -f "results/${CONDITION}.csv" ]]; then
            mv "results/${CONDITION}.csv" "$METRICS_OUTPUT"
            echo "  Metrics complete: $METRICS_OUTPUT"
        else
            echo "WARNING: Metrics output not found in results/"
        fi
    else
        echo "ERROR: Metrics computation failed for $CONDITION"
        popd > /dev/null
        ((FAILED++)) || true
        continue
    fi

    popd > /dev/null

    ((PROCESSED++)) || true
    echo ""
    echo "Condition $CONDITION completed successfully"
    echo ""
done

# ============================================================================
# Aggregate Results
# ============================================================================

echo "============================================================"
echo "Aggregating Results"
echo "============================================================"

# Run aggregate_metrics.py
AGGREGATE_SCRIPT="$SCRIPT_DIR/aggregate_metrics.py"

if [[ ! -f "$AGGREGATE_SCRIPT" ]]; then
    echo "WARNING: aggregate_metrics.py not found at $AGGREGATE_SCRIPT"
    echo "Skipping aggregation step"
else
    echo "Running metrics aggregation..."

    if python "$AGGREGATE_SCRIPT" --experiment-dir "$EXPERIMENT_DIR"; then
        echo "Aggregation complete"
    else
        echo "ERROR: Aggregation failed"
        ((FAILED++)) || true
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
echo "Processed: $PROCESSED conditions"
echo "Failed:    $FAILED conditions"
echo ""
echo "Outputs:"
echo "  Transcriptions: $TRANSCRIPTIONS_DIR/"
echo "  Metrics:        $METRICS_DIR/"
echo "  Summary:        $EXPERIMENT_DIR/results_summary.csv"
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

exit 0
