# Academic Report Generation for Romanian Chatterbox TTS Adaptation

This document describes the comprehensive analysis and reporting pipeline for the Romanian Chatterbox TTS adaptation study using the SWARA dataset.

## Quick Start

### Prerequisites

Install dependencies:
```bash
pip install -r requirements-metrics.txt
pip install tabulate pytest
```

### Basic Usage

Generate a complete academic report with all outputs:
```bash
python scripts/generate_academic_report.py \
  --output-dir my_academic_report \
  --formats html pdf latex markdown json
```

This will:
1. Load data from the default paths (experiments/2026-02-23/metrics, etc.)
2. Run all analyses (dataset, training, performance, adaptation, ablation)
3. Generate visualizations (PNG and PDF)
4. Create tables in LaTeX and Markdown formats
5. Export JSON summaries for reproducibility

### Custom Data Paths

```bash
python scripts/generate_academic_report.py \
  --metrics-dir /path/to/metrics \
  --metadata /path/to/metadata.csv \
  --dataset-splits /path/to/splits \
  --training-log /path/to/trainer_state.json \
  --output-dir my_report \
  --dpi 300
```

## Generated Outputs

### Visualizations

**Dataset Characterization:**
- Speaker distribution (number of samples per speaker)
- Text statistics (character/word length distributions)

**Training Dynamics:**
- Loss curves showing training and validation loss over epochs

**Evaluation Results:**
- Performance comparison: 4-subplot grid showing CER, WER, UTMOS, Speaker Similarity across all 8 conditions

**Ablation Analysis:**
- (Future: interaction plots, heatmaps)

### Tables

**Markdown Format** (readable in documentation):
- `dataset_summary.md` — Dataset characterization summary
- `metrics_by_condition.md` — Per-condition metrics with uncertainty
- `ablation_table.md` — Parameter ablation effects

**LaTeX Format** (ready for inclusion in papers):
- `dataset_summary.tex` — Can be included via `\input{}`
- `metrics_by_condition.tex` — Publication-ready metrics table
- `ablation_table.tex` — Ablation study results table

### Data Files (JSON)

- `dataset_characterization.json` — Speaker counts, text statistics, split information
- `performance_statistics.json` — Per-condition metrics with statistical tests and p-values
- `training_summary.json` — Training metadata (epochs, convergence)
- `ablation_results.json` — Parameter effects and synthesizer comparisons

## Study Design

### Dataset
- **SWARA 1.0** — Romanian TTS dataset
- **Train/Val/Test Split** with **2 holdout speakers** for 1-shot voice adaptation evaluation
- Features captured: text characteristics, audio duration, speaker demographics

### Experimental Conditions
8 conditions testing parameter variations:
- **Synthesizers**: BAS (baseline) vs SGS (spectrogram-guided synthesis)
- **Exaggeration**: 0.3 vs 0.7
- **Guidance Scale**: 0.3 vs 0.5

### Metrics
- **CER** (Character Error Rate) — transcription accuracy
- **WER** (Word Error Rate) — transcription accuracy at word level
- **UTMOS** — Mean Opinion Score proxy (quality assessment)
- **Speaker Cosine Similarity** — voice identity preservation

## Key Features

### 1. Holdout Speaker Analysis
The pipeline specifically tracks and analyzes performance on 2 unseen speakers (1-shot voice adaptation), highlighting the model's ability to generalize to new speakers with minimal data.

### 2. Statistical Rigor
All pairwise comparisons include:
- Independent samples t-tests with p-values
- Cohen's d effect sizes
- Sample counts (n) for transparency

### 3. Publication-Ready Output
- Figures: High-DPI PNG (web/presentations) + PDF (papers)
- Tables: Both readable Markdown and LaTeX-ready formats
- JSON metadata: Complete reproducibility trail

### 4. Flexible Output
Control exactly what gets generated:
```bash
# Only PDF figures and JSON data
python scripts/generate_academic_report.py --formats pdf json

# LaTeX tables + HTML report
python scripts/generate_academic_report.py --formats latex html

# Default: everything
python scripts/generate_academic_report.py
```

## API Reference

See `scripts/report_lib/README.md` for complete API documentation.

### Example: Custom Analysis

```python
from report_lib.loaders import load_metrics_csvs, load_dataset_metadata
from report_lib.analysis import analyze_performance
from report_lib.statistics import compare_conditions
from report_lib.utils import save_json

# Load data
metrics = load_metrics_csvs("experiments/2026-02-23/metrics")

# Analyze performance
perf_analysis = analyze_performance(metrics)

# Get statistical comparisons
stats = compare_conditions(metrics, "CER")

# Save for later
save_json({"performance": perf_analysis, "stats": stats}, "my_analysis.json")

# Access results
best_cer_condition = min(
    perf_analysis["conditions"].items(),
    key=lambda x: x[1]["CER"]["mean"]
)
print(f"Best CER: {best_cer_condition[0]} = {best_cer_condition[1]['CER']['mean']:.3f}")
```

## Testing

Run all tests:
```bash
PYTHONPATH=scripts python -m pytest tests/ -v
```

Run specific test module:
```bash
PYTHONPATH=scripts python -m pytest tests/test_analysis.py -v
```

Test coverage includes:
- Data loading from all sources ✓
- Analysis functions ✓
- Visualization generation ✓
- Table export (LaTeX/Markdown) ✓
- Statistical testing ✓
- Integration tests ✓

## Reproducibility

All outputs are JSON-serializable for complete reproducibility:
- Exact metrics values for all conditions
- Statistical test results (p-values, effect sizes)
- Dataset statistics (speaker counts, text properties)
- Training metadata (epochs, convergence)

## Future Enhancements

- [ ] Interactive HTML dashboard with plotly
- [ ] Automatic paper figure numbering and references
- [ ] Confidence interval visualization
- [ ] Effect size categorization (small/medium/large)
- [ ] Normalized metric comparison across experiments
- [ ] Automatic report writing (markdown skeleton)

## Troubleshooting

### "No CSV files found in metrics directory"
Ensure the metrics directory path is correct and contains `.csv` files.

### "Metadata file not found"
Check that `data/metadata_SWARA1.0_text.csv` exists or provide explicit path via `--metadata`.

### Missing matplotlib display
Matplotlib is configured to save to files only (no display). All plots are saved as PNG and PDF.

### ImportError: No module named 'report_lib'
Set PYTHONPATH: `export PYTHONPATH=/path/to/scripts:$PYTHONPATH`

## Contact

For questions about the Romanian Chatterbox TTS adaptation project, see the main project README.
