# Academic Report Generation Library

Comprehensive analysis and visualization pipeline for Romanian Chatterbox TTS adaptation study on the SWARA dataset.

## Overview

This library provides a complete pipeline for:
- Loading and validating data from multiple sources
- Computing comprehensive analyses (dataset, training, performance, voice adaptation, ablation)
- Generating publication-quality visualizations
- Creating formatted tables (LaTeX/Markdown)
- Running statistical tests and effect size calculations
- Exporting results as JSON for reproducibility

## Modules

### `loaders.py` - Data Loading
- `load_metrics_csvs()` — Load per-condition evaluation metrics from CSV files
- `load_trainer_state()` — Parse training state and loss curves from JSON
- `load_dataset_metadata()` — Load SWARA dataset metadata (speakers, text, duration)
- `load_dataset_splits()` — Load train/val/test splits with holdout speaker identification
- `load_filelists()` — Load sample file lists per condition
- `validate_loaded_data()` — Verify data integrity and completeness

### `analysis.py` - Analysis Functions
- `analyze_dataset_characteristics()` — Speaker distribution, text statistics, audio properties
- `analyze_training_dynamics()` — Loss curves, convergence metrics, best checkpoints
- `analyze_performance()` — Per-condition metrics with overall statistics (mean, std, min, max, median)
- `analyze_voice_adaptation()` — 1-shot adaptation performance on holdout speakers
- `analyze_ablation_effects()` — Parameter ablation effects (synthesizer, exaggeration, guidance)

### `visualizations.py` - Plotting Functions
- `plot_speaker_distribution()` — Bar plot of samples per speaker
- `plot_text_statistics()` — Text length and word count distributions
- `plot_loss_curves()` — Training and validation loss over epochs
- `plot_performance_comparison()` — 4-subplot grid comparing metrics across conditions

### `tables.py` - Table Generation
- `generate_metrics_table()` — Create formatted metrics table with ± notation
- `generate_dataset_summary_table()` — Dataset characterization summary
- `generate_ablation_table()` — Ablation study results
- `export_to_latex()` — Export to compilable LaTeX tables
- `export_to_markdown()` — Export to GitHub-flavored Markdown

### `statistics.py` - Statistical Testing
- `compute_effect_size()` — Calculate Cohen's d effect size
- `run_ttest()` — Independent samples t-test with p-values and effect sizes
- `compare_conditions()` — Pairwise comparisons of all condition pairs

### `utils.py` - Utilities
- `ensure_output_dirs()` — Create output directory structure
- `save_json()` — Save data to JSON with error handling
- `load_json()` — Load JSON data safely

## Usage

### Command Line

```bash
python scripts/generate_academic_report.py \
  --metrics-dir experiments/2026-02-23/metrics \
  --training-log vendor/chatterbox-finetuning/chatterbox_output/trainer_state.json \
  --metadata data/metadata_SWARA1.0_text.csv \
  --dataset-splits data/processed/MyTTSDataset/splits \
  --output-dir academic_report \
  --formats pdf latex markdown json \
  --dpi 300 \
  --style academic
```

### Programmatic Usage

```python
from report_lib.loaders import load_metrics_csvs, load_dataset_metadata
from report_lib.analysis import analyze_dataset_characteristics
from report_lib.visualizations import plot_speaker_distribution

# Load data
metrics = load_metrics_csvs("experiments/2026-02-23/metrics")
metadata = load_dataset_metadata("data/metadata_SWARA1.0_text.csv")

# Analyze
analysis = analyze_dataset_characteristics(metadata, splits={})

# Visualize
plot_speaker_distribution(
    analysis["speaker_distribution"]["per_speaker"],
    output_dir="output/figures",
    dpi=300
)
```

## Output Structure

```
academic_report/
├── figures/
│   ├── dataset_characterization/
│   │   ├── speaker_distribution.png
│   │   ├── text_statistics.png
│   │   ├── speaker_distribution.pdf
│   │   └── text_statistics.pdf
│   ├── training_dynamics/
│   │   ├── loss_curves.png
│   │   └── loss_curves.pdf
│   ├── evaluation_results/
│   │   ├── performance_comparison.png
│   │   └── performance_comparison.pdf
│   └── ablation_analysis/
│       └── (future visualizations)
├── tables/
│   ├── dataset_summary.tex
│   ├── dataset_summary.md
│   ├── metrics_by_condition.tex
│   ├── metrics_by_condition.md
│   ├── ablation_table.tex
│   └── ablation_table.md
├── data/
│   ├── dataset_characterization.json
│   ├── performance_statistics.json
│   ├── training_summary.json
│   └── ablation_results.json
└── report.html (placeholder)
```

## Statistical Methods

### Effect Size (Cohen's d)
Calculated as the standardized difference between condition means:
```
d = (mean1 - mean2) / pooled_std
```

### Pairwise Comparisons
Welch's t-test (does not assume equal variances) is used for all pairwise condition comparisons. P-values < 0.05 are marked as statistically significant.

## Data Format Requirements

### Metrics CSV
Columns: `CER`, `WER`, `UTMOS_Synthesized`, `Speaker_Cosine_Similarity`

### Metadata CSV
Columns: `speaker`, `text`, `duration`

### Split CSVs
Columns: `speaker`, `utterance` (or similar)
Files: `train.csv`, `val.csv`, `test_holdout.csv`

### Trainer State JSON
Keys: `epoch`, `global_step`, `training_loss`, `validation_loss` (optional)

## Design Principles

1. **Modularity** — Each function has a single responsibility
2. **Testability** — All functions have unit tests
3. **Reproducibility** — All parameters and data are saved as JSON
4. **Publication-Ready** — Figures and tables are formatted for academic papers
5. **Graceful Degradation** — Missing optional data doesn't crash the pipeline

## Testing

Run all tests:
```bash
PYTHONPATH=scripts python -m pytest tests/ -v
```

Run specific test module:
```bash
PYTHONPATH=scripts python -m pytest tests/test_analysis.py -v
```

## Dependencies

- pandas >= 2.0.0
- numpy >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.11.0
- jinja2 >= 3.1.0
- tabulate (for Markdown export)

## Author

Claude Opus 4.6 (Co-Authored)
