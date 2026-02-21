# Academic Reporting Quick Reference Guide

**Purpose:** Extract statistics, tables, and plots from project documentation for academic papers and reports.

**Last Updated:** February 21, 2026

## Overview

This guide helps you quickly find the information needed for academic papers, project reports, and presentations. All data is sourced from verified project documentation.

## For Abstract/Executive Summary

**One-Paragraph Summary:**
> This project adapts the Chatterbox text-to-speech model to Romanian by fine-tuning the 500M parameter multilingual variant on the SWARA 1.0 dataset (21,304 utterances, 18 speakers, 21.67 hours). Using a Llama 3-based transformer with character-level tokenization, we enable high-quality Romanian speech synthesis with zero-shot voice cloning capabilities. The model is trained using per-speaker stratified validation with holdout speakers for robust evaluation.

**Source:** [Project Overview](project-overview.md#executive-summary)

## Dataset Statistics

### Quick Stats Table (for papers)

```latex
\begin{table}[h]
\centering
\begin{tabular}{lr}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Total utterances & 21,304 \\
Total duration & 21.67 hours \\
Speakers & 18 (16 train, 2 holdout) \\
Mean duration & 3.66 $\pm$ 2.1 seconds \\
Sample rate & 22,050 Hz \\
Format & WAV 16-bit mono \\
Diacritics & Complete (ă, â, î, ș, ț) \\
\hline
\end{tabular}
\caption{SWARA 1.0 Dataset Statistics}
\label{tab:swara-stats}
\end{table}
```

**Source:** [SWARA Analysis Report](data/swara-analysis-report.md#overall-statistics)

### Data Split Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lrr}
\hline
\textbf{Split} & \textbf{Utterances} & \textbf{Duration (hours)} \\
\hline
Training (16 speakers) & 17,013 & 19.5 \\
Validation (16 speakers) & 1,891 & 2.2 \\
Test (BAS, SGS holdout) & 2,400 & 2.7 \\
\hline
\textbf{Total} & \textbf{21,304} & \textbf{21.67} \\
\hline
\end{tabular}
\caption{Dataset Split Strategy}
\label{tab:data-split}
\end{table}
```

**Source:** [Technical Decisions #4](technical-decisions.md#decision-4-validation-split-strategy)

### Romanian Diacritics Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{clc}
\hline
\textbf{Character} & \textbf{Name} & \textbf{Status} \\
\hline
ă, Ă & a-breve & ✓ \\
â, Â & a-circumflex & ✓ \\
î, Î & i-circumflex & ✓ \\
ș, Ș & s-comma & ✓ \\
ț, Ț & t-comma & ✓ \\
\hline
\end{tabular}
\caption{Romanian Diacritics Coverage in SWARA Dataset}
\label{tab:diacritics}
\end{table}
```

**Source:** [SWARA Analysis Report](data/swara-analysis-report.md#romanian-diacritics)

## Model Architecture

### Architecture Comparison Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lll}
\hline
\textbf{Variant} & \textbf{Multilingual} & \textbf{Turbo} \\
\hline
Backbone & Llama 3 & GPT-2 \\
Parameters & 500M & 350M \\
Tokenizer & Grapheme (2,454) & BPE (52,260) \\
Languages & 23 & English-focused \\
New language stability & High & Low (Issues \#6, \#12) \\
\hline
\textbf{Our choice} & \checkmark & \\
\hline
\end{tabular}
\caption{Chatterbox Model Variants Comparison}
\label{tab:model-comparison}
\end{table}
```

**Source:** [Technical Decisions #1](technical-decisions.md#decision-1-chatterbox-multilingual-not-turbo)

### Model Components Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lll}
\hline
\textbf{Component} & \textbf{Architecture} & \textbf{Training Status} \\
\hline
T3 Transformer & Llama 3 (500M) & Trainable \\
Voice Encoder & ResNet-based & Frozen \\
S3Gen Vocoder & HiFi-GAN derived & Frozen \\
\hline
\end{tabular}
\caption{Chatterbox Model Components}
\label{tab:components}
\end{table}
```

**Source:** [Design Document §1](plans/2026-02-21-chatterbox-romanian-design.md#trainable-components)

## Training Configuration

### Hyperparameters Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lr}
\hline
\textbf{Hyperparameter} & \textbf{Value} \\
\hline
Precision & FP16 \\
Batch size (per GPU) & 4 \\
Gradient accumulation & 8 \\
Effective batch size & 32 \\
Learning rate & $1 \times 10^{-5}$ \\
Epochs & 120 \\
Optimizer & AdamW \\
Gradient checkpointing & Enabled \\
Text max length & 256 tokens \\
Speech max length & 850 tokens \\
\hline
\end{tabular}
\caption{Training Hyperparameters for V100 32GB}
\label{tab:hyperparams}
\end{table}
```

**Source:** [Design Document §4](plans/2026-02-21-chatterbox-romanian-design.md#hardware-v100-32gb)

## Evaluation Framework

### Metrics Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{llp{6cm}}
\hline
\textbf{Category} & \textbf{Metric} & \textbf{Description} \\
\hline
Intelligibility & WER & Word Error Rate via Romanian ASR \\
Intelligibility & CER & Character Error Rate \\
Spectral & MCD & Mel Cepstral Distortion \\
Spectral & RMSE & Root Mean Square Error \\
Perceptual & PESQ & Perceptual Evaluation of Speech Quality \\
Perceptual & STOI & Short-Time Objective Intelligibility \\
Signal & SI-SDR & Scale-Invariant Signal-to-Distortion \\
Speaker & SECS & Speaker Embedding Cosine Similarity \\
Subjective & MOS & Mean Opinion Score (1-5) \\
\hline
\end{tabular}
\caption{Evaluation Metrics}
\label{tab:metrics}
\end{table}
```

**Source:** [Design Document §5](plans/2026-02-21-chatterbox-romanian-design.md#metrics)

### Evaluation Splits Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{lp{8cm}}
\hline
\textbf{Evaluation} & \textbf{Description} \\
\hline
In-speaker & Validation set using same speaker references. Measures quality on known speakers. \\
Zero-shot & BAS and SGS holdout speakers. Tests generalization to unseen Romanian speakers. \\
Diacritic stress test & Minimal pairs (e.g., "paturi" vs "pături"). Verifies diacritic pronunciation. \\
\hline
\end{tabular}
\caption{Evaluation Strategy}
\label{tab:eval-strategy}
\end{table}
```

**Source:** [Design Document §5](plans/2026-02-21-chatterbox-romanian-design.md#evaluation-splits)

## Technical Decisions Summary

### Key Decisions Table

```latex
\begin{table*}[t]
\centering
\begin{tabular}{lp{4cm}p{6cm}p{3cm}}
\hline
\textbf{Decision} & \textbf{Choice} & \textbf{Rationale} & \textbf{Impact} \\
\hline
Model variant & Multilingual & Stability, grapheme tokenizer, cross-lingual transfer & Architecture \\
Environment & PyTorch 26.01 (Python 3.12) & Latest CUDA, verified compatibility & Infrastructure \\
Vendor code & Git submodule (fork) & Version control, modification capability & Code organization \\
Data split & Per-speaker stratification + holdout & Prevents overfitting, enables zero-shot eval & Training/Evaluation \\
Text norm & Chatterbox punc\_norm & Consistency with pretraining & Data preparation \\
\hline
\end{tabular}
\caption{Summary of Key Technical Decisions}
\label{tab:decisions}
\end{table*}
```

**Source:** [Technical Decisions Summary Table](technical-decisions.md#summary-table)

## Environment Specifications

### Software Stack Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{ll}
\hline
\textbf{Component} & \textbf{Version/Specification} \\
\hline
Base Image & nvcr.io/nvidia/pytorch:26.01-py3 \\
CUDA & 13.1 \\
PyTorch & 2.10 \\
Python & 3.12 \\
OS & Ubuntu 22.04 \\
Container & Docker Compose \\
GPU & NVIDIA V100 32GB \\
\hline
\end{tabular}
\caption{Software and Hardware Environment}
\label{tab:environment}
\end{table}
```

**Source:** [Environment Setup](setup/environment-setup.md#base-image-selection)

## Citations and References

### Dataset Citation

```bibtex
@misc{swara2023,
  title={SWARA 1.0: Speech in West And East RomaniA},
  author={SWARA Consortium},
  year={2023},
  howpublished={\url{https://github.com/crosslingual-voices/SWARA1.0}},
  note={Licensed under CC BY-NC 4.0}
}
```

**Source:** [SWARA Analysis Report](data/swara-analysis-report.md#license-and-attribution)

### Model Citation

```bibtex
@software{chatterbox2024,
  title={Chatterbox: Open-Source Text-to-Speech},
  author={Resemble AI},
  year={2024},
  howpublished={\url{https://github.com/ResembleAI/chatterbox}},
  note={Multilingual variant (500M parameters)}
}
```

**Source:** [Project Overview](project-overview.md#references)

### Fine-Tuning Kit Citation

```bibtex
@software{chatterbox-finetuning2024,
  title={Chatterbox Fine-Tuning Kit},
  author={Eraslan, Gokhan},
  year={2024},
  howpublished={\url{https://github.com/gokhaneraslan/chatterbox-finetuning}},
  note={Community-maintained fine-tuning infrastructure}
}
```

**Source:** [Vendor Analysis](vendor/chatterbox-finetuning-analysis.md)

## Figures and Plots (To Be Generated)

### Recommended Visualizations

**1. Duration Distribution Histogram**
- X-axis: Duration (seconds)
- Y-axis: Count
- Show mean, median lines
- Highlight outliers
- **Source data:** Run `scripts/analyze_swara.py`

**2. Speaker Distribution Bar Chart**
- X-axis: Speaker ID
- Y-axis: Utterance count / Duration
- Separate colors for train/val/test
- **Source data:** Run `scripts/analyze_swara.py`

**3. Training Loss Curve**
- X-axis: Steps/Epochs
- Y-axis: Loss (text + speech)
- Show smoothed trend
- **Source data:** TensorBoard logs (after Task 7)

**4. Evaluation Metrics Comparison**
- X-axis: Metrics (WER, CER, MCD, etc.)
- Y-axis: Score
- Compare in-speaker vs zero-shot
- **Source data:** Evaluation results (after Task 10)

**5. Character Frequency Distribution**
- X-axis: Characters (sorted by frequency)
- Y-axis: Count
- Highlight Romanian diacritics
- **Source data:** Run `scripts/analyze_swara.py`

## Quick Copy-Paste Statistics

### For Abstract

- Dataset: 21,304 utterances, 21.67 hours, 18 speakers
- Model: 500M parameter Llama 3-based transformer
- Languages: Romanian (new) + 23 pretrained
- Training: Per-speaker stratified split, holdout evaluation

### For Introduction

- Romanian TTS gap: No open-source neural TTS models
- Chatterbox: State-of-the-art with 23 languages (excluding Romanian)
- SWARA: High-quality read speech corpus for Romanian
- Contribution: First open-source Romanian Chatterbox adaptation

### For Methods

- Architecture: Chatterbox Multilingual (not Turbo)
- Tokenizer: Grapheme (2,454 tokens), extended if needed
- Training: T3 transformer trainable, VE + vocoder frozen
- Data: 90/10 per-speaker split, BAS+SGS holdout
- Hyperparameters: See [Training Configuration](#hyperparameters-table)

### For Results

- TBD after Task 7-10 (training and evaluation)

### For Discussion

- Model choice: Multilingual stability vs Turbo failures
- Data split: Zero-shot capability critical for voice cloning
- Limitations: Read speech only, 21 hours (moderate size)
- Future work: Spontaneous speech, data augmentation

## Document Sources Map

| Information Needed | Primary Source | Secondary Source |
|-------------------|----------------|------------------|
| Project overview | [project-overview.md](project-overview.md) | [Design doc](plans/2026-02-21-chatterbox-romanian-design.md) |
| Dataset statistics | [swara-analysis-report.md](data/swara-analysis-report.md) | [Data prep notes](data/data-preparation-notes.md) |
| Model architecture | [Design doc §1](plans/2026-02-21-chatterbox-romanian-design.md#1-architecture--model-choice) | [Technical decision #1](technical-decisions.md#decision-1-chatterbox-multilingual-not-turbo) |
| Training config | [Design doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration) | [Vendor analysis](vendor/chatterbox-finetuning-analysis.md) |
| Data preparation | [data-preparation-notes.md](data/data-preparation-notes.md) | [Technical decision #5](technical-decisions.md#decision-5-text-normalization-approach) |
| Environment | [environment-setup.md](setup/environment-setup.md) | [Technical decision #2](technical-decisions.md#decision-2-pytorch-2601-python-312) |
| Evaluation plan | [Design doc §5](plans/2026-02-21-chatterbox-romanian-design.md#5-evaluation-framework) | [Technical decision #4](technical-decisions.md#decision-4-validation-split-strategy) |
| Technical rationale | [technical-decisions.md](technical-decisions.md) | Specific design sections |

## Generating New Statistics

### SWARA Analysis Script

**Run full analysis:**
```bash
python scripts/analyze_swara.py \
    metadata_SWARA1.0_text.csv \
    --audio-dir /data/swara \
    > swara_analysis_output.txt
```

**Extract for paper:**
- Duration statistics: Section "DURATION ANALYSIS"
- Speaker statistics: Section "SPEAKER STATISTICS"
- Text analysis: Section "TEXT ANALYSIS"
- Character inventory: Section "FULL CHARACTER INVENTORY"

### Conversion Statistics

**After running conversion:**
```bash
python scripts/convert_swara_to_ljspeech.py ... \
    > conversion_log.txt
```

**Extract:**
- Total files converted
- Train/val split counts
- Holdout speaker exclusions
- Diacritic preservation verification

### Training Metrics (After Task 7)

**TensorBoard:**
```bash
tensorboard --logdir vendor/chatterbox-finetuning/chatterbox_output
```

**Export metrics:**
- Loss curves (JSON or CSV)
- Learning rate schedule
- Gradient norms
- Sample audio quality over time

### Evaluation Results (After Task 10)

**Metrics computation:**
```bash
python scripts/evaluate_model.py \
    --checkpoint <path> \
    --test-set data/processed/test \
    --output results.json
```

**Extract:**
- WER, CER for in-speaker and zero-shot
- MCD, PESQ, STOI quality metrics
- Speaker similarity (SECS)
- Statistical significance tests

## Report Templates

### Conference Paper Template (4-8 pages)

**Structure:**
1. Abstract (150-200 words)
2. Introduction (1 page)
   - Romanian TTS landscape
   - Chatterbox architecture
   - Contributions
3. Related Work (0.5 pages)
   - TTS models comparison
   - Romanian language resources
4. Methodology (2 pages)
   - Model architecture ([source](#model-architecture))
   - Dataset ([source](#dataset-statistics))
   - Training procedure ([source](#training-configuration))
5. Evaluation (1 page)
   - Metrics ([source](#metrics-table))
   - Results (TBD after Task 10)
6. Discussion (1 page)
   - Findings analysis
   - Limitations
   - Future work
7. Conclusion (0.5 pages)
8. References

### Project Report Template (15-20 pages)

**Additional sections:**
- Background and Motivation (2 pages)
- Literature Review (3 pages)
- Design Decisions (2 pages) - [source](technical-decisions.md)
- Implementation Details (3 pages) - [source](data/data-preparation-notes.md)
- Results Analysis (3 pages)
- Appendices:
  - Dataset statistics
  - Hyperparameter tuning
  - Code snippets

### Presentation Template (15-20 slides)

**Slide Breakdown:**
1. Title + Motivation (2 slides)
2. Background: TTS + Romanian gap (2 slides)
3. Chatterbox architecture (2 slides)
4. SWARA dataset (2 slides) - [stats](#dataset-statistics)
5. Training approach (2 slides) - [config](#training-configuration)
6. Evaluation framework (2 slides) - [metrics](#evaluation-framework)
7. Results (3 slides) - TBD
8. Discussion + Future work (2 slides)
9. Conclusion (1 slide)

## Useful Markdown to LaTeX Conversions

### Tables

**Pandoc command:**
```bash
pandoc input.md -o output.tex
```

**Manual adjustment:**
- Replace `|` with `&`
- Add `\\` at row ends
- Wrap in `tabular` environment

### Citations

**From BibTeX:**
```latex
\cite{swara2023}
```

**Inline:**
```latex
We use the SWARA dataset~\cite{swara2023}, which contains 21,304 utterances...
```

## Contact for Clarifications

**Project Lead:** Adrian Stanea
**Email:** [To be added]
**Institution:** ACP3

**For documentation questions:**
- Check [main README](README.md) first
- Review relevant technical document
- Contact project lead if unclear

---

**Last Updated:** February 21, 2026
**Next Update:** After Task 7 completion (training results available)
