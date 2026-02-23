# Romanian Chatterbox Documentation

**Project:** Adapting Chatterbox TTS to Romanian
**Institution:** ACP3 Academic Project
**Status:** Active Development (February 2026)

## Overview

This directory contains comprehensive documentation for the Romanian Chatterbox adaptation project. The documentation is organized by topic and designed to support both development and academic publication.

## Quick Start

**New to this project?** Start here:
1. [Project Overview](project-overview.md) - High-level goals and architecture
2. [Design Document](plans/2026-02-21-chatterbox-romanian-design.md) - Detailed design decisions
3. [Environment Setup](setup/environment-setup.md) - Getting started with Docker and dependencies

**Ready to work?** Task-specific guides:
- Data preparation: [SWARA Analysis](data/swara-analysis-report.md) + [Data Preparation Notes](data/data-preparation-notes.md)
- Training: [Fine-tuning Kit Analysis](vendor/chatterbox-finetuning-analysis.md)
- Technical questions: [Technical Decisions Log](technical-decisions.md)

## Documentation Structure

```
docs/
├── README.md                          # This file - documentation index
├── project-overview.md                # High-level project description
├── technical-decisions.md             # Log of all major technical choices
│
├── data/                              # Dataset documentation
│   ├── swara-analysis-report.md       # SWARA dataset statistics and analysis
│   └── data-preparation-notes.md      # LJSpeech conversion technical notes
│
├── setup/                             # Environment and infrastructure
│   └── environment-setup.md           # Docker, PyTorch, DGX setup
│
├── plans/                             # Implementation plans and designs
│   ├── 2026-02-21-chatterbox-romanian-design.md     # Complete design doc
│   ├── 2026-02-21-chatterbox-romanian-plan.md       # Implementation plan
│   └── chatterbox_ro_implementation_plan.md         # Alternative plan format
│
├── vendor/                            # Third-party code documentation
│   └── chatterbox-finetuning-analysis.md  # Analysis of community fine-tuning kit
│
├── codex/                             # AI assistant knowledge base
│   ├── README.md                      # Codex documentation index
│   ├── chatterbox_ro_report.md        # Research report
│   ├── chatterbox_ro_executive_summary.md
│   ├── devcontainers_requirements.md  # Devcontainer design
│   ├── tts_researcher.md              # TTS research notes
│   └── tts_researcher_details.md
│
└── gemini/                            # Additional research notes
    └── chatterbox_romanian_report.md  # Gemini-generated research
```

## Core Documentation

### Essential Reading

**For Understanding the Project:**

| Document | Purpose | Audience |
|----------|---------|----------|
| [Project Overview](project-overview.md) | High-level goals, architecture, timeline | Everyone |
| [Design Document](plans/2026-02-21-chatterbox-romanian-design.md) | Complete technical design | Developers, researchers |
| [Technical Decisions](technical-decisions.md) | Rationale for key choices | Developers, reviewers |

**For Working with the Code:**

| Document | Purpose | Audience |
|----------|---------|----------|
| [Environment Setup](setup/environment-setup.md) | Docker, GPU, dependencies | Developers |
| [SWARA Analysis](data/swara-analysis-report.md) | Dataset statistics | Data engineers, researchers |
| [Data Preparation](data/data-preparation-notes.md) | Conversion pipeline details | Data engineers |
| [Fine-tuning Kit Analysis](vendor/chatterbox-finetuning-analysis.md) | Training infrastructure | ML engineers |

## Documentation by Phase

### Phase 1: Infrastructure Setup (Completed)

**Completed Tasks:**
- ✓ Task 1: Docker Compose and devcontainer - [Environment Setup](setup/environment-setup.md)
- ✓ Task 2: Fine-tuning kit integration - [Vendor Analysis](vendor/chatterbox-finetuning-analysis.md)
- ✓ Task 3: SWARA analysis - [SWARA Report](data/swara-analysis-report.md)
- ✓ Task 4: Data conversion - [Preparation Notes](data/data-preparation-notes.md)

**Key Documents:**
- Environment decisions: [Technical Decisions #2](technical-decisions.md#decision-2-pytorch-2601-python-312)
- Vendor code integration: [Technical Decisions #3](technical-decisions.md#decision-3-git-submodule-for-vendor-code)

### Phase 2: Training Preparation (In Progress)

**Completed Tasks:**
- ✓ Task 5: Tokenizer verification - [Task 5 Summary](TASK-5-SUMMARY.md)

**Current Tasks:**
- ⧗ Task 6: Preprocessing pipeline

**Relevant Documents:**
- Tokenizer verification: [Tokenizer Verification Guide](tokenizer-verification.md)
- Tokenizer setup: [Quick Setup Guide](TOKENIZER-SETUP-GUIDE.md)
- Execution status: [Execution Status](EXECUTION-STATUS.md)
- Tokenizer strategy: [Design Doc §2](plans/2026-02-21-chatterbox-romanian-design.md#tokenizer)
- Preprocessing details: [Fine-tuning Kit Analysis §3](vendor/chatterbox-finetuning-analysis.md#3-preprocessing-pipeline)

### Phase 3: Training and Evaluation (Pending)

**Upcoming Tasks:**
- ☐ Task 7: Training run
- ☐ Task 8: Model integration
- ☐ Task 9: Evaluation samples
- ☐ Task 10: Metrics computation

**Relevant Documents:**
- Training configuration: [Design Doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration)
- Evaluation framework: [Design Doc §5](plans/2026-02-21-chatterbox-romanian-design.md#5-evaluation-framework)

### Phase 4: Release (Pending)

**Upcoming Tasks:**
- ☐ Task 11: HuggingFace packaging

**Relevant Documents:**
- Release plan: [Design Doc §6](plans/2026-02-21-chatterbox-romanian-design.md#6-inference-integration--release)

## Documentation by Topic

### Dataset

**SWARA 1.0 (Romanian Speech Corpus):**
- [Complete Analysis Report](data/swara-analysis-report.md)
  - 21,304 utterances, 18 speakers, 21.67 hours
  - Speaker statistics and distribution
  - Duration analysis and quality assessment
  - Romanian diacritic verification
  - Character inventory for tokenizer

- [Data Preparation Notes](data/data-preparation-notes.md)
  - LJSpeech format conversion
  - Path mapping solutions
  - Text normalization (Chatterbox punc_norm)
  - Validation split strategy
  - Quality assurance procedures

### Architecture

**Model Selection:**
- [Technical Decision #1](technical-decisions.md#decision-1-chatterbox-multilingual-not-turbo)
  - Why Multilingual (not Turbo)
  - 500M Llama 3 backbone
  - Grapheme tokenizer advantages
  - Documented Turbo mode failures

**Components:**
- [Design Doc §1](plans/2026-02-21-chatterbox-romanian-design.md#1-architecture--model-choice)
  - T3 transformer (trainable)
  - Voice encoder (frozen)
  - S3Gen vocoder (frozen)
  - Tokenizer strategy

### Environment

**Container Infrastructure:**
- [Environment Setup Guide](setup/environment-setup.md)
  - PyTorch 26.01 (Python 3.12, CUDA 13.1)
  - Docker Compose configuration
  - VS Code devcontainer integration
  - DGX compatibility notes

**Technical Decisions:**
- [PyTorch Version Decision](technical-decisions.md#decision-2-pytorch-2601-python-312)
  - Python 3.12 compatibility research
  - CUDA 13.1 selection
  - Dependency verification

### Training

**Fine-Tuning Infrastructure:**
- [Community Kit Analysis](vendor/chatterbox-finetuning-analysis.md)
  - Repository structure
  - Configuration system
  - Preprocessing pipeline
  - Training loop details
  - Known limitations and patches

**Training Configuration:**
- [Design Doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration)
  - Hyperparameters for V100
  - Batch size and gradient accumulation
  - FP16 precision (V100 constraint)
  - Multi-GPU considerations

### Evaluation

**Metrics and Methodology:**
- [Design Doc §5](plans/2026-02-21-chatterbox-romanian-design.md#5-evaluation-framework)
  - WER, CER (intelligibility)
  - MCD, PESQ, STOI (quality)
  - SECS (speaker similarity)
  - MOS (subjective quality)

**Evaluation Splits:**
- In-speaker: Validation set (known speakers)
- Zero-shot: BAS and SGS holdout (unseen speakers)
- Diacritic test: Romanian minimal pairs

## Research Notes

### TTS Landscape Research

**Background Research:**
- [TTS Researcher Notes](codex/tts_researcher.md) - General TTS landscape
- [TTS Researcher Details](codex/tts_researcher_details.md) - Deep dive
- [Chatterbox Research Report](codex/chatterbox_ro_report.md) - Initial research
- [Executive Summary](codex/chatterbox_ro_executive_summary.md) - Quick reference

**Gemini Research:**
- [Gemini Report](gemini/chatterbox_romanian_report.md) - Alternative research perspective

### Implementation Plans

**Multiple Planning Iterations:**
- [Latest Design](plans/2026-02-21-chatterbox-romanian-design.md) - **CANONICAL VERSION**
- [Implementation Plan](plans/2026-02-21-chatterbox-romanian-plan.md) - Step-by-step tasks
- [Alternative Plan](plans/chatterbox_ro_implementation_plan.md) - Earlier iteration

## Statistics Summary (for Papers)

### Dataset Statistics

**SWARA 1.0:**
```
Total utterances:     21,304
Total duration:       21.67 hours
Speakers:             18 (16 training, 2 holdout)
Mean duration:        3.66 ± 2.1 seconds
Sample rate:          22,050 Hz
Diacritics:           Complete (ă, â, î, ș, ț)
Train/val/test:       81% / 9% / 11%
```

**See:** [SWARA Analysis Report](data/swara-analysis-report.md)

### Model Configuration

**Chatterbox Multilingual:**
```
Architecture:         Llama 3 (500M parameters)
Tokenizer:            Grapheme (2,454 tokens)
Trainable:            T3 transformer only
Frozen:               Voice encoder + S3Gen vocoder
Context length:       256 tokens (text), 850 tokens (speech)
```

**See:** [Project Overview](project-overview.md#architecture-overview)

### Training Setup

**Hyperparameters:**
```
Precision:            FP16 (V100 constraint)
Batch size:           4 per GPU
Gradient accum:       8 steps
Effective batch:      32
Learning rate:        1e-5
Epochs:               120 (or 50k-150k steps)
```

**See:** [Design Doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration)

## For Academic Publication

### Paper Sections Mapping

**Introduction:**
- Context: [TTS Researcher](codex/tts_researcher.md)
- Romanian TTS gap: [Project Overview §Goal](project-overview.md#project-goal)

**Related Work:**
- TTS architectures: [TTS Researcher Details](codex/tts_researcher_details.md)
- Chatterbox background: [Design Doc §1](plans/2026-02-21-chatterbox-romanian-design.md#1-architecture--model-choice)

**Methodology:**
- Model selection: [Technical Decision #1](technical-decisions.md#decision-1-chatterbox-multilingual-not-turbo)
- Data preparation: [Data Preparation Notes](data/data-preparation-notes.md)
- Training procedure: [Design Doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration)

**Dataset:**
- Statistics: [SWARA Analysis Report](data/swara-analysis-report.md)
- Quality assessment: [SWARA Report §Data Quality](data/swara-analysis-report.md#data-quality-assessment)

**Evaluation:**
- Metrics: [Design Doc §5](plans/2026-02-21-chatterbox-romanian-design.md#5-evaluation-framework)
- Splits: [Technical Decision #4](technical-decisions.md#decision-4-validation-split-strategy)

**Results:**
- TBD after training (Task 7-10)

**Discussion:**
- Design decisions: [Technical Decisions](technical-decisions.md)
- Limitations: [Design Doc §7](plans/2026-02-21-chatterbox-romanian-design.md#7-risks--mitigations)

### Reproducibility Checklist

**Code and Data:**
- [ ] Dataset citation: [SWARA Report §License](data/swara-analysis-report.md#license-and-attribution)
- [ ] Environment specification: [Environment Setup](setup/environment-setup.md)
- [ ] Hyperparameters: [Design Doc §4](plans/2026-02-21-chatterbox-romanian-design.md#4-training-configuration)
- [ ] Random seeds: [Data Preparation Notes §Best Practices](data/data-preparation-notes.md#best-practices)

**Transparency:**
- [ ] Technical decisions logged: [Technical Decisions](technical-decisions.md)
- [ ] Alternatives considered: Each decision includes alternatives
- [ ] Limitations documented: [Design Doc §8](plans/2026-02-21-chatterbox-romanian-design.md#8-non-goals-explicit-exclusions)

## Version Control

### Documentation Standards

**File Naming:**
- Descriptive names (no abbreviations)
- Include dates for time-sensitive docs (e.g., `2026-02-21-*.md`)
- Use hyphens for multi-word names

**Content Standards:**
- Markdown format for version control
- Tables for structured data
- Code blocks with language tags
- Internal links for cross-references

**Update Triggers:**
- After major technical decisions
- After completing tasks
- When discovering new information
- Before major milestones

### Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| Project Overview | Current | 2026-02-21 |
| Design Document | Current | 2026-02-21 |
| Technical Decisions | Current | 2026-02-21 |
| Environment Setup | Current | 2026-02-21 |
| SWARA Analysis | Current | 2026-02-21 |
| Data Preparation | Current | 2026-02-21 |
| Vendor Analysis | Current | 2026-02-21 |
| Task 5 Summary | Current | 2026-02-21 |
| Tokenizer Verification | Current | 2026-02-21 |
| Execution Status | Current | 2026-02-21 |

## Getting Help

### For Code Issues

1. Check relevant technical doc:
   - Environment: [Setup Guide](setup/environment-setup.md)
   - Data: [Preparation Notes](data/data-preparation-notes.md)
   - Training: [Vendor Analysis](vendor/chatterbox-finetuning-analysis.md)

2. Review technical decisions: [Decisions Log](technical-decisions.md)

3. Check troubleshooting sections in relevant docs

### For Design Questions

1. Check design rationale: [Design Doc](plans/2026-02-21-chatterbox-romanian-design.md)

2. Review decision log: [Technical Decisions](technical-decisions.md)

3. Consult research notes: [Codex](codex/) directory

### For Dataset Questions

1. Statistics: [SWARA Analysis Report](data/swara-analysis-report.md)

2. Preprocessing: [Data Preparation Notes](data/data-preparation-notes.md)

3. Quality issues: Check QA sections in both docs

## Contributing to Documentation

### When to Update

**Always document:**
- New technical decisions (add to [Technical Decisions](technical-decisions.md))
- Task completion (update relevant guide)
- Issues discovered (add to troubleshooting)
- Significant findings (update analysis reports)

**Consider documenting:**
- Workarounds for bugs
- Performance optimizations
- Useful commands or scripts
- Lessons learned

### How to Update

1. **Identify relevant document** (use this README to find it)
2. **Check current status** (avoid duplicate information)
3. **Follow existing format** (match style and structure)
4. **Add cross-references** (link to related docs)
5. **Update this README** (if adding new document)

### Documentation Quality Checklist

- [ ] Clear and concise writing
- [ ] Code examples use correct syntax
- [ ] Tables formatted properly
- [ ] Internal links work
- [ ] External links are stable (no temporary URLs)
- [ ] Date-stamped if time-sensitive
- [ ] Cross-references added where relevant

## License and Attribution

**Project Documentation:** Same license as code (to be determined)

**Dataset Attribution:** SWARA 1.0 requires citation (see [SWARA Report](data/swara-analysis-report.md#license-and-attribution))

**Vendor Code:** Community fine-tuning kit (forked), original license preserved

## Changelog

| Date | Change | Documents Affected |
|------|--------|-------------------|
| 2026-02-21 | Initial documentation structure | All |
| 2026-02-21 | Add comprehensive docs for Phase 1 | 6 new documents |
| 2026-02-21 | Create documentation index | This README |
| 2026-02-21 | Task 5 implementation complete | TASK-5-SUMMARY.md, tokenizer-verification.md, TOKENIZER-SETUP-GUIDE.md, EXECUTION-STATUS.md |

---

**Last Updated:** February 21, 2026
**Maintained By:** Adrian Stanea
**Status:** Active Development
