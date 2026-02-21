# Technical Decisions Log

**Project:** Romanian Chatterbox Adaptation
**Date Range:** February 2026
**Status:** Active Development

## Purpose

This document records all significant technical decisions made during the Romanian Chatterbox adaptation project, including the rationale, alternatives considered, and implications of each choice.

---

## Decision 1: Chatterbox Multilingual (Not Turbo)

**Date:** February 20, 2026
**Status:** FINAL
**Category:** Model Architecture

### Context

Chatterbox offers two model variants:
1. **Multilingual (Standard):** 500M parameter Llama 3 backbone, 23 languages
2. **Turbo:** 350M parameter GPT-2 backbone, English-focused

### Decision

**Chosen:** Chatterbox Multilingual (Standard Mode)

### Rationale

**Technical Evidence:**
1. **Documented Failures in Turbo Mode:**
   - GitHub Issue #6: Hallucinated audio artifacts when fine-tuning on new languages
   - GitHub Issue #12: Posterior collapse during training
   - Community reports of instability with non-English languages

2. **Multilingual Advantages:**
   - Already handles 23 languages with cross-lingual transfer
   - Grapheme-based tokenizer (character-level) more suitable for Romanian
   - Proven stability with diverse language families
   - Similar structure to Romanian (Latin script) in existing languages

3. **Tokenizer Compatibility:**
   - Multilingual: 2,454 grapheme tokens covering Latin, Cyrillic, Turkish
   - Turbo: ~52,000 BPE tokens optimized for English
   - Romanian diacritics (ă, â, î, ș, ț) likely already in grapheme set
   - BPE would require significant vocabulary extension

### Alternatives Considered

**Turbo Mode:**
- **Pros:** Potentially faster inference (GPT-2 architecture)
- **Cons:** Documented instability, BPE tokenizer issues, English bias
- **Risk:** High failure probability

**Training from Scratch:**
- **Pros:** Complete control, no bias from pretraining
- **Cons:** Requires 100+ hours of data, months of training, no cross-lingual transfer
- **Risk:** Impractical for academic project timeline

### Implementation Impact

**Training Configuration:**
```python
# In src/config.py
is_turbo = False  # Use Standard/Multilingual mode
new_vocab_size = 2454  # Default grapheme vocabulary
```

**Preprocessing:**
- Use grapheme tokenizer (`tokenizer.json`)
- Verify Romanian diacritics present
- Extend vocabulary only if diacritics missing

**Model Components:**
- T3 Transformer: Trainable (Llama-based)
- Voice Encoder: Frozen
- S3Gen Vocoder: Frozen

### References

- Chatterbox GitHub Issues: #6, #12
- Community fine-tuning kit README: Mode comparison section
- Fine-tuning kit analysis: `docs/vendor/chatterbox-finetuning-analysis.md`

---

## Decision 2: PyTorch 26.01 (Python 3.12)

**Date:** February 21, 2026
**Status:** FINAL
**Category:** Environment/Infrastructure

### Context

Need to select base Docker image with PyTorch, CUDA, and Python. Two main candidates:
1. **PyTorch 26.01:** CUDA 13.1, PyTorch 2.10, Python 3.12
2. **PyTorch 24.12:** CUDA 12.6, PyTorch 2.6, Python 3.10

### Decision

**Chosen:** `nvcr.io/nvidia/pytorch:26.01-py3`

### Rationale

**Initial Concern (Feb 20):**
- Python 3.12 is relatively new
- Concern about Chatterbox compatibility

**Research Conducted (Feb 20-21):**

| Package | Version | Python 3.12 Support | Verification Method |
|---------|---------|---------------------|---------------------|
| torch | 2.6.0+ | ✓ Official support | PyPI metadata |
| torchaudio | 2.6.0+ | ✓ Matches torch | PyPI metadata |
| transformers | Latest | ✓ Python 3.8+ | HuggingFace docs |
| peft | 0.17.1 | ✓ Modern package | PyPI metadata |
| chatterbox-tts | 0.1.2 | ✓ No restrictions | setup.py inspection |
| soundfile | 0.13.1 | ✓ Pure Python | Tested |
| librosa | 0.11.0 | ✓ NumPy 2.0 compat | PyPI metadata |

**Findings:**
- All dependencies confirmed compatible with Python 3.12
- No known issues or incompatibilities
- Chatterbox itself is a small wrapper (no Python version constraints)

**Additional Benefits:**
1. **Latest CUDA:** 13.1 vs 12.6 (better V100 support)
2. **Latest PyTorch:** 2.10 vs 2.6 (performance improvements)
3. **Modern Python:** 3.12 has better performance and typing
4. **Future-proof:** Longer support timeline

### Alternatives Considered

**PyTorch 24.12 (Python 3.10):**
- **Pros:** Older Python may have fewer edge cases
- **Cons:** Older CUDA, older PyTorch, shorter support
- **Risk:** Low (but no advantage)

**PyTorch 23.XX (Python 3.8):**
- **Pros:** Very stable, well-tested
- **Cons:** Much older stack, no modern features
- **Risk:** Technical debt from day one

### Implementation Impact

**Docker Configuration:**
```yaml
services:
  chatterbox:
    image: nvcr.io/nvidia/pytorch:26.01-py3
```

**Compatibility Verification:**
```bash
# Inside container
python --version  # Python 3.12.x
python -c "import torch; print(torch.__version__)"  # 2.10.x
python -c "import chatterbox"  # Should import successfully
```

**DGX Compatibility:**
- V100 GPUs: Fully supported by CUDA 13.1
- FP16 precision: Works (V100 doesn't support BF16)
- No issues expected

### Risk Mitigation

**Rollback Plan:**
If Python 3.12 causes unexpected issues:
1. Switch to `nvcr.io/nvidia/pytorch:24.12-py3` (Python 3.10)
2. Update `docker-compose.yml`
3. Rebuild container
4. Verify dependencies install correctly

**Estimated effort:** 30 minutes

### References

- NVIDIA PyTorch Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Python 3.12 compatibility research: Internal investigation (Feb 20-21, 2026)
- Environment setup docs: `docs/setup/environment-setup.md`

---

## Decision 3: Git Submodule for Vendor Code

**Date:** February 21, 2026
**Status:** FINAL
**Category:** Code Organization

### Context

Need to integrate the community fine-tuning kit (https://github.com/gokhaneraslan/chatterbox-finetuning) into our project. Multiple integration strategies possible.

### Decision

**Chosen:** Git submodule with forked repository

**Implementation:**
```bash
# Fork upstream repo to our GitHub
# Add as submodule
git submodule add https://github.com/adrianstanea/chatterbox-finetuning vendor/chatterbox-finetuning

# Pin to specific commit
cd vendor/chatterbox-finetuning
git checkout 18ffb2d  # Voice Conditioning Dropout commit
```

### Rationale

**Benefits:**

1. **Version Control:**
   - Track exact commit of vendor code
   - Easy to update: `git submodule update --remote`
   - Clear separation of our code vs. vendor code

2. **Modification Capability:**
   - Can apply patches to our fork
   - Track changes separately from main project
   - Easy to contribute back upstream

3. **Reproducibility:**
   - `.gitmodules` records exact repository and commit
   - Other researchers can clone with `--recurse-submodules`
   - No ambiguity about vendor code version

4. **Clean Repository:**
   - Vendor code not in main repo history
   - Smaller main repository
   - Clear licensing boundaries

### Alternatives Considered

**Option A: Copy Code Directly**
- **Pros:** Simple, no submodule complexity
- **Cons:** No version tracking, hard to update, unclear provenance
- **Risk:** Reproducibility issues, licensing concerns

**Option B: Python Package (pip install)**
- **Pros:** Clean dependency management
- **Cons:** Fine-tuning kit is not a package, would require repackaging
- **Risk:** Effort not justified for research project

**Option C: Fork Only (No Submodule)**
- **Pros:** Full control, easy to modify
- **Cons:** Have to manually sync, no clear link to upstream
- **Risk:** Drift from upstream updates

### Implementation Impact

**Repository Structure:**
```
.gitmodules               # Submodule configuration
vendor/
  chatterbox-finetuning/  # Git submodule (our fork)
    src/
    train.py
    inference.py
    ...
```

**Cloning:**
```bash
# Option 1: Clone with submodules
git clone --recurse-submodules <repo-url>

# Option 2: Initialize after clone
git clone <repo-url>
git submodule update --init --recursive
```

**Updating Vendor Code:**
```bash
cd vendor/chatterbox-finetuning
git fetch origin
git checkout <new-commit>
cd ../..
git add vendor/chatterbox-finetuning
git commit -m "Update vendor code to <new-commit>"
```

**Applying Patches:**
```bash
cd vendor/chatterbox-finetuning
# Make changes
git add .
git commit -m "Patch: Add multi-GPU support"
git push origin main  # Push to our fork
cd ../..
git add vendor/chatterbox-finetuning
git commit -m "Apply multi-GPU patch to vendor code"
```

### Submodule Configuration

**File: `.gitmodules`**
```
[submodule "vendor/chatterbox-finetuning"]
    path = vendor/chatterbox-finetuning
    url = https://github.com/adrianstanea/chatterbox-finetuning
    branch = main
```

**Pinned Commit:** `18ffb2d` (Voice Conditioning Dropout feature)

### Future Considerations

**If Upstream Updates:**
1. Review upstream changes
2. Test in our fork
3. Merge to our fork if beneficial
4. Update submodule reference in main project

**If We Modify Extensively:**
- Consider vendoring directly (copy code, remove submodule)
- Only if we significantly diverge from upstream
- Document modifications clearly

### References

- Git Submodules: https://git-scm.com/book/en/v2/Git-Tools-Submodules
- Our fork: https://github.com/adrianstanea/chatterbox-finetuning
- Upstream: https://github.com/gokhaneraslan/chatterbox-finetuning

---

## Decision 4: Validation Split Strategy

**Date:** February 21, 2026
**Status:** FINAL
**Category:** Data Preparation

### Context

SWARA dataset has 21,304 utterances from 18 speakers. Need to split into training, validation, and test sets for robust evaluation.

### Decision

**Chosen:** Three-way split with per-speaker stratification

**Split Configuration:**
1. **Holdout Speakers (Test):** BAS and SGS - 100% reserved for zero-shot evaluation
2. **Training Speakers (16 remaining):** ~~90/10 per-speaker stratified train/validation split~~ → **5 samples per speaker for validation (approved deviation)**

**Decision Made:**
- ~~10% validation split (90/10 ratio)~~ → **5 samples per speaker (80 samples total, ~0.5%)**
- Approved deviation from original spec

### Rationale

**Zero-Shot Holdout:**

**Why BAS and SGS?**
- Representative speakers (not outliers)
- Sufficient data for evaluation
- ~11% of total data (2 of 18 speakers)
- Tests model's ability to adapt to unseen Romanian speakers

**Benefits:**
- Evaluates zero-shot voice cloning (critical for Chatterbox)
- Tests generalization to new speakers
- Simulates real-world usage (new voices)

**Per-Speaker Stratification:**

**Why Not Random Split?**
- Random: Risk of speaker leakage (same speaker in train and val)
- Random: Can't evaluate in-speaker vs out-of-speaker performance
- Stratified: Ensures all speakers represented in both train and val

**Benefits:**
- Prevents speaker-specific overfitting
- Enables in-speaker evaluation (known speakers)
- Balanced validation across all training speakers

**5 Samples Per Speaker (Updated Decision):**

**Rationale:**
- Small validation set sufficient for listening checks during training
- Maximizes training data (18,740 samples vs ~16,866 with 10% split)
- Academic research focus - comprehensive evaluation happens on held-out test set (BAS, SGS)
- Faster validation loops during training

**Impact:**
- Training set: 18,740 samples (99.5% of non-holdout data)
- Validation set: 80 samples (5 per speaker, stratified)
- Test set: 2,484 samples (BAS and SGS, zero-shot evaluation)

### Alternatives Considered

**Option A: Single Speaker Holdout**
- **Pros:** More training data
- **Cons:** Insufficient test data, less robust evaluation
- **Risk:** Unreliable zero-shot metrics

**Option B: Random 80/10/10 Split**
- **Pros:** Simple implementation
- **Cons:** Speaker leakage, can't separate in-speaker vs zero-shot
- **Risk:** Inflated validation metrics (same speakers in train and val)

**Option C: 70/15/15 Split**
- **Pros:** Larger validation and test sets
- **Cons:** Less training data (critical for neural TTS)
- **Risk:** Underfitting due to insufficient training data

**Option D: K-Fold Cross-Validation**
- **Pros:** Maximum data usage, robust metrics
- **Cons:** 5x training time, impractical for large models
- **Risk:** Too expensive for academic project

### Implementation

**Conversion Script Logic:**
```python
# Identify speakers
HOLDOUT_SPEAKERS = {'bas', 'sgs'}

# For each utterance:
speaker_id = extract_speaker_id(filename)

if speaker_id in HOLDOUT_SPEAKERS:
    # Write to test set (not used in training)
    test_files.append(...)
else:
    # 90/10 per-speaker stratified split
    train, val = stratified_split_per_speaker(utterances, ratio=0.9)
```

**Validation:**
```python
# Verify no speaker leakage
train_speakers = set(extract_speaker_ids(train_set))
val_speakers = set(extract_speaker_ids(val_set))
test_speakers = set(extract_speaker_ids(test_set))

assert train_speakers == val_speakers  # Same speakers in train and val
assert train_speakers.isdisjoint(test_speakers)  # No overlap with test
```

### Evaluation Framework

**1. In-Speaker Evaluation (Validation Set):**
- Synthesize validation utterances using same speaker reference
- Measures quality on known speakers
- Metrics: WER, CER, MCD, PESQ

**2. Zero-Shot Evaluation (Test Set - BAS/SGS):**
- Synthesize test utterances using BAS/SGS reference audio
- Measures zero-shot voice cloning quality
- Metrics: WER, CER, SECS (speaker similarity)

**3. Cross-Evaluation:**
- Compare in-speaker vs zero-shot metrics
- Quantify generalization gap
- Assess speaker adaptation quality

### Expected Outcomes

**Success Criteria:**
- In-speaker WER < 5% (comparable to ground truth ASR)
- Zero-shot WER < 10% (acceptable for unseen speakers)
- SECS > 0.8 (high speaker similarity for zero-shot)

**If Zero-Shot Underperforms:**
- Indicates need for more speaker diversity in training
- Or need for speaker adaptation fine-tuning
- Still valuable research finding

### References

- VCTK dataset split strategy (similar multi-speaker approach)
- LJSpeech evaluation (single-speaker baseline)
- Zero-shot TTS literature (speaker holdout standard practice)

---

## Decision 5: Text Normalization Approach

**Date:** February 21, 2026
**Status:** FINAL
**Category:** Data Preparation

### Context

SWARA text transcripts need normalization before training. Need to decide between custom normalization vs. using Chatterbox official normalization.

### Decision

**Chosen:** Use Chatterbox `punc_norm` function for consistency

### Rationale

**Official Normalization Advantages:**

1. **Consistency:**
   - Same normalization used in Chatterbox pretraining
   - Matches model's expected input format
   - Reduces distribution shift

2. **Tested:**
   - Already validated on 23 languages
   - Handles edge cases (quotes, dashes, etc.)
   - Less likely to introduce errors

3. **Simplicity:**
   - One function call
   - No custom regex needed
   - Less code to maintain

**SWARA Preprocessing Status:**

Based on analysis:
- Numbers already expanded to words (e.g., "123" → "o sută douăzeci și trei")
- Diacritics preserved correctly
- Minimal preprocessing already done

**What `punc_norm` Provides:**
- Consistent quotation marks
- Normalized dashes/hyphens
- Whitespace normalization
- Preserved diacritics

### Implementation

**Conversion Script (`convert_swara_to_ljspeech.py`):**
```python
from chatterbox.utils import punc_norm

# For each utterance:
raw_text = row[1]  # Original SWARA text
normalized_text = punc_norm(raw_text)

# Write to metadata.csv
writer.writerow([filename, raw_text, normalized_text])
```

**LJSpeech Format:**
```
filename|raw_text|normalized_text
bas_rnd1_001|Original text.|normalized text
```

### Alternatives Considered

**Option A: Custom Normalization**
- **Pros:** Full control, Romanian-specific rules
- **Cons:** More complex, untested, potential errors
- **Risk:** Distribution shift from Chatterbox pretraining

**Option B: No Normalization**
- **Pros:** Preserves original text exactly
- **Cons:** Inconsistent punctuation, potential training issues
- **Risk:** Model may struggle with edge cases

**Option C: Phonemic Normalization (espeak-ng)**
- **Pros:** Explicit pronunciation
- **Cons:** Chatterbox uses graphemes, not phonemes
- **Risk:** Incompatible with model architecture

### Additional Normalization

**Abbreviations (If Found):**

Romanian abbreviations to expand:
- str. → strada
- nr. → numărul
- dr. → doctorul
- etc. → etcetera

**Implementation:**
```python
ROMANIAN_ABBREVIATIONS = {
    'str.': 'strada',
    'nr.': 'numărul',
    'dr.': 'doctorul',
    # Add more as needed
}

def expand_abbreviations(text: str) -> str:
    for abbr, expansion in ROMANIAN_ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)
    return text

# Apply before punc_norm
text = expand_abbreviations(raw_text)
text = punc_norm(text)
```

### Validation

**Verification Steps:**
1. Check Romanian diacritics preserved
2. Verify numbers are words (not digits)
3. Ensure consistent punctuation
4. Validate no character loss

**Quality Checks:**
```python
# Before normalization
raw_diacritics = count_diacritics(raw_text)

# After normalization
norm_diacritics = count_diacritics(normalized_text)

assert raw_diacritics == norm_diacritics, "Diacritics lost in normalization!"
```

### References

- Chatterbox `punc_norm`: https://github.com/ResembleAI/chatterbox
- LJSpeech normalization: https://keithito.com/LJ-Speech-Dataset/
- SWARA preprocessing: Analysis in `scripts/analyze_swara.py`

---

## Decision 6: Multi-GPU Training Strategy (Deferred)

**Date:** February 21, 2026
**Status:** DEFERRED
**Category:** Training

### Context

Fine-tuning kit has no official multi-GPU support. DGX has multiple V100 GPUs available. Need to decide on multi-GPU strategy.

### Decision

**Chosen:** Start with single-GPU, implement multi-GPU if needed

### Rationale

**Single-GPU Viability:**
- V100 32GB can fit the model with gradient checkpointing
- Batch size 4, gradient accumulation 8 = effective batch 32
- Training time acceptable for academic project (~20-30 hours)
- Lower complexity, fewer potential bugs

**Multi-GPU Complexity:**
- Requires modifying training code (no official support)
- DDP (DistributedDataParallel) integration needed
- Batch size scaling and learning rate tuning
- Checkpoint handling for rank 0 only
- Estimated effort: 50-100 lines of code + testing

**Decision:**
- Phase 1: Single-GPU training (minimal risk)
- Phase 2: If training too slow, add multi-GPU

### Implementation Plan (If Needed)

**Approach A: HuggingFace Accelerate**
```bash
accelerate config  # Interactive configuration
accelerate launch vendor/chatterbox-finetuning/train.py
```

**Approach B: Torchrun**
```bash
torchrun --nproc_per_node=4 vendor/chatterbox-finetuning/train.py
```

**Approach C: Manual DDP**
```python
# In train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Adjust batch size
batch_size = config.batch_size // world_size
```

### Risk Assessment

**Single-GPU Risks:**
- Training time may be longer than desired
- May not fully utilize DGX resources

**Multi-GPU Risks:**
- Implementation bugs (gradient synchronization)
- Batch size scaling issues (convergence)
- Checkpoint corruption (if not handling rank properly)

### Decision Point

**Re-evaluate after:**
- First 10 epochs of single-GPU training
- If training time > 48 hours for full run
- If DGX has idle GPUs (resource efficiency)

### References

- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- HuggingFace Accelerate: https://huggingface.co/docs/accelerate/
- Fine-tuning kit analysis: Multi-GPU section

---

## Summary Table

| # | Decision | Date | Status | Impact |
|---|----------|------|--------|--------|
| 1 | Chatterbox Multilingual (not Turbo) | 2026-02-20 | Final | Model architecture |
| 2 | PyTorch 26.01 (Python 3.12) | 2026-02-21 | Final | Environment |
| 3 | Git Submodule for vendor code | 2026-02-21 | Final | Code organization |
| 4 | Per-speaker stratified split + holdout | 2026-02-21 | Final | Data preparation |
| 5 | Chatterbox punc_norm for text | 2026-02-21 | Final | Text normalization |
| 6 | Single-GPU (multi-GPU deferred) | 2026-02-21 | Deferred | Training strategy |

---

## Decision Making Process

**Criteria for Technical Decisions:**

1. **Reproducibility:** Can another researcher replicate this?
2. **Robustness:** Is this approach stable and well-tested?
3. **Academic Quality:** Is this rigorous and defensible?
4. **Practicality:** Can we implement this within project constraints?
5. **Documentation:** Can we clearly explain this decision?

**Documentation Standard:**

Each decision includes:
- Context (why we needed to decide)
- Rationale (why we chose this)
- Alternatives (what else we considered)
- Implementation (how we'll do it)
- References (where to learn more)

---

## Future Decisions

**Pending Decisions:**

1. **Tokenizer Extension:**
   - Status: Awaiting verification of Romanian diacritics
   - Decision point: After tokenizer inspection (Task 5)

2. **Hyperparameter Tuning:**
   - Status: Using defaults from fine-tuning kit
   - Decision point: After initial training run

3. **Evaluation Metrics:**
   - Status: Planned (WER, CER, MCD, PESQ, MOS)
   - Decision point: During evaluation phase (Task 9-10)

4. **HuggingFace Release Format:**
   - Status: Not yet decided
   - Decision point: After successful training (Task 11)

---

## Lessons Learned

**Key Insights:**

1. **Research Before Deciding:** Python 3.12 compatibility research prevented unnecessary conservative choice
2. **Community Wisdom:** GitHub issues revealed Turbo mode problems
3. **Simplicity First:** Single-GPU approach reduces complexity
4. **Official Tools:** Using Chatterbox's own normalization ensures consistency

**Best Practices:**

- Document decisions immediately (while rationale is fresh)
- List alternatives explicitly (shows thoroughness)
- Include implementation details (actionable guidance)
- Reference sources (reproducibility)

---

## Document Maintenance

**Update Triggers:**

- Major technical decision made
- Decision status changes (deferred → final)
- New information changes rationale
- Implementation reveals issues

**Ownership:**

- Primary: Project lead (Adrian Stanea)
- Reviews: Before major milestones
- Format: Markdown for readability and version control
