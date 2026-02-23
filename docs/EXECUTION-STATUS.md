# Execution Status: Task 5 - Tokenizer Verification

## Status: READY FOR MANUAL EXECUTION

**Created**: 2026-02-21
**Task**: Verify and Extend Tokenizer for Romanian
**Phase**: 3 - Tokenizer & Preprocessing

## What Has Been Completed

### 1. Verification Script Created ✓
- **File**: `scripts/verify_tokenizer.py`
- **Status**: Created and committed (a4c21da)
- **Features**:
  - Loads tokenizer vocabulary from JSON
  - Checks all 10 Romanian diacritics (ă, â, î, ș, ț + capitals)
  - Analyzes grapheme patterns
  - Processes metadata for character usage
  - Provides actionable recommendations

### 2. Documentation Created ✓
- **tokenizer-verification.md**: Full verification procedure and findings template
- **TOKENIZER-SETUP-GUIDE.md**: Quick reference for execution
- **scripts/README.md**: Script usage documentation

### 3. Git Commit ✓
- Commit: `a4c21da`
- Message: "feat: add tokenizer verification script for Romanian character coverage"
- All documentation and scripts committed

## What Needs Manual Execution

### Prerequisites
1. **Start Docker daemon** (requires sudo)
   ```bash
   sudo service docker start
   ```

2. **Verify .env configuration**
   ```bash
   cat .env  # Check SWARA_PATH and OUTPUT_PATH
   ```

### Execution Steps

#### Step 1: Start Docker Container
```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation
docker compose up -d chatterbox
docker compose ps  # Verify running
```

#### Step 2: Install Dependencies and Download Models
```bash
# Enter container
docker compose exec chatterbox bash

# Inside container:
cd /workspace
pip install -r requirements.txt

# Download pretrained models (creates tokenizer.json)
cd vendor/chatterbox-finetuning
python setup.py
```

**Expected Output**:
```
--- Chatterbox Pretrained Model Setup ---
Directory found: pretrained_models
Mode: CHATTERBOX-TTS (Checking 5 files)
Downloading: ve.safetensors...
Downloading: t3_cfg.safetensors...
Downloading: s3gen.safetensors...
Downloading: conds.pt...
Downloading: tokenizer.json...
INSTALLATION COMPLETE (CHATTERBOX-TTS MOD)
All models are set up in 'pretrained_models/' folder.
Note: 'grapheme_mtl_merged_expanded_v1.json' was saved as 'tokenizer.json' for the new vocabulary.
```

#### Step 3: Verify Tokenizer Location
```bash
# Still inside container:
ls -la /workspace/vendor/chatterbox-finetuning/pretrained_models/
```

Expected files:
- `tokenizer.json` (the grapheme vocabulary)
- `ve.safetensors`
- `t3_cfg.safetensors`
- `s3gen.safetensors`
- `conds.pt`

#### Step 4: Run Verification Script
```bash
# Inside container:
cd /workspace

python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

**Expected Output Scenarios**:

**Scenario A - All Characters Present**:
```
================================================================================
Romanian Character Coverage Analysis
================================================================================
✓ 'ă' found as '[ă]' (ID: 1234)
✓ 'â' found as '[â]' (ID: 1235)
...
Coverage Summary:
  Found: 10/10 characters
  Missing: 0 characters
================================================================================
RECOMMENDATIONS
================================================================================
✓ All Romanian characters are covered by the tokenizer.
✓ No action needed - proceed with preprocessing.
```

**Scenario B - Missing Characters**:
```
================================================================================
Romanian Character Coverage Analysis
================================================================================
✗ 'ă' NOT FOUND in vocabulary
✗ 'â' NOT FOUND in vocabulary
...
Coverage Summary:
  Found: 0/10 characters
  Missing: 10 characters
  Missing characters: ă, â, î, ș, ț, Ă, Â, Î, Ș, Ț
================================================================================
RECOMMENDATIONS
================================================================================
⚠ WARNING: 10 Romanian characters are missing from the tokenizer!
ACTION REQUIRED: Extend the tokenizer vocabulary
```

#### Step 5a: If Extension Needed

**5a.1: Backup and Edit Tokenizer**
```bash
# Inside container:
cd /workspace/vendor/chatterbox-finetuning/pretrained_models

# Backup
cp tokenizer.json tokenizer.json.backup

# Edit tokenizer
nano tokenizer.json
```

Add missing characters to `model.vocab` section:
```json
{
  "model": {
    "vocab": {
      ... existing entries ...
      "[ă]": 2454,
      "[â]": 2455,
      "[î]": 2456,
      "[ș]": 2457,
      "[ț]": 2458,
      "[Ă]": 2459,
      "[Â]": 2460,
      "[Î]": 2461,
      "[Ș]": 2462,
      "[Ț]": 2463
    }
  }
}
```

**5a.2: Update Configuration**
```bash
# Inside container:
nano /workspace/vendor/chatterbox-finetuning/src/config.py
```

Update line ~38:
```python
new_vocab_size: int = 52260 if is_turbo else 2464  # Changed from 2454 to 2464
```

**5a.3: Re-verify**
```bash
# Inside container:
cd /workspace
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

Should now show all characters found.

#### Step 5b: If No Extension Needed

Simply proceed to next task (preprocessing).

#### Step 6: Document Results

Exit container and update documentation:
```bash
# Exit container
exit

# Update docs/tokenizer-verification.md with actual results
nano docs/tokenizer-verification.md
```

Fill in the "Expected Results" section with:
- Actual vocab size
- Character coverage status
- Character usage statistics
- Whether tokenizer was extended

#### Step 7: Commit Changes (if tokenizer was extended)

```bash
git add vendor/chatterbox-finetuning/pretrained_models/tokenizer.json
git add vendor/chatterbox-finetuning/src/config.py
git add docs/tokenizer-verification.md

git commit -m "feat: extend tokenizer for Romanian character coverage

- Add Romanian diacritics to tokenizer vocabulary
- Update vocab_size from 2454 to 2464 (10 new characters)
- Characters added: ă, â, î, ș, ț, Ă, Â, Î, Ș, Ț
- Verification successful - all characters now covered

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

## Current Blocker

**Docker daemon not running**
- Requires sudo privileges to start
- Cannot be automated in this session
- User must manually start Docker

**Command to unblock**:
```bash
sudo service docker start
```

## Files Ready for Use

All files are ready and committed. Once Docker is running:

1. **Verification Script**: `scripts/verify_tokenizer.py` (executable)
2. **Setup Guide**: `docs/TOKENIZER-SETUP-GUIDE.md` (quick reference)
3. **Full Documentation**: `docs/tokenizer-verification.md` (detailed procedures)
4. **Scripts README**: `scripts/README.md` (usage information)

## Next Task After Completion

After tokenizer verification is complete:

**Task 6: Run Preprocessing Pipeline**
- Convert metadata and audio to preprocessed format
- Tokenize text using verified tokenizer
- Extract audio features
- Prepare training batches

## Testing Checklist

Before moving to Task 6, verify:

- [ ] Docker container running
- [ ] setup.py executed successfully
- [ ] tokenizer.json exists at `vendor/chatterbox-finetuning/pretrained_models/`
- [ ] All pretrained models downloaded (5 files)
- [ ] Verification script executed
- [ ] Romanian character coverage confirmed (all 10 characters)
- [ ] If extended: vocab_size updated in config.py
- [ ] If extended: changes committed to git
- [ ] Documentation updated with actual results

## Estimated Time

- Docker setup: 5 minutes
- Model download (setup.py): 10-15 minutes (depends on internet speed)
- Verification: 1 minute
- Extension (if needed): 10-15 minutes
- Documentation: 5 minutes

**Total**: 30-40 minutes

## Notes

- The tokenizer is grapheme-based (character-level)
- Original vocabulary: 2,454 tokens for 23 languages
- Romanian likely not in original 23 languages
- Extension is expected and normal for new language support
- Tokenizer format uses brackets: `[ă]`, `[â]`, etc.

## References

- HuggingFace Model: ResembleAI/chatterbox
- Tokenizer File: `grapheme_mtl_merged_expanded_v1.json`
- Config Mode: Standard (is_turbo=False)
