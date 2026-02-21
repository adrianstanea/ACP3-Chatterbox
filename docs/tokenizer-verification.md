# Tokenizer Verification for Romanian Character Coverage

## Task Overview

**Status**: Ready for execution (requires Docker container)
**Date**: 2026-02-21
**Phase**: 3 - Tokenizer & Preprocessing
**Task**: 5 - Verify and Extend Tokenizer for Romanian

## Objective

Verify that the Chatterbox grapheme tokenizer (2,454 tokens for 23 languages) includes all Romanian diacritical characters, and extend it if necessary.

## Romanian Character Requirements

### Diacritics
- Lowercase: ă, â, î, ș, ț
- Uppercase: Ă, Â, Î, Ș, Ț

### Common Grapheme Patterns
The tokenizer should also ideally include common Romanian phonetic patterns:
- Vowel combinations: ea, oa, ia, iu, ie
- Individual diacritics as tokens (likely in `[char]` bracket format)

## Implementation Steps

### Step 1: Start Docker Container

**Note**: Docker daemon must be running. Start it manually if needed:
```bash
sudo service docker start
# or
sudo systemctl start docker
```

Then start the container:
```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation

# Verify .env file exists and has correct paths
cat .env

# Start container
docker compose up -d chatterbox

# Verify container is running
docker compose ps
```

### Step 2: Install Dependencies and Run Setup

Enter the container and install dependencies:
```bash
# Enter container
docker compose exec chatterbox bash

# Inside container:
cd /workspace
pip install -r requirements.txt

# Navigate to fine-tuning kit
cd vendor/chatterbox-finetuning

# Run setup to download pretrained models
python setup.py
```

**Expected Output**:
- `setup.py` will download models from HuggingFace
- For non-Turbo mode (is_turbo=False): downloads `tokenizer.json`
  (actually `grapheme_mtl_merged_expanded_v1.json`)
- Files saved to `pretrained_models/` directory
- Expected vocab size: 2,454 tokens (standard Chatterbox)

### Step 3: Locate Tokenizer File

After `setup.py` completes, find the tokenizer:
```bash
# Inside container:
find /workspace/vendor/chatterbox-finetuning -name "tokenizer.json" -type f

# Expected location:
# /workspace/vendor/chatterbox-finetuning/pretrained_models/tokenizer.json
```

### Step 4: Run Verification Script

Execute the verification script with the tokenizer path:
```bash
# Inside container:
cd /workspace

# Run verification (adjust paths as needed)
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

**What the script does**:
1. Loads tokenizer vocabulary from JSON
2. Checks for all Romanian characters (ă, â, î, ș, ț + capitals)
3. Analyzes common grapheme patterns
4. If metadata.csv provided: analyzes character usage in dataset
5. Reports coverage and recommendations

### Step 5: Interpret Results

#### Scenario A: All Characters Present ✓
```
✓ All Romanian characters are covered by the tokenizer.
✓ No action needed - proceed with preprocessing.
```

**Action**: Document findings and proceed to Task 6 (Preprocessing)

#### Scenario B: Missing Characters ✗
```
⚠ WARNING: 10 Romanian characters are missing from the tokenizer!
Missing: ă, â, î, ș, ț, Ă, Â, Î, Ș, Ț
```

**Action**: Extend tokenizer (see Step 6)

### Step 6: Extend Tokenizer (If Needed)

If Romanian characters are missing, manually extend the tokenizer:

#### 6.1: Edit tokenizer.json

```bash
# Inside container:
cd /workspace/vendor/chatterbox-finetuning/pretrained_models

# Backup original
cp tokenizer.json tokenizer.json.backup

# Edit tokenizer.json
# Add missing characters to model.vocab section
```

**Example addition to `tokenizer.json`**:
```json
{
  "model": {
    "vocab": {
      "existing_tokens": "...",
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

**Important**: Assign sequential IDs starting from current vocab_size (2454)

#### 6.2: Update Configuration Files

Update `new_vocab_size` in both files:

**File 1**: `/workspace/vendor/chatterbox-finetuning/src/config.py`
```python
# Line 38 (approximately)
new_vocab_size: int = 52260 if is_turbo else 2464  # Changed from 2454 to 2464
```

**File 2**: `/workspace/vendor/chatterbox-finetuning/inference.py`
Look for `new_vocab_size` variable and update accordingly.

#### 6.3: Verify Extension

Re-run the verification script:
```bash
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

Should now report all characters present.

### Step 7: Document Findings

Exit container and create findings document:
```bash
# Exit container
exit

# Back on host, document results
```

Update this file with:
- Actual tokenizer vocabulary size found
- Romanian character coverage status
- Action taken (no change needed / tokenizer extended)
- New vocab_size if changed

### Step 8: Commit Changes

```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation

git add scripts/verify_tokenizer.py
git add docs/tokenizer-verification.md

# If tokenizer was extended:
git add vendor/chatterbox-finetuning/pretrained_models/tokenizer.json
git add vendor/chatterbox-finetuning/src/config.py
git add vendor/chatterbox-finetuning/inference.py

git commit -m "feat: verify tokenizer coverage for Romanian characters

- Add verification script for Romanian diacritics (ă, â, î, ș, ț)
- Script analyzes tokenizer vocabulary and metadata text
- Documents findings and recommendations
[Extended tokenizer to include missing Romanian characters]
[Updated vocab_size from 2454 to XXXX]"
```

## Verification Script Features

The `scripts/verify_tokenizer.py` script provides:

1. **Vocabulary Loading**: Loads tokenizer.json and extracts vocabulary
2. **Character Coverage**: Checks all 10 Romanian diacritics (5 lowercase + 5 uppercase)
3. **Grapheme Analysis**: Checks for common Romanian phonetic patterns
4. **Metadata Analysis**: Analyzes character usage in actual dataset text
5. **Recommendations**: Provides actionable steps if extension needed

### Usage Examples

```bash
# Basic verification (tokenizer only)
python scripts/verify_tokenizer.py tokenizer.json

# With metadata analysis (recommended)
python scripts/verify_tokenizer.py tokenizer.json metadata.csv

# Full path example (inside container)
python /workspace/scripts/verify_tokenizer.py \
    /workspace/vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    /workspace/data/processed/MyTTSDataset/metadata.csv
```

## Expected Results (To Be Filled After Execution)

### Tokenizer Information
- **Original Vocab Size**: TBD
- **Tokenizer Format**: JSON (grapheme_mtl_merged_expanded_v1.json)
- **Source**: HuggingFace - ResembleAI/chatterbox
- **Commit Hash**: TBD (document from setup.py download)

### Romanian Character Coverage
- **ă**: TBD
- **â**: TBD
- **î**: TBD
- **ș**: TBD
- **ț**: TBD
- **Ă**: TBD
- **Â**: TBD
- **Î**: TBD
- **Ș**: TBD
- **Ț**: TBD

### Dataset Character Usage
(To be filled from metadata analysis)
- Total characters analyzed: TBD
- Romanian diacritic occurrences: TBD
- Percentage of text with diacritics: TBD%

### Action Taken
- [ ] No action needed - all characters present
- [ ] Tokenizer extended - added X missing characters
- [ ] vocab_size updated from 2454 to XXXX

### Files Modified
(To be filled after execution)
- [ ] tokenizer.json (extended vocabulary)
- [ ] src/config.py (updated new_vocab_size)
- [ ] inference.py (updated new_vocab_size)

## Success Criteria

- ✓ Verification script created and executable
- ✓ Docker container running successfully
- ✓ setup.py executed (models downloaded)
- ✓ Tokenizer located and inspected
- ✓ Romanian character coverage verified
- ✓ Findings documented
- ✓ If needed: tokenizer extended and vocab_size updated
- ✓ Changes committed to git

## Next Steps

After successful verification:
1. If no extension needed: Proceed to Task 6 (Run Preprocessing Pipeline)
2. If extended: Re-run verification, then proceed to Task 6

## Notes

- The tokenizer uses grapheme-based representation (likely with `[char]` brackets)
- Original Chatterbox supports 23 languages with 2,454 tokens
- Romanian was likely not in the original 23 languages
- Extension is expected and normal for new language support
- Keep original tokenizer as backup before modification

## References

- Chatterbox Repository: https://github.com/ResembleAI/chatterbox
- Fine-tuning Kit: Community implementation
- Tokenizer Format: HuggingFace tokenizers library
- Romanian Language: 5 diacritics (ă, â, î, ș, ț) + capitals
