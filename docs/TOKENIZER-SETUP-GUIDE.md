# Quick Setup Guide: Tokenizer Verification

## Prerequisites
- Docker daemon running
- .env file configured with correct paths

## Quick Commands

### 1. Start Docker (if needed)
```bash
sudo service docker start
```

### 2. Start Container
```bash
cd /home/astanea/git-repos/adrianstanea/ACP3-Chatterbox/.worktrees/romanian-adaptation
docker compose up -d chatterbox
```

### 3. Enter Container and Setup
```bash
docker compose exec chatterbox bash

# Inside container:
cd /workspace
pip install -r requirements.txt
cd vendor/chatterbox-finetuning
python setup.py
```

### 4. Verify Tokenizer
```bash
# Still inside container:
cd /workspace
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

### 5. If Extension Needed

Edit tokenizer (inside container):
```bash
cd /workspace/vendor/chatterbox-finetuning/pretrained_models
cp tokenizer.json tokenizer.json.backup
nano tokenizer.json  # Add missing Romanian characters
```

Update vocab_size:
```bash
cd /workspace/vendor/chatterbox-finetuning/src
nano config.py  # Update new_vocab_size on line ~38

cd /workspace/vendor/chatterbox-finetuning
nano inference.py  # Update new_vocab_size
```

Re-verify:
```bash
cd /workspace
python scripts/verify_tokenizer.py \
    vendor/chatterbox-finetuning/pretrained_models/tokenizer.json \
    data/processed/MyTTSDataset/metadata.csv
```

### 6. Exit and Commit
```bash
# Exit container
exit

# Commit changes
git add scripts/verify_tokenizer.py docs/
git add vendor/  # If tokenizer was modified
git commit -m "feat: verify and extend tokenizer for Romanian"
```

## Expected Paths

- **Tokenizer**: `/workspace/vendor/chatterbox-finetuning/pretrained_models/tokenizer.json`
- **Metadata**: `/workspace/data/processed/MyTTSDataset/metadata.csv`
- **Config**: `/workspace/vendor/chatterbox-finetuning/src/config.py`
- **Inference**: `/workspace/vendor/chatterbox-finetuning/inference.py`

## Romanian Characters to Check

Lowercase: ă, â, î, ș, ț
Uppercase: Ă, Â, Î, Ș, Ț

Likely format in tokenizer: `[ă]`, `[â]`, etc.

## Troubleshooting

### Container won't start
```bash
docker compose logs chatterbox
docker compose down
docker compose up -d chatterbox
```

### setup.py fails
- Check internet connection (downloads from HuggingFace)
- Verify requirements.txt installed correctly
- Check available disk space

### Tokenizer not found
```bash
# Inside container:
find /workspace -name "tokenizer.json" -o -name "*grapheme*.json"
```

### Verification script errors
```bash
# Inside container:
python -c "import json; print(json.load(open('vendor/chatterbox-finetuning/pretrained_models/tokenizer.json'))['model']['vocab'])" | head
```
