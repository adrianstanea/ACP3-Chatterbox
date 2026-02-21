# Chatterbox Romanian Adaptation -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune Chatterbox Multilingual (500M Llama 3) for Romanian using the SWARA dataset, producing an academically rigorous evaluation and a HuggingFace-releasable model.

**Architecture:** Fine-tune only the T3 transformer on Romanian text-speech pairs from SWARA (21h, 17 speakers). VE and S3Gen remain frozen. The model extends the existing 23-language Multilingual variant to include Romanian as language #24.

**Tech Stack:** PyTorch 2.10, HuggingFace Trainer, chatterbox-tts, chatterbox-finetuning community kit, espeak-ng, NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:26.01-py3), Docker Compose, V100 32GB GPUs.

**Design Doc:** `docs/plans/2026-02-21-chatterbox-romanian-design.md`

---

## Phase 1: Environment & Infrastructure

### Task 1: Create Docker Compose and Devcontainer Configuration

**Files:**
- Create: `docker-compose.yml`
- Create: `.devcontainer/devcontainer.json`
- Create: `.env.example`

**Step 1: Create `.env.example`**

```env
# Paths - adjust for your environment
SWARA_PATH=/data/swara
OUTPUT_PATH=/data/output

# Training
NUM_GPUS=1
```

**Step 2: Create `docker-compose.yml`**

```yaml
services:
  chatterbox:
    image: nvcr.io/nvidia/pytorch:26.01-py3
    volumes:
      - .:/workspace
      - ${SWARA_PATH}:/data/swara:ro
      - ${OUTPUT_PATH}:/data/output
    working_dir: /workspace
    env_file: .env
    shm_size: '16gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: bash -c "pip install -r requirements.txt && bash"
```

**Step 3: Create `.devcontainer/devcontainer.json`**

```json
{
  "name": "Chatterbox Romanian",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "chatterbox",
  "workspaceFolder": "/workspace",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter"
      ]
    }
  }
}
```

**Step 4: Create `requirements.txt`**

```
chatterbox-tts
espeak-phonemize
librosa
soundfile
tensorboard
jiwer
pesq
pystoi
```

**Step 5: Verify compose file is syntactically valid**

Run: `docker compose config`
Expected: Parsed YAML output, no errors.

**Step 6: Commit**

```bash
git add docker-compose.yml .devcontainer/ .env.example requirements.txt
git commit -m "feat: add Docker Compose and devcontainer for DGX training"
```

---

### Task 2: Clone and Integrate Community Fine-Tuning Kit

**Files:**
- Create: `vendor/chatterbox-finetuning/` (git clone)
- Modify: `.gitignore` (add vendor exclusion or submodule)

**Step 1: Clone the community fine-tuning kit**

Run: `git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git vendor/chatterbox-finetuning`

**Step 2: Inspect the repo structure**

Run: `ls -la vendor/chatterbox-finetuning/`

Understand: `setup.py`, `train.py`, `inference.py`, `src/config.py`, `src/` directory structure.

**Step 3: Read `src/config.py` to understand all configurable parameters**

Key things to note:
- `is_turbo` flag (must be `False` for Multilingual/Standard)
- `new_vocab_size` value
- `dataset_dir` path
- `ljspeech` flag (must be `True`)
- `batch_size`, `learning_rate`, `num_epochs`, precision settings
- `max_text_len`, `max_speech_len`

**Step 4: Read `setup.py` to understand model download and tokenizer setup**

Understand: What models are downloaded, where tokenizer is stored, how `new_vocab_size` is set.

**Step 5: Read the preprocessing scripts**

Read: `preprocess_ljspeech.py` (or equivalent) to understand `.pt` artifact generation.

**Step 6: Read `train.py` to understand training loop**

Key things to note:
- HuggingFace Trainer usage
- Multi-GPU support (or lack thereof)
- Checkpoint saving
- Loss function

**Step 7: Read `inference.py` to understand how fine-tuned model is loaded**

Understand: How `t3_finetuned.safetensors` is loaded, tokenizer usage, audio generation pipeline.

**Step 8: Document findings**

Create a brief summary of what works, what needs patching, and any gaps (especially multi-GPU, config flexibility).

**Step 9: Commit**

```bash
git add vendor/chatterbox-finetuning
git commit -m "feat: import community fine-tuning kit as vendor dependency"
```

> **Note:** Alternatively, add as a git submodule: `git submodule add https://github.com/gokhaneraslan/chatterbox-finetuning.git vendor/chatterbox-finetuning`

---

## Phase 2: Data Preparation

### Task 3: SWARA Dataset Analysis

**Files:**
- Create: `scripts/analyze_swara.py`

**Step 1: Write analysis script**

```python
"""Analyze SWARA dataset: duration distribution, speaker stats, text stats."""
import csv
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import soundfile as sf


def analyze(metadata_path: str):
    speakers = defaultdict(lambda: {"count": 0, "total_dur": 0.0, "durations": []})
    all_durations = []
    missing_files = []
    all_chars = Counter()

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            wav_path, text, speaker_id = row[0], row[1], row[2]

            # Character analysis
            for ch in text:
                all_chars[ch] += 1

            if not os.path.exists(wav_path):
                missing_files.append(wav_path)
                continue

            info = sf.info(wav_path)
            dur = info.duration
            sr = info.samplerate

            speakers[speaker_id]["count"] += 1
            speakers[speaker_id]["total_dur"] += dur
            speakers[speaker_id]["durations"].append(dur)
            all_durations.append(dur)

    # Print results
    print(f"=== SWARA Dataset Analysis ===")
    print(f"Total utterances: {len(all_durations)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Total duration: {sum(all_durations)/3600:.2f} hours")
    print(f"Sample rate: {sr} Hz")
    print(f"\nDuration stats:")
    print(f"  Min: {min(all_durations):.2f}s")
    print(f"  Max: {max(all_durations):.2f}s")
    print(f"  Mean: {sum(all_durations)/len(all_durations):.2f}s")
    print(f"  <0.5s: {sum(1 for d in all_durations if d < 0.5)}")
    print(f"  >30s: {sum(1 for d in all_durations if d > 30)}")
    print(f"  3-10s: {sum(1 for d in all_durations if 3 <= d <= 10)}")

    print(f"\nSpeaker stats:")
    for spk, info in sorted(speakers.items()):
        print(f"  {spk}: {info['count']} utterances, {info['total_dur']/60:.1f} min")

    # Romanian diacritics check
    ro_diacritics = set("ăâîșțĂÂÎȘȚ")
    found = ro_diacritics & set(all_chars.keys())
    missing = ro_diacritics - found
    print(f"\nRomanian diacritics found: {sorted(found)}")
    if missing:
        print(f"Romanian diacritics MISSING: {sorted(missing)}")
    else:
        print("All Romanian diacritics present in transcripts.")

    # Top characters
    print(f"\nTop 30 characters:")
    for ch, count in all_chars.most_common(30):
        print(f"  '{ch}': {count}")


if __name__ == "__main__":
    analyze(sys.argv[1])
```

**Step 2: Run analysis on SWARA metadata**

Run: `python scripts/analyze_swara.py metadata_SWARA1.0_text.csv`

Expected: Statistics about duration distribution, speaker counts, sample rate, diacritic presence.

**Step 3: Review results and note any issues**

- Are there clips <0.5s or >30s?
- Are all speakers well-represented?
- What is the actual sample rate?
- Are all diacritics present in transcripts?

**Step 4: Commit**

```bash
git add scripts/analyze_swara.py
git commit -m "feat: add SWARA dataset analysis script"
```

---

### Task 4: SWARA to LJSpeech Conversion Script

**Files:**
- Create: `scripts/convert_swara_to_ljspeech.py`

**Step 1: Write conversion script**

```python
"""Convert SWARA dataset to LJSpeech format for Chatterbox fine-tuning.

Output structure:
  data/processed/MyTTSDataset/
    metadata.csv        # filename|raw_text|normalized_text
    wavs/               # symlinks or copies of WAV files
    splits/
      train.csv         # subset of metadata.csv
      val.csv
      test_holdout.csv  # BAS and SGS speakers only

Original dataset is never modified.
"""
import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


HOLDOUT_SPEAKERS = {"1"}  # speaker_id for BAS and SGS -- update after analysis
# Placeholder: actual speaker IDs from metadata need to be mapped


def normalize_text(text: str) -> str:
    """Romanian text normalization preserving diacritics."""
    # Replace common abbreviations
    replacements = {
        " str. ": " strada ",
        " nr. ": " numărul ",
        " bl. ": " blocul ",
        " sc. ": " scara ",
        " et. ": " etajul ",
        " ap. ": " apartamentul ",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized


def convert(metadata_path: str, swara_base_dir: str, output_dir: str,
            val_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)
    output = Path(output_dir)
    wavs_dir = output / "wavs"
    splits_dir = output / "splits"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Parse metadata
    entries_by_speaker = defaultdict(list)
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            wav_path, text, speaker_id = row[0].strip(), row[1].strip(), row[2].strip()
            filename = Path(wav_path).stem
            entries_by_speaker[speaker_id].append({
                "filename": filename,
                "wav_path": wav_path,
                "raw_text": text,
                "normalized_text": normalize_text(text),
                "speaker_id": speaker_id,
            })

    # Split
    train_entries = []
    val_entries = []
    test_entries = []

    for speaker_id, entries in entries_by_speaker.items():
        # Identify holdout speakers by prefix in filename
        sample_filename = entries[0]["filename"]
        speaker_prefix = sample_filename.split("_")[0].upper()

        if speaker_prefix in {"BAS", "SGS"}:
            test_entries.extend(entries)
            continue

        random.shuffle(entries)
        split_idx = max(1, int(len(entries) * val_ratio))
        val_entries.extend(entries[:split_idx])
        train_entries.extend(entries[split_idx:])

    # Create symlinks for audio files
    for entry in train_entries + val_entries + test_entries:
        src = Path(entry["wav_path"])
        dst = wavs_dir / f"{entry['filename']}.wav"
        if not dst.exists():
            if src.exists():
                os.symlink(src.resolve(), dst)
            else:
                print(f"WARNING: Missing audio file: {src}")

    # Write metadata files
    def write_csv(entries, path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            for e in entries:
                f.write(f"{e['filename']}|{e['raw_text']}|{e['normalized_text']}\n")

    all_trainval = train_entries + val_entries
    write_csv(all_trainval, output / "metadata.csv")
    write_csv(train_entries, splits_dir / "train.csv")
    write_csv(val_entries, splits_dir / "val.csv")
    write_csv(test_entries, splits_dir / "test_holdout.csv")

    print(f"Train: {len(train_entries)} utterances")
    print(f"Val: {len(val_entries)} utterances")
    print(f"Test (holdout BAS+SGS): {len(test_entries)} utterances")
    print(f"Total in metadata.csv: {len(all_trainval)} (train+val for fine-tuning kit)")
    print(f"Output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Path to SWARA metadata CSV")
    parser.add_argument("--output", default="data/processed/MyTTSDataset")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    convert(args.metadata, os.path.dirname(args.metadata), args.output,
            val_ratio=args.val_ratio)
```

**Step 2: Run conversion**

Run: `python scripts/convert_swara_to_ljspeech.py metadata_SWARA1.0_text.csv --output data/processed/MyTTSDataset`

Expected: `data/processed/MyTTSDataset/` with `metadata.csv`, `wavs/`, and `splits/`.

**Step 3: Validate output**

- Check that `metadata.csv` has the correct format (`filename|raw_text|normalized_text`)
- Verify symlinks point to real files
- Confirm BAS and SGS are only in `test_holdout.csv`
- Confirm diacritics are preserved in normalized text

**Step 4: Commit**

```bash
git add scripts/convert_swara_to_ljspeech.py
git commit -m "feat: add SWARA to LJSpeech conversion with speaker-stratified splits"
```

---

## Phase 3: Tokenizer & Preprocessing

### Task 5: Verify and Extend Tokenizer for Romanian

**Files:**
- Create: `scripts/verify_tokenizer.py`

**Step 1: Write tokenizer verification script**

```python
"""Verify Chatterbox tokenizer covers all Romanian characters in SWARA."""
import json
import sys
from pathlib import Path


def verify(tokenizer_path: str, metadata_path: str):
    # Load tokenizer
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    vocab = set(tokenizer.keys()) if isinstance(tokenizer, dict) else set()
    # Handle different tokenizer formats
    if "model" in tokenizer and "vocab" in tokenizer["model"]:
        vocab = set(tokenizer["model"]["vocab"].keys())
    elif "added_tokens" in tokenizer:
        vocab = {t["content"] for t in tokenizer["added_tokens"]}

    print(f"Tokenizer vocab size: {len(vocab)}")

    # Collect all unique characters from metadata
    all_chars = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                text = parts[2]  # normalized_text
            elif len(parts) >= 2:
                text = parts[1]  # raw_text
            else:
                continue
            all_chars.update(text)

    # Check coverage
    ro_diacritics = set("ăâîșțĂÂÎȘȚ")
    print(f"\nRomanian diacritics in dataset: {ro_diacritics & all_chars}")
    print(f"All unique characters in dataset: {len(all_chars)}")

    # This script provides the character inventory.
    # Actual tokenizer verification depends on the tokenizer format
    # used by the community kit (inspect after running setup.py).
    print(f"\nCharacter inventory for tokenizer extension (if needed):")
    for ch in sorted(all_chars):
        print(f"  U+{ord(ch):04X} '{ch}'")


if __name__ == "__main__":
    verify(sys.argv[1], sys.argv[2])
```

**Step 2: Run `setup.py` from the community kit to download models and create tokenizer**

Run (inside container):
```bash
cd vendor/chatterbox-finetuning
python setup.py
```

This downloads pretrained models and creates the tokenizer.

**Step 3: Inspect the generated tokenizer**

Find where `tokenizer.json` was saved (likely `pretrained_models/` or similar).
Run: `python scripts/verify_tokenizer.py <path_to_tokenizer.json> data/processed/MyTTSDataset/metadata.csv`

**Step 4: If Romanian diacritics are missing, extend the tokenizer**

Modify the tokenizer JSON to add missing characters. Update `new_vocab_size` in:
- `vendor/chatterbox-finetuning/src/config.py`
- `vendor/chatterbox-finetuning/inference.py`

**Step 5: Verify tokenization produces no unknowns**

Test by tokenizing a sample of Romanian sentences and checking for UNK tokens.

**Step 6: Commit**

```bash
git add scripts/verify_tokenizer.py
git commit -m "feat: add tokenizer verification script for Romanian character coverage"
```

---

### Task 6: Run Preprocessing Pipeline

**Step 1: Configure the community kit for SWARA**

Modify `vendor/chatterbox-finetuning/src/config.py`:
- `is_turbo = False`
- `ljspeech = True`
- `dataset_dir = "/data/output/MyTTSDataset"` (or the path to converted SWARA)
- `new_vocab_size = <exact count from tokenizer>`
- `fp16 = True` (V100)
- `bf16 = False`

**Step 2: Run preprocessing**

Run (inside container):
```bash
cd vendor/chatterbox-finetuning
python preprocess_ljspeech.py
```

Expected: `.pt` files generated in a `preprocessed/` directory, one per utterance, each containing `text_tokens`, `speech_tokens`, `speaker_emb`, `prompt_tokens`.

**Step 3: Validate preprocessing output**

```python
import torch
import os

pt_dir = "preprocessed/"  # adjust path
files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
print(f"Total .pt files: {len(files)}")

# Spot-check 5 files
for f in files[:5]:
    data = torch.load(os.path.join(pt_dir, f))
    print(f"\n{f}:")
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val)}")
```

**Step 4: Commit configuration changes**

```bash
git add -p vendor/chatterbox-finetuning/src/config.py
git commit -m "feat: configure fine-tuning kit for Romanian SWARA dataset"
```

---

## Phase 4: Training

### Task 7: Training Run

**Step 1: Verify configuration one final time**

Check `src/config.py`:
- `is_turbo = False`
- `new_vocab_size` matches tokenizer
- `batch_size`, `gradient_accumulation_steps` appropriate for V100
- `fp16 = True`, `bf16 = False`
- `gradient_checkpointing = True`

**Step 2: Start training (single GPU)**

Run (inside container):
```bash
cd vendor/chatterbox-finetuning
python train.py
```

**Step 3: Monitor training**

- Watch loss curve via TensorBoard: `tensorboard --logdir chatterbox_output/`
- Check for NaN losses
- At each checkpoint (every 500 steps), listen to a generated sample

**Step 4: Generate sample audio at checkpoints**

Run (periodically):
```bash
python inference.py --text "Bună ziua, mă numesc Maria." --ref_audio /data/swara/<some_speaker_ref>.wav
```

Listen to output and assess quality progression.

**Step 5: If multi-GPU is needed, attempt torchrun**

Run:
```bash
torchrun --nproc_per_node=N vendor/chatterbox-finetuning/train.py
```

If this doesn't work with the existing `train.py`, wrap with `accelerate launch` or patch `train.py` to support DDP.

**Step 6: When training converges, save final checkpoint**

The final checkpoint is `chatterbox_output/t3_finetuned.safetensors`.

**Step 7: Commit training configuration and any patches**

```bash
git add -p
git commit -m "feat: complete training run configuration and patches"
```

---

## Phase 5: Inference Integration

### Task 8: Integrate Fine-Tuned Model with Chatterbox Inference

**Files:**
- Create: `scripts/inference_romanian.py`

**Step 1: Write Romanian inference script**

```python
"""Generate Romanian speech using fine-tuned Chatterbox model."""
import argparse
from chatterbox.tts import ChatterboxTTS

def synthesize(text: str, ref_audio: str, checkpoint: str, output: str):
    # Load model with fine-tuned T3 weights
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # Load fine-tuned T3 checkpoint
    # (exact loading mechanism depends on how the community kit saves weights)
    model.load_t3_checkpoint(checkpoint)

    # Generate
    wav = model.generate(text=text, audio_prompt_path=ref_audio)

    # Save
    import torchaudio
    torchaudio.save(output, wav, model.sr)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--checkpoint", default="chatterbox_output/t3_finetuned.safetensors")
    parser.add_argument("--output", default="output.wav")
    args = parser.parse_args()
    synthesize(args.text, args.ref_audio, args.checkpoint, args.output)
```

> **Note:** The exact model loading API will depend on inspecting both `chatterbox-tts` and the community kit's `inference.py`. This script is a template to be adapted.

**Step 2: Test inference with several Romanian sentences**

Test sentences covering different phonetic challenges:
```
"Bună ziua, mă numesc Adrian."
"Studenții din anul trei au examene în sesiunea de iarnă."
"Căzăturile sunt la ordinea zilei."
"Capitala și nord-estul au cunoscut cele mai mari ritmuri de creștere."
```

**Step 3: Test with held-out speakers (BAS, SGS)**

Use reference audio clips from BAS and SGS to test zero-shot cloning on unseen speakers.

**Step 4: Commit**

```bash
git add scripts/inference_romanian.py
git commit -m "feat: add Romanian inference script with fine-tuned model loading"
```

---

## Phase 6: Evaluation

### Task 9: Generate Evaluation Samples

**Files:**
- Create: `scripts/generate_eval_samples.py`

**Step 1: Write batch generation script**

```python
"""Generate evaluation samples for all test utterances."""
import argparse
import csv
import os
from pathlib import Path


def generate_eval_samples(
    metadata_csv: str,
    ref_audio_dir: str,
    checkpoint: str,
    output_dir: str,
):
    """Generate synthesized audio for each row in metadata CSV.

    Output structure:
      output_dir/
        generated/
          <filename>.wav      # synthesized audio
        reference/
          <filename>.wav      # original audio (symlinked)
        texts/
          <filename>.txt      # input text
    """
    out = Path(output_dir)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    (out / "texts").mkdir(parents=True, exist_ok=True)

    # Load model once
    from chatterbox.tts import ChatterboxTTS
    import torchaudio

    model = ChatterboxTTS.from_pretrained(device="cuda")
    # Load fine-tuned checkpoint -- adapt based on actual API
    model.load_t3_checkpoint(checkpoint)

    with open(metadata_csv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            filename, raw_text, norm_text = parts[0], parts[1], parts[2]

            # Find a reference audio for this speaker
            # (use the first available clip from the same speaker prefix)
            speaker_prefix = filename.split("_")[0]
            ref_candidates = list(Path(ref_audio_dir).glob(f"{speaker_prefix}_*.wav"))
            if not ref_candidates:
                print(f"SKIP {filename}: no reference audio for {speaker_prefix}")
                continue
            ref_audio = str(ref_candidates[0])

            # Generate
            wav = model.generate(text=norm_text, audio_prompt_path=ref_audio)
            torchaudio.save(str(out / "generated" / f"{filename}.wav"), wav, model.sr)

            # Symlink reference
            orig_path = Path(ref_audio_dir) / f"{filename}.wav"
            if orig_path.exists():
                ref_link = out / "reference" / f"{filename}.wav"
                if not ref_link.exists():
                    os.symlink(orig_path.resolve(), ref_link)

            # Save text
            with open(out / "texts" / f"{filename}.txt", "w") as tf:
                tf.write(norm_text)

    print(f"Generated samples in: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Val or test CSV")
    parser.add_argument("--ref-audio-dir", required=True, help="Directory with WAV files")
    parser.add_argument("--checkpoint", default="chatterbox_output/t3_finetuned.safetensors")
    parser.add_argument("--output", required=True, help="Output directory for eval samples")
    args = parser.parse_args()
    generate_eval_samples(args.metadata, args.ref_audio_dir, args.checkpoint, args.output)
```

**Step 2: Generate in-speaker eval samples (val set)**

Run:
```bash
python scripts/generate_eval_samples.py \
  --metadata data/processed/MyTTSDataset/splits/val.csv \
  --ref-audio-dir data/processed/MyTTSDataset/wavs/ \
  --checkpoint chatterbox_output/t3_finetuned.safetensors \
  --output eval/in_speaker/
```

**Step 3: Generate out-of-speaker eval samples (holdout BAS+SGS)**

Run:
```bash
python scripts/generate_eval_samples.py \
  --metadata data/processed/MyTTSDataset/splits/test_holdout.csv \
  --ref-audio-dir data/processed/MyTTSDataset/wavs/ \
  --checkpoint chatterbox_output/t3_finetuned.safetensors \
  --output eval/out_of_speaker/
```

**Step 4: Commit**

```bash
git add scripts/generate_eval_samples.py
git commit -m "feat: add batch evaluation sample generation script"
```

---

### Task 10: Run Evaluation Metrics

**Step 1: Adapt existing evaluation infrastructure**

The existing evaluation script expects directories with `generated/` and `reference/` subdirectories. Verify the directory structure from Task 9 matches. Adapt if needed.

**Step 2: Run objective metrics on in-speaker samples**

Run the evaluation script on `eval/in_speaker/`. Expected metrics: WER, CER, MCD, PESQ, STOI, SI-SDR, RMSE, SECS.

**Step 3: Run objective metrics on out-of-speaker samples**

Run the evaluation script on `eval/out_of_speaker/`. Pay special attention to SECS (speaker embedding cosine similarity) for zero-shot cloning quality.

**Step 4: Create diacritic stress test set**

Create a small CSV with Romanian minimal pairs:
```
diacritic_test_01|Copiii dorm în paturi.|Copiii dorm în paturi.
diacritic_test_02|Bunica are pături frumoase.|Bunica are pături frumoase.
diacritic_test_03|Aceasta este o vară caldă.|Aceasta este o vară caldă.
diacritic_test_04|Vara este frumoasă.|Vara este frumoasă.
```

Generate and evaluate these separately.

**Step 5: Compile results into a summary table**

```markdown
| Metric | In-Speaker | Out-of-Speaker | Diacritic Test |
|--------|-----------|----------------|----------------|
| WER    | X.XX      | X.XX           | X.XX           |
| CER    | X.XX      | X.XX           | X.XX           |
| MCD    | X.XX      | X.XX           | X.XX           |
| PESQ   | X.XX      | X.XX           | X.XX           |
| STOI   | X.XX      | X.XX           | X.XX           |
| SECS   | X.XX      | X.XX           | N/A            |
```

**Step 6: Commit results**

```bash
git add eval/
git commit -m "feat: add evaluation results for Romanian Chatterbox"
```

---

## Phase 7: Release

### Task 11: Package for HuggingFace Release

**Files:**
- Create: `release/README.md` (model card)
- Create: `release/config.json`

**Step 1: Prepare model artifacts**

Collect:
- `t3_finetuned.safetensors` (fine-tuned T3 checkpoint)
- Extended `tokenizer.json` (if modified)
- Evaluation results summary

**Step 2: Write HuggingFace model card**

Include:
- Model description (Chatterbox Multilingual + Romanian)
- Training data (SWARA, CC BY-NC 4.0, with proper attribution)
- Training procedure (hyperparameters, hardware, duration)
- Evaluation results table
- Usage example
- Limitations and biases
- Citation

**Step 3: Test that the model loads and generates from the packaged artifacts**

Verify the released model card's usage example actually works end-to-end.

**Step 4: Upload to HuggingFace (when ready)**

Run:
```bash
huggingface-cli upload <username>/chatterbox-romanian release/
```

**Step 5: Commit release artifacts**

```bash
git add release/
git commit -m "feat: prepare HuggingFace release package for Romanian Chatterbox"
```

---

## Summary of Phases and Dependencies

```
Phase 1: Environment (Tasks 1-2) ─── no dependencies
    │
    v
Phase 2: Data Prep (Tasks 3-4) ──── depends on Task 2 (kit cloned)
    │
    v
Phase 3: Tokenizer & Preprocessing (Tasks 5-6) ── depends on Tasks 2, 4
    │
    v
Phase 4: Training (Task 7) ──── depends on Task 6
    │
    v
Phase 5: Inference (Task 8) ──── depends on Task 7
    │
    v
Phase 6: Evaluation (Tasks 9-10) ── depends on Task 8
    │
    v
Phase 7: Release (Task 11) ──── depends on Task 10
```
