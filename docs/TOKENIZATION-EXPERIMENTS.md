# Tokenization Experiments for Romanian Chatterbox Adaptation

**Project:** Romanian Chatterbox TTS Adaptation (ACP3)
**Author:** Adrian Stanea
**Date:** February 22, 2026
**Status:** Experiment 2 in progress

## 1. Introduction

This document reports on two tokenization experiments conducted during the adaptation of the Chatterbox text-to-speech model to Romanian. The base model uses a character-level BPE tokenizer with 2,454 tokens (`grapheme_mtl_merged_expanded_v1.json`), covering 23 languages but lacking full Romanian diacritical character support.

The SWARA 1.0 corpus (21,304 utterances, 18 speakers, 21.67 hours) is used as the training dataset. Only the T3 transformer (text-to-speech token predictor) is fine-tuned; the Voice Encoder and S3Gen vocoder remain frozen.

### Romanian Characters and Tokenizer Coverage

Romanian uses 5 diacritical characters (10 including uppercase) not universally present in the base tokenizer:

| Character | IPA | Frequency in SWARA | In Base Vocab? | Token ID |
|-----------|-----|-------------------|----------------|----------|
| ă (a-breve) | /ə/ | 35,998 (2.71%) | ✅ Yes | 2413 |
| â (a-circumflex) | /ɨ/ | 5,909 (0.45%) | ✅ Yes | 395 |
| î (i-circumflex) | /ɨ/ | 9,309 (0.70%) | ✅ Yes | 407 |
| ș (s-comma-below) | /ʃ/ | 13,135 (0.99%) | ❌ No | — |
| ț (t-comma-below) | /ts/ | 12,762 (0.96%) | ❌ No | — |

The vowels (ă, â, î) are already present. Only the consonants ș and ț are missing, affecting approximately 2% of the corpus characters.

---

## 2. Experiment 1: Vocabulary Extension

### 2.1. Approach

Following the method recommended by the community fine-tuning toolkit ([gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)), we extended the tokenizer vocabulary by adding 5 missing characters:

| New Token | Unicode | Assigned ID |
|-----------|---------|-------------|
| ș | U+0219 | 2454 |
| ț | U+021B | 2455 |
| Ș | U+0218 | 2456 |
| Ț | U+021A | 2457 |
| Ă | U+0102 | 2458 |

**Vocabulary change:** 2,454 → 2,459 tokens (+5)

**Weight initialization:** The T3 model has two text-dependent layers — `text_emb` (embedding) and `text_head` (output projection). For token IDs 2454–2458, both layers were initialized with the **mean** of all existing token embeddings (mean initialization), as implemented in `resize_and_load_t3_weights()`.

**Implementation:**
- `scripts/extend_tokenizer.py` — generic tokenizer extension tool
- `scripts/vocab_extensions/romanian.json` — character definitions
- Tokenizer JSON patched at setup time via `.devcontainer/post-create.sh`

### 2.2. Results

After fine-tuning the T3 model with the extended vocabulary on the SWARA dataset:

- **Audio quality degraded severely** within a few hundred training steps
- Generated speech became **unintelligible** — sounding like babble or white noise
- The model appeared to **ignore the text input** and instead reconstruct a distorted version of the reference (speaker prompt) audio
- Loss decreased initially but did not correlate with improved speech quality

### 2.3. Analysis: Posterior Collapse

The failure is consistent with **posterior collapse**, a known phenomenon where the model ignores a weak conditioning signal (text) in favor of a stronger one (audio).

**Mechanism:**

The pretrained T3 model encodes text through an embedding layer where each of the original 2,454 tokens has a well-trained, discriminative embedding vector learned over the full pretraining dataset. When new tokens are added and initialized with the mean of all existing embeddings, these new vectors are:

1. **Non-discriminative:** They lie at the centroid of the embedding space, equidistant from all natural token clusters
2. **Low-magnitude signal:** Compared to the pretrained tokens' embeddings which encode rich, discriminative features
3. **Overwhelmed by the audio encoder:** The S3Gen vocoder provides reconstruction quality that far exceeds what the weakly-conditioned text path can offer

The model then learns to ignore text entirely and degenerates into an autoencoder that maps reference audio → generated audio.

### 2.4. Community Corroboration

This failure mode was independently reported by multiple users of the same fine-tuning toolkit:

| Issue | Reporter | Language | Symptom |
|-------|----------|----------|---------|
| [#6](https://github.com/gokhaneraslan/chatterbox-finetuning/issues/6) | 23rli | Turkish | "Gibberish audio", turbo-like noise |
| [#12](https://github.com/gokhaneraslan/chatterbox-finetuning/issues/12) | lramberg | Norwegian, German | Posterior collapse — model reproduces reference audio |
| [#14](https://github.com/gokhaneraslan/chatterbox-finetuning/issues/14) | MuhammedBuyukkinaci | Arabic | Poor pronunciation quality |

The toolkit maintainer confirmed on February 20, 2026: *"The new Weighted Text Embedding is too weak right now compared to the strong Audio Encoder"* (Issue #6). The maintainer also identified this as a structural problem requiring architectural changes, not a hyperparameter fix.

---

## 3. Experiment 2: Phoneme-Level Preprocessing

### 3.1. Motivation

Given that:
1. Vocabulary extension causes posterior collapse (confirmed experimentally and by community)
2. Only 2 characters (ș, ț) are truly missing from the vocabulary
3. Both missing characters have phonetically exact equivalents already in the vocabulary

We hypothesized that mapping Romanian special characters to phonetically equivalent existing tokens would avoid the weak-embedding problem entirely while preserving pronunciation accuracy.

### 3.2. Approach

Instead of extending the vocabulary, we preprocess all Romanian text before tokenization, replacing missing characters with sequences from the existing vocabulary that represent the same phoneme:

**Consonant mappings (characters not in vocab):**

| Romanian | IPA | Mapped To | Phonetic Basis | BPE Token ID |
|----------|-----|-----------|----------------|--------------|
| ș, Ș | /ʃ/ | `sh` | English digraph for /ʃ/ (e.g., "ship") | 120 |
| ş, Ş | /ʃ/ | `sh` | Cedilla variant (same phoneme) | 120 |
| ț, Ț | /ts/ | `ts` | English cluster /ts/ (e.g., "bits") | 192 |
| ţ, Ţ | /ts/ | `ts` | Cedilla variant (same phoneme) | 192 |

**Vowel handling (characters already in vocab):**

| Romanian | IPA | Treatment | Token ID |
|----------|-----|-----------|----------|
| ă | /ə/ | Kept as-is | 2413 |
| â | /ɨ/ | Kept as-is | 395 |
| î | /ɨ/ | Kept as-is | 407 |
| Ă → ă | — | Lowercased | 2413 |
| Â → â | — | Lowercased | 395 |
| Î → î | — | Lowercased | 407 |

**Additional normalization:**
- All text lowercased (reduces token diversity, aids generalization)
- Unicode normalization (NFC) applied to handle decomposed forms
- Romanian-specific punctuation normalization (e.g., „ and " quotes)

**Implementation:** `vendor/chatterbox-finetuning/src/romanian_preprocessor.py` (334 lines)

Two modes are available:
- **Phoneme mode** (default): Only maps missing consonants; preserves existing vowels
- **ASCII mode** (fallback): Maps all diacritics to ASCII (`ă→a`, `â→a`, `î→i`)

### 3.3. Examples

```
Phoneme mode:
  "Știință și înțelepciune."  →  "shtiintsă shi întselepciune."
  "Țara mea frumoasă."       →  "tsara mea frumoasă."
  "De asemenea, contează și dacă imobilul este la stradă sau nu."
    →  "de asemenea, contează shi dacă imobilul este la stradă sau nu."

ASCII mode (for comparison):
  "Știință și înțelepciune."  →  "shtiintsa shi intselepciune."
  "Țara mea frumoasă."       →  "tsara mea frumoasa."
```

### 3.4. Verification

After preprocessing the full SWARA dataset (21,304 utterances):

| Metric | Value |
|--------|-------|
| Total `.pt` files generated | 21,304 |
| Maximum text token ID | 2413 |
| Vocabulary size | 2,454 |
| Out-of-vocabulary tokens | **0** |
| Token ID validation (20 random samples) | All pass |

### 3.5. Training Configuration

```python
# Key differences from Experiment 1:
new_vocab_size = 2454      # Original (was 2459)
romanian_preprocessing = True
romanian_mode = "phoneme"
learning_rate = 1e-5
batch_size = 16
save_steps = 50
```

Training is in progress. The T3 model loads pretrained weights with a 1:1 copy (no new tokens to initialize), preserving full embedding quality.

### 3.6. Results

*Training in progress — results to be updated upon completion.*

---

## 4. Comparison and Discussion

### 4.1. Experiment Summary

| Aspect | Exp. 1: Vocab Extension | Exp. 2: Phoneme Mapping |
|--------|------------------------|------------------------|
| Vocabulary size | 2,459 (+5) | 2,454 (original) |
| Missing char ș | New token (ID 2454) | → `sh` (ID 120) |
| Missing char ț | New token (ID 2455) | → `ts` (ID 192) |
| Vowels ă, â, î | Already in vocab | Already in vocab |
| Embedding init | Mean of existing (weak) | Pretrained (strong) |
| Audio quality | Unintelligible | Pending |
| Posterior collapse | Yes | Not expected |
| Information loss | None | Minimal (ș≈sh, ț≈ts phoneme mapping) |

### 4.2. Why Phoneme Mapping Preserves Information

The phoneme mappings are not approximations — they are phonetically exact:

- Romanian /ʃ/ (represented by ș) and English /ʃ/ (represented by "sh") are the same phone. The IPA transcription is identical.
- Romanian /ts/ (represented by ț) and English /ts/ (as in "bits", "cats") are the same consonant cluster.

The only theoretical information loss is at the grapheme level: the model can no longer distinguish ș from the sequence "sh" in text. In practice, this is not a concern because:
1. The TTS model operates on phonetic realization, not orthographic identity
2. Romanian "sh" sequences are extremely rare outside of diacritics
3. The audio signal provides the correct pronunciation regardless

### 4.3. Relationship to Prior Work

This approach is comparable to **grapheme-to-phoneme (G2P) preprocessing** commonly used in TTS systems, where text is converted to a phonemic representation before model input. The key difference is that our mapping is partial (only for characters outside the existing vocabulary) and uses English grapheme sequences rather than IPA symbols, because the base vocabulary is grapheme-based, not phoneme-based.

The strategy is also related to **transliteration-based adaptation** methods used in multilingual NLP, where unseen scripts are mapped to a shared character set. Our approach is more conservative — we only transliterate the minimum number of characters needed for vocabulary coverage.

---

## 5. References

1. gokhaneraslan/chatterbox-finetuning — Community fine-tuning toolkit
   - Issue #6: Turkish gibberish audio (https://github.com/gokhaneraslan/chatterbox-finetuning/issues/6)
   - Issue #12: Norwegian/German posterior collapse (https://github.com/gokhaneraslan/chatterbox-finetuning/issues/12)
   - Issue #14: Arabic quality issues (https://github.com/gokhaneraslan/chatterbox-finetuning/issues/14)
   - Issue #8: Multilingual challenges (https://github.com/gokhaneraslan/chatterbox-finetuning/issues/8)
2. stlohrey/chatterbox-finetuning — Alternative fine-tuning repo (fork of resemble-ai/chatterbox)
3. ResembleAI/chatterbox — Original Chatterbox model and tokenizer
4. SWARA 1.0 — Romanian speech corpus (21,304 utterances, 18 speakers, 21.67 hours)
