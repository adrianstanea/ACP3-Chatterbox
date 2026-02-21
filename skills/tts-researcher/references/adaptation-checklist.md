# Language Adaptation Checklist

Use this checklist when adapting an existing TTS architecture to a new language.

## 1. Data Requirements
- [ ] **Clean Audio**: 5-10 hours for a single-speaker model; 20+ hours for multi-speaker.
- [ ] **Sample Rate**: Minimum 22,050Hz (Standard: 44,100Hz or 48,000Hz).
- [ ] **Transcripts**: Precise text matching the audio exactly (no fillers unless desired).
- [ ] **Alignment**: Does the model need pre-computed alignments (e.g., Montreal Forced Aligner)?

## 2. Linguistic Processing
- [ ] **Phonemizer**: Is there a robust G2P (Grapheme-to-Phoneme) tool for the language?
- [ ] **Character Set**: Does the model handle the specific script (Cyrillic, Arabic, Hanzi)?
- [ ] **Cleaning**: Text normalization rules (numbers, abbreviations, dates).

## 3. Architecture Specifics
- [ ] **Speaker Embeddings**: If multi-speaker, does it use d-vectors or X-vectors?
- [ ] **Pitch/Energy**: Does the model require external pitch extraction during training?
- [ ] **Vocoder Compatibility**: Is the vocoder pre-trained on the target language, or universal?

## 4. Hardware & Training
- [ ] **VRAM**: 12GB minimum for small models; 24GB+ for SOTA transformers/diffusion.
- [ ] **Checkpointing**: Availability of a pre-trained "backbone" (e.g., English VITS) for transfer learning.
- [ ] **Evaluation Set**: 50-100 phrases held out for validation.
