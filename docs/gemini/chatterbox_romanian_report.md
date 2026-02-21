# Romanian Language Adaptation Report: Chatterbox TTS

This report outlines the methodical approach to adapting, training, and evaluating the **Chatterbox TTS** model (developed by Resemble AI) for the **Romanian language**, utilizing the **SWARA dataset**.

---

## 1. Architectural Overview & Requirements

Chatterbox is an LLM-based Text-to-Speech system featuring a **0.5 billion parameter Llama backbone**. Unlike classical acoustic models, it treats speech synthesis as a language modeling task over discrete or alignment-based tokens.

### Core Requirements for Romanian Training:
*   **Audio Quality**: Chatterbox typically operates at **24kHz or 44.1kHz**. The **SWARA dataset** (48kHz) exceeds these requirements, providing high-fidelity source material.
*   **Text Processing**: The model consumes plain text or phonemes. For Romanian, handling diacritics (**ă, â, î, ș, ț**) is critical.
*   **Alignment**: While Chatterbox uses internal alignment mechanisms, having the **phone-level alignments** provided by SWARA (in HTS format) is a significant advantage for initializing or validating the duration/alignment components.

---

## 2. Linguistic Adaptation for Romanian

Romanian is a phonetic language but requires specific handling to achieve natural prosody and correct pronunciation.

### Phonemization & G2P:
*   **Primary Tool**: `espeak-ng` is the recommended community standard for Romanian phonemization (`-v ro`).
*   **Integration**: Use the International Phonetic Alphabet (IPA) as an intermediate representation. Ensure the Chatterbox tokenizer is updated to include Romanian-specific tokens if they are not already present in the multilingual backbone.
*   **Stress Placement**: Romanian word stress is variable. Pre-processing transcripts with a dictionary-based stress marker (e.g., using `num2words` for normalization) is a best practice.

---

## 3. SWARA Dataset Preparation

The SWARA corpus (21+ hours, 17 speakers) provides the scale needed for a high-quality multi-speaker model.

*   **Transcription Cleanup**: Normalize numbers, dates, and abbreviations (e.g., "str." to "strada") to match the spoken audio exactly.
*   **Format Conversion**: Convert SWARA's HTS-format labels into a format compatible with the Chatterbox training pipeline (typically JSONL or CSV containing `path` and `transcript`).
*   **Speaker Embeddings**: Leverage Chatterbox's zero-shot capabilities by providing speaker reference clips (7-20 seconds) from each of the 17 SWARA speakers to calibrate the speaker-embedding space.

---

## 4. Training Strategy (Transfer Learning)

Directly training a 0.5B parameter model from scratch on 21 hours is inefficient. **Transfer Learning** is the best practice.

1.  **Base Model**: Start with the official `ResembleAI/chatterbox` (Multilingual) weights.
2.  **Fine-tuning**: Use the `chatterbox-tts` library or the `gokhaneraslan/chatterbox-finetuning` toolkit.
3.  **Optimization**: 
    *   **LoRA/QLoRA**: If hardware is limited (e.g., <24GB VRAM), use Low-Rank Adaptation to fine-tune the Llama backbone.
    *   **Freeze Decoder**: Initially freeze the distilled speech-token decoder and only train the LLM backbone on Romanian text-audio pairs.

---

## 5. Evaluation Framework

To ensure the model meets official quality standards, a dual evaluation strategy is required.

### Objective Metrics:
*   **WER (Word Error Rate)**: Use a pre-trained Romanian ASR (like `gigant/romanian-wav2vec2`) to transcribe synthesized speech and compare it to the input text.
*   **MCD (Mel Cepstral Distortion)**: Measure the spectral distance between synthesized and original SWARA audio.
*   **F0 Correlation**: Validate that the prosody and intonation match Romanian linguistic patterns.

### Subjective Metrics:
*   **MOS (Mean Opinion Score)**: Conduct a blind test with native Romanian speakers to rate Naturalness and Similarity (scale 1-5).
*   **Diacritic Stress Test**: Specifically evaluate sentences with minimal pairs (e.g., "paturi" vs. "pături") to check for diacritic-sensitive pronunciation.

---

## 6. Recommended Tools & Best Practices

| Category | Recommended Tool | Why? |
| :--- | :--- | :--- |
| **API/Library** | `chatterbox-tts` (Official) | Seamless integration and real-time performance. |
| **Training Infrastructure** | **Hugging Face Jobs** | Scalable GPU access with `TRL` (Transformer Reinforcement Learning) support. |
| **Monitoring** | **Trackio** | Real-time metric visualization and experiment tracking. |
| **Safety** | **PerTh Watermarking** | Embedded in the official model for ethical AI traceability. |

### Summary Checklist for Readiness:
- [x] **Dataset**: SWARA (48kHz, segmented).
- [x] **Phonemizer**: `espeak-ng` (Romanian support).
- [x] **Base Model**: `ResembleAI/chatterbox` (Multilingual).
- [x] **Training Framework**: PyTorch + `transformers` / `trl`.
