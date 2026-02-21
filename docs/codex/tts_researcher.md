 Summary (What’s feasible now)

  - The official Chatterbox repo provides installation and inference
    usage, plus a multilingual model with a fixed language list.
    Romanian is not listed among supported languages in the README, so
    a Romanian model will require adaptation beyond default inference.
    (github.com (https://github.com/resemble-ai/chatterbox))
  - A community fine-tuning kit exists that explicitly targets
    Chatterbox/Turbo fine-tuning, includes multilingual vocabulary
    extension, dataset formatting guidance, and preprocessing
    requirements. This is currently the most concrete pathway for
    Romanian adaptation that aligns with the official model family.
    (github.com (https://github.com/gokhaneraslan/chatterbox-
    finetuning))
  - SWARA is a viable starting corpus for Romanian (read speech, 21+
    hours, 17 speakers, 19,279 utterances with transcripts). It’s
    research-licensed and requires a license agreement.
    (speech.utcluj.ro (https://speech.utcluj.ro/swarasc/index.html))

  ———

  1. Core Resources (Official + Community + Dataset)

  - Official model repo: resemble-ai/chatterbox (model zoo,
    installation, inference examples, supported languages list).
    (github.com (https://github.com/resemble-ai/chatterbox))
  - Official package: chatterbox-tts on PyPI (versioning, citation).
    (pypi.org (https://pypi.org/project/chatterbox-tts/?
    utm_source=openai))
  - Community fine-tuning kit: gokhaneraslan/chatterbox-finetuning
    (fine-tune pipeline for Chatterbox and Turbo, multilingual vocab
    extension, preprocessing, dataset format, sample rates).
    (github.com (https://github.com/gokhaneraslan/chatterbox-
    finetuning))
  - Dataset: SWARA corpus (Romanian read speech; 21+ hours, 17
    speakers, 19,279 utterances; includes transcripts; research
    license agreement; CC BY‑NC 4.0). (speech.utcluj.ro (https://
    speech.utcluj.ro/swarasc/index.html))

  ———

  2. Architecture Fit and Gap for Romanian

  - Official multilingual model supports a fixed set of 23 languages;
    Romanian is not listed. (github.com (https://github.com/resemble-
    ai/chatterbox))
  - Implication (inference): Romanian adaptation will need tokenizer/
    vocabulary coverage plus fine-tuning. This is aligned with the
    community finetuning kit’s design for new languages. (github.com
    (https://github.com/resemble-ai/chatterbox))

  ———

  3. Data Requirements and Preparation (SWARA → Chatterbox)
  From the community finetuning kit:

  - Dataset format: LJSpeech-style metadata.csv with filename|
    raw_text|normalized_text and a wavs/ directory. (github.com
    (https://github.com/gokhaneraslan/chatterbox-finetuning))
  - Audio sample rate: input resampled to 16 kHz; vocoder outputs at
    24 kHz. (github.com (https://github.com/gokhaneraslan/chatterbox-
    finetuning))
  - Preprocessing is mandatory: offline preprocessing extracts speaker
    embeddings and acoustic tokens to .pt files for faster training.
    (github.com (https://github.com/gokhaneraslan/chatterbox-
    finetuning))

  For SWARA:

  - Scope: 21+ hours, 17 speakers, 19,279 utterances, transcripts
    included. (speech.utcluj.ro (https://speech.utcluj.ro/swarasc/
    index.html))
  - Licensing: research license agreement required; CC BY‑NC 4.0.
    (speech.utcluj.ro (https://speech.utcluj.ro/swarasc/index.html))

  Actionable data prep steps

  1. Normalize SWARA transcripts into raw_text and normalized_text
     (Romanian-specific normalization).
  2. Convert audio to the expected format; let the pipeline resample
     to 16 kHz.
  3. Create train/val split per speaker (recommend balancing across
     speakers).
  4. Run the preprocessing stage in the community kit.

  ———

  4. Tokenizer and Vocabulary Adaptation
  From the community fine-tuning kit:

  - Tokenizer: grapheme (character) based for Standard mode, with a
    default multilingual set of 23 languages. (github.com (https://
    github.com/gokhaneraslan/chatterbox-finetuning))
  - Customization: if target language has characters not in the
    default tokenizer, create a custom tokenizer.json. (github.com
    (https://github.com/gokhaneraslan/chatterbox-finetuning))
  - Critical requirement: NEW_VOCAB_SIZE must exactly match the token
    count in tokenizer.json. (github.com (https://github.com/
    gokhaneraslan/chatterbox-finetuning))

  Romanian-specific implication (inference)
  You should verify whether Romanian characters are present in the
  default tokenizer and extend if needed; otherwise you risk OOV
  behavior and degraded pronunciation. (github.com (https://
  github.com/gokhaneraslan/chatterbox-finetuning))

  ———

  5. Model Choice: Standard vs Turbo
  From the community kit:

  - Standard mode: Llama-based, smaller multilingual grapheme vocab;
    good for deeper language control. (github.com (https://github.com/
    gokhaneraslan/chatterbox-finetuning))
  - Turbo mode: GPT-2 based, large English vocab plus automatic
    multilingual extension; good for faster, high-quality fine-tuning
    from English base. (github.com (https://github.com/gokhaneraslan/
    chatterbox-finetuning))
  - Mode switching caveat: must delete pretrained_models and
    preprocessed dir when switching modes to avoid corrupted token
    files. (github.com (https://github.com/gokhaneraslan/chatterbox-
    finetuning))

  Recommendation (pragmatic)

  - Start with Turbo mode for faster convergence; keep Standard as a
    fallback if Romanian pronunciation quality is weak or if you need
    tighter control. This aligns with the community kit’s “smart
    merging” approach for non-English languages. (github.com (https://
    github.com/gokhaneraslan/chatterbox-finetuning))

  ———

  6. Training Workflow (Best Practices Aligned with Community Tooling)

  1. Install official Chatterbox and confirm inference works. The
     official repo documents Python 3.11 on Debian 11 and pinned
     dependencies. (github.com (https://github.com/resemble-ai/
     chatterbox))
  2. Set up fine-tuning kit and choose mode (is_turbo flag).
     (github.com (https://github.com/gokhaneraslan/chatterbox-
     finetuning))
  3. Prepare SWARA in LJSpeech format. (github.com (https://
     github.com/gokhaneraslan/chatterbox-finetuning))
  4. Run preprocessing to generate .pt artifacts. (github.com
     (https://github.com/gokhaneraslan/chatterbox-finetuning))
  5. Train with correct NEW_VOCAB_SIZE matching tokenizer. (github.com
     (https://github.com/gokhaneraslan/chatterbox-finetuning))
  6. Inference check using Romanian prompts and a Romanian reference
     clip (zero-shot or voice-cloned).

  ———

  7. Evaluation Plan (Primary-Source Based)
  Subjective

  - MOS (Mean Opinion Score): use ITU‑T MOS terminology and test
    definitions for clarity and comparability. (itu.int (https://
    www.itu.int/dms_pubrec/itu-t/rec/p/T-REC-P.800.1-201602-
    S%21%21SUM-HTM-E.htm?utm_source=openai))
  - MUSHRA: use ITU‑R BS.1534 for multi-stimulus subjective quality
    assessment. (itu.int (https://www.itu.int/rec/R-REC-BS.1534/en?
    utm_source=openai))

  Objective

  - WER via ASR: measure intelligibility by running ASR on generated
    speech and computing WER (defined using substitutions, deletions,
    insertions over reference words). (catalog.ldc.upenn.edu (https://
    catalog.ldc.upenn.edu/docs/LDC2002S25/hub5nev3.htm?
    utm_source=openai))

  Practical evaluation setup
  ———

  8. Risks and Mitigations

    fidelity zero-shot quality → consider speaker filtering,
    augmentation, or adding other Romanian datasets (if license-
    compatible). (speech.utcluj.ro (https://speech.utcluj.ro/swarasc/
    index.html))
  - License: SWARA is CC BY‑NC 4.0 and requires a license agreement;
    swarasc/index.html))

  ———