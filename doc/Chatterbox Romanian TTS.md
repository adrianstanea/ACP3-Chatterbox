# State of the Art: Adapting the Chatterbox Architecture to the Romanian Language

## Executive Summary

This report synthesizes the recent literature relevant to composing the introduction of a research paper on adapting Resemble AI's Chatterbox text-to-speech (TTS) architecture to the Romanian language. It covers the evolution of modern zero-shot TTS systems, the specific architectural details of Chatterbox and related models, existing work on Romanian speech synthesis, available Romanian speech corpora, and methods for low-resource language adaptation. The review spans key publications from 2023–2025 across neural speech synthesis, multilingual TTS, and Romanian-specific speech technology.

***

## Modern Zero-Shot Text-to-Speech Architectures

### The Neural Codec Language Model Paradigm

The landscape of TTS was fundamentally reshaped by VALL-E (Wang et al., 2023), which reframed speech synthesis as a conditional language modeling task over discrete audio codes derived from a neural audio codec. By conditioning on just a 3-second acoustic prompt, VALL-E demonstrated that a language model could achieve zero-shot voice cloning, significantly outperforming prior systems like YourTTS in both speech naturalness and speaker similarity. This paradigm—treating TTS as token prediction rather than continuous signal regression—opened the door to scaling speech models with the same strategies used for large language models, including pretraining on massive unlabeled corpora.[^1][^2][^3]

Building on this, several systems extended the neural codec LM approach. VALL-E X adapted the framework for cross-lingual zero-shot TTS. NaturalSpeech 3 (Ju et al., 2024) introduced factorized vector quantization (FACodec) to disentangle speech into subspaces of content, prosody, timbre, and acoustic details, paired with factorized diffusion models for attribute-specific generation. Scaling to 1B parameters and 200K hours of training data, NaturalSpeech 3 achieved on-par quality with human recordings. Seed-TTS (ByteDance, 2024) further explored language-model-based foundations for TTS, incorporating a reinforcement learning approach for improved robustness, speaker similarity, and controllability, alongside a fully diffusion-based non-autoregressive variant.[^4][^5][^6][^7][^8]

### Flow Matching as a Generative Backbone

An important parallel development has been the adoption of flow matching for speech synthesis. Unlike diffusion models that iteratively denoise signals, flow matching learns a continuous probability flow solved via an Ordinary Differential Equation (ODE), enabling stable, high-fidelity synthesis with fewer inference steps. F5-TTS (Chen et al., 2024/2025) is a prominent example: a fully non-autoregressive system based on flow matching with a Diffusion Transformer (DiT) backbone. F5-TTS eliminates the need for duration models, explicit text encoders, or phoneme alignments—text input is simply padded with filler tokens to match mel-spectrogram length, and denoising generates speech. It uses a ConvNeXt module for text encoding and was trained on a public 100K-hour multilingual dataset, exhibiting strong zero-shot ability and seamless code-switching. Recent work on Fast F5-TTS has demonstrated that inference can be accelerated to 7 steps (4× faster) through Empirically Pruned Step Sampling (EPSS) without retraining.[^9][^10][^11][^12][^13][^14][^15]

CosyVoice (Du et al., 2024) proposed supervised semantic tokens derived from a multilingual ASR model, combined with an LLM for text-to-token generation and a conditional flow matching model for token-to-speech synthesis. Its successor, CosyVoice 2, achieves human-parity naturalness with a unified streaming and offline architecture, supporting robust code-switching and zero-shot voice cloning across multiple languages.[^16][^17]

***

## The Chatterbox Architecture

### Chatterbox (Original, 2025)

Chatterbox, developed by Resemble AI, is a family of state-of-the-art open-source TTS models released under the MIT license. The original Chatterbox model employs a non-autoregressive Flow Matching architecture built on a 0.5B-parameter Llama transformer backbone. Trained on approximately 500,000 hours of cleaned speech data, it achieves production-grade zero-shot voice cloning from just a few seconds of reference audio.[^18][^19][^20][^21][^9]

The architectural pipeline operates in several stages:[^22]

1. **EnTokenizer**: A BPE-based tokenizer converts input text into text tokens.
2. **T3 Conditioning**: Uses voice encoder embeddings and reference speech tokens to guide the generation of speech tokens from text tokens.
3. **S3 Conditioning**: Uses reference speech tokens, prompt-speech mel-spectrograms, and speaker embeddings to synthesize realistic audio from the generated speech tokens.
4. **Mel Generation & Vocoder**: Speech tokens are converted to mel-spectrograms via a Conditional Flow Matching (CFM) module, then transformed into waveforms by a vocoder.

A distinctive feature of Chatterbox is its emotion exaggeration control, allowing users to adjust the emotional intensity of synthesized speech—a capability absent in most competing open-source models. All generated audio includes a PerTh (Perceptual Threshold) watermark for traceability and ethical AI compliance.[^19][^21][^18]

### Chatterbox Multilingual (2025)

Chatterbox Multilingual extends the original framework to support zero-shot voice cloning across 23 languages, including Arabic, Hindi, Chinese, Swahili, Turkish, and many others. With a 500M-parameter architecture, it introduces emotion categories (e.g., happy, sad, angry) and an exaggeration parameter, enabling expressive delivery vital for gaming, dialogue agents, and storytelling. Notably, Romanian is not among the explicitly listed supported languages, motivating adaptation research.[^23][^24]

### Chatterbox Turbo (2025)

Chatterbox Turbo is the most efficient variant, built on a streamlined 350M-parameter architecture. Its key innovation is the distillation of the speech-token-to-mel decoder from 10 steps to a single step, achieving up to 6× faster-than-real-time generation with sub-200ms time-to-first-sound. It also natively supports paralinguistic tags (e.g., `[laugh]`, `[cough]`, `[chuckle]`) for added realism.[^25][^18]

***

## Multilingual Zero-Shot TTS Systems

The push toward multilingual zero-shot TTS has produced several significant systems beyond Chatterbox. The table below compares the most relevant architectures:

| System | Year | Architecture | Languages | Key Innovation |
|--------|------|-------------|-----------|----------------|
| VALL-E / VALL-E X | 2023 | Neural codec LM (AR) | EN / Cross-lingual | Language modeling paradigm for TTS[^2][^5] |
| XTTS | 2024 | GPT-2 based AR transformer | 17 | Massively multilingual ZS-TTS, built on Tortoise[^26] |
| CosyVoice 2 | 2024–25 | LLM + chunk-aware causal flow matching | 9+ major | Supervised semantic tokens, streaming + offline[^17] |
| F5-TTS | 2024–25 | Non-AR, flow matching + DiT | Multilingual | No duration model, character-level, ConvNeXt text encoder[^14] |
| NaturalSpeech 3 | 2024 | Factorized diffusion models | Multi-speaker | FACodec disentanglement, 1B params[^7] |
| Seed-TTS | 2024 | LM-based + DiT variant | Multi | RL for robustness, speaker similarity[^4] |
| Chatterbox | 2025 | Flow matching + 0.5B Llama | EN (orig), 23 (multilingual) | Emotion control, PerTh watermark, MIT license[^18][^23] |
| MMS-TTS | 2023–24 | VITS-based | 1,100+ | Massive language coverage, religious text data[^27][^28] |

A critical observation for Romanian is that most high-quality ZS-TTS systems (XTTS, Chatterbox, CosyVoice) do not natively support Romanian, while MMS-TTS covers Romanian but with comparatively lower synthesis quality as it relies on a VITS architecture trained on limited data.[^29][^27]

***

## Romanian Speech Synthesis: Existing Work

### Classical and Early Neural Approaches

Early work on Romanian TTS employed unit concatenation techniques with a Harmonic plus Noise Model (HNM) for signal processing, focusing on a natural language processing pipeline for input text analysis, letter-to-phone conversion, and prosody estimation. These systems achieved acceptable intelligibility but struggled with prosodic naturalness—incorrect accents, unnatural pauses, and inconsistent melodic profiles remained persistent challenges.[^30]

### FastPitch Adaptation (Răgman & Stan, 2024)

Răgman and Stan (2024) adapted the FastPitch architecture to Romanian, extending it from single-speaker to 18 speakers by conditioning on speaker embeddings extracted with TitaNet-L. Their work explored speaker anonymization (synthesizing with an anonymous identity) and speaker adaptation for unseen speakers using short reference samples. The study used the SWARA corpus and demonstrated that non-autoregressive architectures could produce natural-sounding Romanian speech, though the system was not designed for zero-shot voice cloning.[^31][^32]

### F5-TTS-RO: Lightweight Input Adaptation (Chivereanu & Boros, December 2025)

The most directly relevant recent work is F5-TTS-RO by Chivereanu and Boros (2025), which introduced a lightweight, non-invasive ConvNeXt-based input adapter to extend F5-TTS to Romanian. The adapter replaces the character embedding layer with a new learnable matrix plus a 1D ConvNeXt module that produces continuous "soft" letter-to-sound representations, while keeping the entire F5-TTS model frozen. This design preserves the original model's capabilities (English, Chinese, voice cloning) and reduces overfitting risk.[^33][^29]

Key results from evaluation with 20 native Romanian speakers and objective metrics on 1,000 samples from Common Voice:[^29]

| Metric | MMS-TTS-RON | RO-F5TTS (Adapter) | F5-TTS Full Fine-tune |
|--------|-------------|--------------------|-----------------------|
| WER (↓) | 5.77% | 5.27% | 3.62% |
| Speaker Similarity (mean cosine) | — | 0.9013 | 0.7946 |

The adapter achieved lower WER than MMS-TTS-RON and substantially higher speaker similarity than full fine-tuning, highlighting a trade-off between pronunciation accuracy and voice cloning fidelity. However, residual English accent characteristics were observed in some synthesized Romanian samples, and the code-switching capability (Romanian-English within a single utterance) showed promise but remained limited.[^29]

### MMS-TTS for Romanian

Meta's Massively Multilingual Speech project (Pratap et al., 2023/2024) includes a Romanian TTS model (MMS-TTS-RON) based on the VITS architecture trained on Romanian data from religious texts. While it provides explicit Romanian support, it operates as a single-speaker system without zero-shot voice cloning capabilities, and its synthesis quality is below that of state-of-the-art models like F5-TTS or Chatterbox.[^27][^28][^29]

***

## Romanian Speech Corpora and Resources

### The SWARA Corpus

The SWARA Speech Corpus (Stan et al., 2017) is one of the largest freely available Romanian speech datasets. It comprises over 21 hours of high-quality recordings from 17 different speakers, segmented into 19,279 utterances with orthographic transcripts and semi-automatic phone-level alignments. Recorded at 48 kHz in a professional studio environment, it is released under a CC-BY-SA 4.0 license. SWARA has served as the primary training corpus for recent Romanian TTS work, including both the FastPitch adaptation and F5-TTS-RO.[^32][^34][^35][^29]

### Additional Resources

The Romanian Anonymous Speech Corpus (RASC) contains approximately 3,000 utterances from various speakers. The RSC corpus offers over 100 hours of speech from 157 speakers collected via an online platform, primarily for automatic speech recognition. Common Voice (Ardila et al., 2020) also includes a Romanian subset that has been used for evaluation purposes in TTS research. The RoDia dataset (2024) provides 2 hours of dialect-annotated speech from five Romanian regions.[^36][^34][^29]

### Romanian ASR as a Complementary Resource

Pîrlogeanu et al. (2025) recently presented a state-of-the-art Romanian ASR system based on NVIDIA's FastConformer architecture, trained on over 2,600 hours of weakly supervised transcriptions, achieving up to 27% relative WER reduction over prior systems. Such ASR models are valuable not only for evaluating TTS output (via WER on transcribed synthesized speech) but also as potential reward signals for reinforcement-learning-based TTS optimization, as demonstrated in recent work.[^37][^38][^39]

***

## Strategies for Low-Resource Language Adaptation in TTS

### Adapter-Based Approaches

Adapter-based methods, as exemplified by F5-TTS-RO, train lightweight modules while freezing pretrained model weights. This approach mitigates catastrophic forgetting—where fine-tuning degrades previously acquired knowledge—and is especially effective when training data is limited. The F5-TTS-RO adapter required only 21 hours of Romanian data and trained for 12 hours on a single NVIDIA A100 GPU.[^29]

### Reinforcement Learning with ASR Feedback

Align2Speak (2025) proposed a GRPO-based framework for adapting multilingual TTS models to low-resource languages using ASR models as reward signals. The pipeline first pretrains a multilingual TTS baseline with IPA tokens, fine-tunes on limited paired data of the target language, and then applies GRPO with multi-objective rewards (ASR intelligibility, speaker verification, audio quality). With only 30 minutes of Portuguese data, CER dropped from 33% to 3.94% after fine-tuning and GRPO, suggesting that preference alignment can dramatically bridge the gap for new languages. A similar approach, F5R-TTS, integrates GRPO directly into flow-matching-based architectures by reformulating deterministic outputs as probabilistic Gaussian distributions.[^40][^37]

### Data Augmentation and Transfer Learning

Low-resource conversational TTS has been addressed through F0-conditioned voice conversion as a data augmentation technique, enabling natural conversational synthesis with as little as 1 hour of data when combined with non-expressive data in the same language. Transfer learning from high-resource to low-resource languages, particularly through multilingual pretraining with IPA tokenization, has proven effective across typologically diverse languages.[^41][^42][^37]

### Full Fine-Tuning vs. Frozen Adaptation

The F5-TTS-RO study directly compared adapter-based and full fine-tuning approaches, finding a clear trade-off: full fine-tuning yielded better WER (3.62% vs. 5.27%) but significantly worse speaker similarity (cosine 0.7946 vs. 0.9013). This trade-off is critical for zero-shot voice cloning applications and suggests that hybrid strategies—potentially combining adapters with selective parameter unfreezing—merit investigation.[^29]

***

## Gaps and Motivation for Chatterbox-Romanian Adaptation

Several key observations emerge from this literature review that motivate the adaptation of Chatterbox to Romanian:

- **Chatterbox's architecture is distinct from F5-TTS**: While F5-TTS uses a DiT-based flow matching backbone with character-level padding, Chatterbox employs a Llama-based transformer with BPE tokenization and a two-stage (T3→S3) conditioning pipeline. Adapting Chatterbox thus poses different challenges and opportunities than adapting F5-TTS.[^22][^9]

- **Romanian is not natively supported**: Neither the original English-only Chatterbox nor the 23-language Chatterbox Multilingual explicitly includes Romanian. Existing Romanian TTS options remain limited to MMS-TTS-RON (VITS-based, single-speaker), the FastPitch adaptation, and the recent F5-TTS-RO adapter.[^23][^18][^32][^27][^29]

- **Zero-shot voice cloning for Romanian is underexplored**: Only F5-TTS-RO has demonstrated zero-shot cloning for Romanian, and it reported residual accent issues. Chatterbox's unique architecture—including its exaggeration control and alignment-informed inference—may offer complementary capabilities.[^29]

- **Flow matching adaptation strategies are not yet explored for Chatterbox**: Prior adaptation work for Romanian has targeted FastPitch (non-autoregressive, duration-model-based) and F5-TTS (flow matching, DiT-based). Chatterbox's Llama-based flow matching backbone has not been adapted to any new language in the published literature.

- **Romanian-specific challenges persist**: The Romanian phonological system—including diacritics (ă, â, î, ș, ț), vowel harmony patterns, and prosodic characteristics—requires specialized handling. Residual English accent artifacts in F5-TTS-RO suggest that input-level adaptation alone may be insufficient.[^29]

- **Reinforcement learning alignment is a promising but unexplored direction for Chatterbox**: Given the availability of Romanian ASR models and the demonstrated effectiveness of GRPO-based TTS alignment, integrating preference optimization into a Chatterbox adaptation pipeline could significantly improve intelligibility and naturalness beyond supervised fine-tuning alone.[^38][^37][^40]

***

## Conclusion

The rapid evolution of zero-shot TTS architectures—from neural codec language models to flow-matching-based systems—has dramatically expanded the capabilities of speech synthesis. However, Romanian remains a low-to-medium resource language in the TTS domain, with existing solutions either lacking voice cloning capabilities (MMS-TTS-RON) or exhibiting residual accent artifacts (F5-TTS-RO). The Chatterbox architecture, with its Llama-based flow matching backbone, emotion control, and demonstrated state-of-the-art performance in English, presents a compelling foundation for Romanian adaptation. This adaptation would contribute the first exploration of Chatterbox's architecture for a new language, potentially leveraging adapter strategies, reinforcement learning with ASR feedback, and Romanian-specific phonological processing to achieve high-quality, expressive, zero-shot Romanian TTS.

---

## References

1. [VALL-E architecture - learnius](https://learnius.com/slp/9+Speech+Synthesis/2+Advanced+Topics/4+Neural+Codec+Language+Modeling/VALL-E+architecture) - Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers introducing a language modeli...

2. [[PDF] Neural Codec Language Models are Zero-Shot Text to Speech ...](https://arxiv.org/pdf/2301.02111.pdf) - VALL-E directly enables various speech synthesis applications, such as zero-shot TTS, speech editing...

3. [Neural Codec Language Models are Zero-Shot Text to ...](https://arxiv.org/abs/2301.02111) - Vall-E significantly outperforms the state-of-the-art zero-shot TTS system in terms of speech natura...

4. [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models](http://arxiv.org/pdf/2406.02430.pdf) - ...a reinforcement learning approach to enhance model robustness, speaker
similarity, and controllab...

5. [VALL-E - Microsoft](https://www.microsoft.com/en-us/research/project/vall-e-x/) - VALL-E emerges in-context learning capabilities and can be used to synthesize high-quality personali...

6. [NaturalSpeech 3 - Speech Research](https://speechresearch.github.io/naturalspeech3/) - We propose NaturalSpeech 3, a TTS system with novel factorized diffusion models to generate natural ...

7. [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized ...](https://arxiv.org/abs/2403.03100) - We propose NaturalSpeech 3, a TTS system with novel factorized diffusion models to generate natural ...

8. [[PDF] NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized ...](https://arxiv.org/pdf/2403.03100.pdf) - In this paper, we develop a TTS system that consists of 1) a novel neural speech codec with factoriz...

9. [Calliope: A TTS-based Narrated E-book Creator Ensuring Exact ...](https://arxiv.org/html/2602.10735) - The implementation supports both XTTS-v2 casanova2024xtts and Chatterbox chatterboxtts2025 for audio...

10. [Generative Pre-training for Speech with Flow Matching](https://arxiv.org/pdf/2310.16338.pdf) - ...directly. In this work, we take a step toward this direction by
showing a single pre-trained gene...

11. [Accelerating Flow-Matching-Based Text-to-Speech via Empirically ...](https://arxiv.org/html/2505.19931v1) - We propose Fast F5-TTS, a training-free, plug-and-play approach that accelerates inference in FM-OT-...

12. [F5-TTS: Flow-Matching Text-to-Speech System - Emergent Mind](https://www.emergentmind.com/topics/f5-tts) - F5-TTS is a non-autoregressive TTS framework that leverages conditional flow matching to transform G...

13. [[PDF] F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech ...](https://aclanthology.org/2025.acl-long.313.pdf) - This paper introduces F5-TTS, a fully non- autoregressive text-to-speech system based on flow matchi...

14. [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech ...](https://arxiv.org/abs/2410.06885) - This paper introduces F5-TTS, a fully non-autoregressive text-to-speech system based on flow matchin...

15. [Accelerating Flow-Matching-Based Text-to-Speech ... - ISCA Archive](https://www.isca-archive.org/interspeech_2025/zheng25d_interspeech.html) - To this end, we introduce Fast F5-TTS, a training-free approach to accelerate the inference of flow-...

16. [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech ... - arXiv](https://arxiv.org/abs/2407.05407) - In this paper, we propose to represent speech with supervised semantic tokens, which are derived fro...

17. [CosyVoice2: Zero-Shot Multilingual TTS Model - Emergent Mind](https://www.emergentmind.com/topics/cosyvoice2-tts-model) - CosyVoice2 is a zero-shot multilingual TTS model that integrates advanced supervised semantic tokeni...

18. [resemble-ai/chatterbox: SoTA open-source TTS - GitHub](https://github.com/resemble-ai/chatterbox) - We are excited to introduce Chatterbox-Turbo, our most efficient model yet. Built on a streamlined 3...

19. [Chatterbox: An Open Source Breakthrough in Speech Synthesis](https://www.actuia.com/en/news/chatterbox-an-open-source-breakthrough-in-speech-synthesis/) - Chatterbox is based on a 0.5 billion parameter architecture, trained on 500,000 hours of cleaned dat...

20. [Calliope: A TTS-based Narrated E-book Creator Ensuring ... - arXiv](https://arxiv.org/html/2602.10735v1) - Chatterbox employs a non-autoregressive Flow Matching architecture built on a 0.5B-parameter Llama t...

21. [README.md · ResembleAI/chatterbox at ...](https://huggingface.co/ResembleAI/chatterbox/blob/d828d02b6e80903a6db69c3a20a180c34ecb196f/README.md) - We're excited to introduce Chatterbox, Resemble AI's first production-grade open source TTS model. L...

22. [Exploring Chatterbox: A Voice Cloning TTS Model - LinkedIn](https://www.linkedin.com/posts/abhishek-kumar-tiwari-b24952208_tts-chatterbox-chatterbox-activity-7340069237715030018-Q_pd) - ... Chatterbox. #Chatterbox is a voice cloning #TTS model that takes text and a reference audio clip...

23. [Chatterbox Multilingual: Open-Source TTS for Zero-Shot Cloning](https://tech-now.io/en/blogs/chatterbox-multilingual-open-source-zero-shot-tts) - Resemble AI has launched Chatterbox Multilingual—a production-grade, open-source Text-to-Speech (TTS...

24. [Chatterbox multilingual text to speech model released - Facebook](https://www.facebook.com/groups/DeepNetGroup/posts/2584933958566137/) - Resemble AI Chatterbox (Model B) is strongly preferred (8.75%) (25.00%) ElevenLabsPreferred ElevenLa...

25. [Resemble AI Unleashes Chatterbox Turbo: A New Era for Open ...](https://markets.financialcontent.com/wral/article/tokenring-2025-12-15-resemble-ai-unleashes-chatterbox-turbo-a-new-era-for-open-source-real-time-voice-ai) - At the heart of Chatterbox Turbo's prowess lies a streamlined 350M parameter architecture, a signifi...

26. [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](http://arxiv.org/pdf/2406.04904.pdf) - Most Zero-shot Multi-speaker TTS (ZS-TTS) systems support only a single
language. Although models li...

27. [Scaling Speech Technology to 1000+ Languages - arXiv](https://arxiv.org/abs/2305.13516) - The Massively Multilingual Speech (MMS) project increases the number of supported languages by 10-40...

28. [[PDF] Scaling Speech Technology to 1,000+ Languages](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf) - The Massively Multilingual Speech (MMS) project aims to expand speech technology ... One model, many...

29. [[PDF] Extending F5-TTS to Romanian TTS via Lightweight Input Adaptation](https://www.arxiv.org/pdf/2512.12297.pdf) - This work introduces a lightweight input-level adapter for the F5-TTS model that enables Romanian La...

30. [[PDF] RECENT ADVANCES IN ROMANIAN LANGUAGE TEXT-TO ...](https://acad.ro/sectii2002/proceedings/doc2010-1/13-Burileanu.pdf) - This paper describes our recent work in developing a high-quality text-to-speech synthesis system in...

31. [Efficient training strategies for natural sounding speech synthesis ...](https://arxiv.org/abs/2410.06787) - This paper focuses on adapting the functionalities of the FastPitch model to the Romanian language; ...

32. [Efficient training strategies for natural sounding speech synthesis ...](https://arxiv.org/html/2410.06787v1) - This paper focuses on adapting the functionalities of the FastPitch model to the Romanian language; ...

33. [Extending F5-TTS to Romanian TTS via Lightweight Input Adaptation](https://www.arxiv.org/abs/2512.12297) - Abstract:This work introduces a lightweight input-level adapter for the F5-TTS model that enables Ro...

34. [[PDF] The SWARA Speech Corpus: A Large Parallel Romanian Read ...](https://adrianastan.com/papers/2017_SPED_Swara.pdf) - The dataset comprises speech data recorded over the last year from 12 speakers, along with 5 other s...

35. [SWARA Corpus - A Large Parallel Romanian Read Speech Dataset](https://speech.utcluj.ro/swarasc/) - The corpus contains over 21 hours of high quality recordings from 17 different speakers. The data is...

36. [RoDia: A New Dataset for Romanian Dialect Identification from Speech](https://arxiv.org/pdf/2309.03378.pdf) - We introduce RoDia, the first dataset for Romanian dialect identification
from speech. The RoDia dat...

37. [Align2Speak: Improving TTS for Low Resource Languages via ASR ...](https://arxiv.org/html/2509.21718v1) - This section outlines our proposed framework for adapting a multilingual text-to-speech (TTS) model ...

38. [Open Source State-Of-the-Art Solution for Romanian Speech ...](https://arxiv.org/html/2511.03361v1) - In this work, we present a new state-of-the-art Romanian Automatic Speech Recognition (ASR) system b...

39. [Open Source State-Of-the-Art Solution for Romanian Speech ...](https://arxiv.org/abs/2511.03361) - In this work, we present a new state-of-the-art Romanian Automatic Speech Recognition (ASR) system b...

40. [F5R-TTS: Improving Flow-Matching based Text-to-Speech with Group
  Relative Policy Optimization](https://arxiv.org/html/2504.02407) - We present F5R-TTS, a novel text-to-speech (TTS) system that integrates
Gradient Reward Policy Optim...

41. [Low-data? No problem: low-resource, language-agnostic conversational
  text-to-speech via F0-conditioned data augmentation](http://arxiv.org/pdf/2207.14607.pdf) - The availability of data in expressive styles across languages is limited,
and recording sessions ar...

42. [[PDF] Speech LLMs in Low-Resource Scenarios - ISCA Archive](https://www.isca-archive.org/interspeech_2025/fong25_interspeech.pdf) - Overall, this work presents the following contributions to provide insights into the applicability o...

