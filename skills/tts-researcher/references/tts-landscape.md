# TTS Architecture Landscape

Guide for identifying and categorizing TTS models based on their underlying architecture.

## 1. Classical Neural TTS (Encoder-Decoder)
- **Tacotron 2**: Mel-spectrogram generation + WaveNet vocoder.
- **FastSpeech / FastSpeech 2**: Non-autoregressive, uses duration predictors. Fast and robust.

## 2. End-to-End Models (VITS & Successors)
- **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)**: Integrates acoustic model and vocoder. State-of-the-art for many years.
- **VITS2**: Improved stochastic duration predictor and flow-based transformer.
- **MeloTTS**: Optimized for speed and multi-language support.

## 3. Diffusion & Flow-Matching Models
- **Grad-TTS**: Score-based modeling of mel-spectrograms.
- **Matcha-TTS**: Fast flow-matching for high-quality synthesis with fewer steps.
- **E2 TTS / F5-TTS**: Recent breakthrough using large-scale flow-matching, often zero-shot.

## 4. Large Language Model (LLM) Based TTS
- **Vall-E / Vall-E X**: Neural codec language modeling. Excellent zero-shot cloning.
- **GPT-SoVITS**: Powerful zero-shot and few-shot cloning based on GPT and VITS backends.
- **Fish Speech**: SOTA LLM-based speech synthesis using neural codecs.

## 5. Vocoders (Neural Inverters)
- **HiFi-GAN**: Standard high-quality, fast vocoder.
- **BigVGAN**: Improved robustness for out-of-distribution voices.
- **MP-HiFiGAN**: Multi-period analysis for better high-frequency reconstruction.
