# TTS Evaluation Guide

Use this as a quick reference for selecting evaluation metrics, tools, and test design for TTS research.

## Objective Metrics (Common)
- **MCD (Mel Cepstral Distortion)**: Measures spectral distance; lower is better.
- **F0 Correlation / RMSE**: Measures pitch accuracy; higher correlation is better.
- **WER (Word Error Rate)**: Run ASR on generated audio; lower is better.
- **MOSNet or DNSMOS**: Automated quality proxies; use only for triage, not final claims.

## Subjective Metrics (Gold Standard)
- **MOS (Mean Opinion Score)**: 5-point or 7-point scale; report confidence intervals.
- **MUSHRA**: Use for comparing multiple systems and references; good for fine-grained quality differences.
- **AB / ABX Tests**: Useful for preference or intelligibility comparisons.

## Recommended Test Design
- **Raters**: Target at least 20 to 30 listeners for MOS; more if claims are subtle.
- **Clips**: Balance across speaker, phonetic coverage, and text difficulty.
- **Controls**: Include natural speech and a known baseline system for calibration.
- **Language Considerations**: For new languages, ensure raters are native or near-native.

## Reporting Checklist
- Specify dataset and sampling strategy.
- Provide rater instructions and UI details.
- Report inter-rater reliability if possible.
- Include confidence intervals for MOS or preference rates.
