# System card

## Overview

This project compares GLM-OCR and Qwen3-VL-4B-Instruct for single-line transcription of historical handwritten records from Barbados. It includes two LoRA/QLoRA curricula and an optional character-level KenLM decoder for the fine-tuned GLM system.

No base weights, fine-tuned adapters, competition data, predictions, or submissions are distributed in this repository.

## Evaluated systems

| System | Base model | Selected state | Neural training | Inference |
|---|---|---|---:|---|
| Base GLM-OCR | `zai-org/GLM-OCR` | Untouched base | 0 steps / 0 epochs | Greedy |
| Fine-tuned GLM-OCR | `zai-org/GLM-OCR` | Stage 2 checkpoint 642 | 3,521 cumulative steps / 1.5005 cumulative epochs | Greedy |
| Fine-tuned GLM-OCR + KenLM | `zai-org/GLM-OCR` | Stage 2 checkpoint 642 | 3,521 cumulative steps / 1.5005 cumulative epochs | Beam 4 with character 6-gram shallow fusion, lambda 0.10 |
| Base Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | Untouched base | 0 steps / 0 epochs | Greedy, NF4 loading |
| Fine-tuned Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | Stage 2 checkpoint 257 | 1,121 cumulative steps / 0.9509 cumulative epochs | Greedy, NF4 loading |

The Qwen Stage 2 training process was interrupted after checkpoint 257 had been saved and evaluated. Checkpoint 257 is therefore the latest selected available checkpoint, not the output of a completed Stage 2 schedule.

## Training summary

### GLM-OCR

- native OCR prompt: `Text Recognition:`
- rank-16 LoRA over selected language and vision components
- trainable multimodal projector
- Stage 1: clean and mild augmented views
- Stage 2: initialized from the selected Stage 1 checkpoint with additional complex-crop exposure
- selected neural checkpoint: Stage 2 step 642

### Qwen3-VL-4B

- exact-transcription instruction prompt
- NF4 4-bit base-model loading
- rank-16 QLoRA over all applicable linear modules
- Stage 1: clean and augmented views
- Stage 2: initialized from Stage 1 checkpoint 864 with additional complex-crop exposure
- selected available neural checkpoint: Stage 2 step 257

### Character language model

- character-level 6-gram KenLM
- trained only from authorized training labels
- shallow-fusion weight selected on a dedicated 100-image calibration subset
- selected weight: lambda 0.10
- frozen evaluation performed on the remaining 300 GLM development images

## Evaluation

The final cross-system comparison is performed through separate submissions on the competition test set used by the Zindi leaderboard. All five systems receive the same test IDs. Base and fine-tuned variants within each architecture use the same architecture-specific prompt and preprocessing.

Because test labels are not public, this repository cannot independently recompute official leaderboard scores or per-example test errors. Leaderboard results should be recorded with the submission identity, checkpoint, decoder, training steps, and training epochs produced by `notebooks/final_test_benchmark.ipynb`.

Local GLM and Qwen validation scores were produced on different development splits and must not be interpreted as a controlled comparison between model families.

## Intended use

- research and competition experiments on data the user is authorized to access
- comparison of multimodal OCR architectures and decoding strategies
- transcription assistance with human review
- controlled study of fine-tuning and character-language-model fusion

## Limitations

- historical spelling, abbreviations, names, superscripts, degradation, and crop geometry can cause severe errors
- the KenLM decoder can prefer common character sequences over rare names or faithful historical spelling
- results may not generalize to other archives, languages, page layouts, or scanning conditions
- the Qwen selected checkpoint comes from an interrupted training schedule
- model-specific prompts, tokenization, quantization, and image preprocessing prevent the comparison from isolating architecture alone
- Zindi leaderboard evaluation does not expose test labels or per-example errors
- repeated leaderboard submissions can indirectly influence model or decoder selection and should be documented
- development checkpoints were selected using development data, so their local scores are selection estimates rather than unbiased generalization estimates
- OCR output should not be treated as an authoritative archival transcription without human review

## Reproducibility and artifacts

The notebooks record or verify model revisions, checkpoint identities, prompts, preprocessing limits, decoding settings, and cached prediction identities. The final benchmark loads models sequentially and writes one Zindi-compatible submission per system plus a combined local `benchmark.csv` containing model and training metadata.

Exact reproduction requires authorized access to the Zindi data and local copies of the base models and selected adapters. Those artifacts are intentionally excluded from Git.

## Licenses

Repository code is MIT-licensed. Competition data, downloaded models, trained adapters, generated predictions, submissions, and third-party components are not covered by the repository license.

GLM-OCR, Qwen3-VL, KenLM, and their dependencies retain their respective upstream licenses and usage terms. Verify those terms before redistributing model weights or adapters.
