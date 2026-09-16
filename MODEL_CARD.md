# Model card

## Model

This project adapts `zai-org/GLM-OCR` for single-line transcription of historical handwritten records from Barbados. It trains a rank-16 LoRA adapter over selected language and vision components and includes a trainable multimodal projector.

No base weights or fine-tuned adapters are distributed in this repository.

## Intended use

- research and competition experiments on data the user is authorized to access
- transcription assistance with human review
- controlled comparison of image resolution and decoding strategies

## Limitations

- validation scores come from a development split used during model selection
- historical handwriting, spelling, abbreviations, degradation, and crop geometry can cause severe errors
- the KenLM decoder may prefer common character sequences over rare names or faithful historical spelling
- results may not generalize to other archives, languages, layouts, or scanning conditions
- OCR output should not be treated as an authoritative archival transcription without review

## Training summary

- native prompt: `Text Recognition:`
- maximum spatial budget: 1536 visual tokens during the main run
- Stage 1: clean plus mild augmented views
- Stage 2: initialized from the best Stage 1 checkpoint with complex-crop sampling
- checkpoint cadence: ten checkpoints per Stage 2 epoch
- selection metric: weighted word and character error combined into the project score proxy

See the curriculum notebook for the complete, executable configuration and run-identity safeguards.

## Licenses

Repository code is MIT-licensed. The upstream GLM-OCR repository states that its code is Apache-2.0 and its model weights are MIT-licensed. Competition data and third-party components are not covered by this repository's license.

