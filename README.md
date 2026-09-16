# GLM-OCR for the Barbados Historic Handwriting Challenge

Fine-tuning and inference experiments for the [R.O.A.D. Barbados Historic Handwriting Challenge](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge), built around GLM-OCR and a reproducible two-stage LoRA curriculum.

This repository publishes code only. Competition images, labels, submissions, downloaded model weights, adapters, and cached predictions are deliberately excluded.

## Current pipeline

The strongest tested workflow is:

1. Fine-tune GLM-OCR with the native OCR prompt and identical spatial preprocessing for training, validation, and inference.
2. Run Stage 1 on clean and mild augmented views.
3. Start Stage 2 from the best Stage 1 checkpoint, with additional emphasis on complex crops.
4. Save Stage 2 every 0.1 epoch and select by the fixed validation proxy.
5. Evaluate targeted 2048-token resolution for crops capped at the default 1536-token budget.
6. Optionally apply character-level KenLM shallow fusion selected on a calibration split.

The saved experiments produced the following local validation evidence. These values are development-set score proxies, not official leaderboard scores.

| Experiment | Evaluation scope | Local proxy | Gain |
|---|---:|---:|---:|
| Checkpoint 642, greedy at 1536 visual tokens | 400 images | 0.888709 | — |
| 2048 tokens only for capped crops | 400-image hybrid | 0.890397 | +0.001687 |
| Beam 4 baseline | 300-image KenLM evaluation split | 0.891962 | — |
| Beam 4 + character KenLM, lambda 0.10 | Same 300 images | 0.899150 | +0.007189 |

The newer Stage 2-from-best-Stage-1 run selected checkpoint 770 at epoch 0.30 with a 0.884229 proxy. Because that run and checkpoint 642 were produced under different selection histories, their scores should not be treated as a controlled comparison.

## Repository layout

```text
notebooks/
  glm_finetuning.ipynb
  glm_greedy_vs_beam4.ipynb
  glm_kenlm_decoding.ipynb
  glm_higher_resolution.ipynb
  base_models_benchmark.ipynb
  qwen3_vl_4b_finetuning.ipynb
src/
  barbados_ocr_augmentation.py
  transcription_metrics.py
tests/
resources/        # create locally; never commit competition files
model/glm-ocr/    # download locally; never commit model weights
training_outputs/ # generated locally; never commit checkpoints or predictions
```

The notebooks in `notebooks/` have no execution output or embedded dataset examples. The original working notebooks can remain in the repository root on your machine and are ignored by Git.

## Setup

The recorded training run used Linux, Python 3.11, CUDA, PyTorch 2.8.0+cu128, Transformers 5.8.0, PEFT 0.18.1, Albumentations 2.0.8, OpenCV 5.0.0, and NumPy 2.4.6. A CUDA GPU with bfloat16 support is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA driver if the default wheel is unsuitable. KenLM is optional and needed only for the KenLM notebook:

```bash
python -m pip install -r requirements-kenlm.txt
```

## Private assets

Download the competition data from Zindi after accepting its terms, then arrange it locally as:

```text
resources/
  Train.csv
  Test.csv
  SampleSubmission.csv
  images/
    <ID>.jpg
```

Download the `zai-org/GLM-OCR` base model into `model/glm-ocr/`. The notebooks intentionally use local files so the exact base checkpoint can be hashed and recorded in each run identity.

See [DATA.md](DATA.md) for the data boundary and [MODEL_CARD.md](MODEL_CARD.md) for model scope and limitations.

## Running the notebooks

Start Jupyter from the project root so all relative paths resolve correctly:

```bash
jupyter lab
```

Run the notebooks in this order:

1. `notebooks/glm_finetuning.ipynb`
2. `notebooks/glm_greedy_vs_beam4.ipynb`
3. `notebooks/glm_kenlm_decoding.ipynb` (optional)
4. `notebooks/glm_higher_resolution.ipynb` (optional)
5. `notebooks/base_models_benchmark.ipynb` (optional three-model benchmark)
6. `notebooks/qwen3_vl_4b_finetuning.ipynb` (independent Qwen curriculum run)

The benchmark notebook downloads commit-pinned Qwen3-VL-4B-Instruct and Qwen2.5-VL-7B-Instruct snapshots, then compares both zero-shot baselines with the untouched base GLM-OCR model on the exact saved 400-image validation split. Install its additional 4-bit inference dependencies with `requirements-benchmark.txt`. It caches every prediction and loads the models sequentially.

The Qwen curriculum notebook uses seed 1705 and its own stratified 400-image holdout, so it is independent of both the GLM split and the earlier seed-42 Qwen experiment. It starts Stage 2 from the best Stage 1 checkpoint and saves Stage 2 every 0.1 epoch. Install its dependencies with `requirements-qwen-training.txt`.

Training is not started by setup or preview cells. The two training cells in the curriculum notebook are explicitly separated. Run identities guard against accidentally resuming incompatible settings; use a new `run_name` for a genuinely different experiment.

## Lightweight verification

Repository checks do not train or load GLM-OCR:

```bash
python -m pip install -r requirements-dev.txt
pytest
python scripts/check_public_notebooks.py
```

## Licensing and attribution

Project code is released under the [MIT License](LICENSE). The license does not grant rights to the Zindi competition data, generated transcriptions derived from that data, downloaded models, or third-party software.

GLM-OCR code is Apache-2.0 and its model weights are MIT-licensed; consult the [official GLM-OCR repository](https://github.com/zai-org/GLM-OCR) for the current terms. KenLM and every other dependency retain their own licenses.

## Responsible publication

Before any push, inspect `git status` and confirm that no dataset rows, manuscript images, checkpoints, predictions, submissions, credentials, or notebook outputs are staged. If such a file was ever committed, adding it to `.gitignore` is not sufficient; remove it from Git history before publishing.
