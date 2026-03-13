# RoBERTa -> MNLI linear probe note, checked against GLUE-X

This note is for the **head-only MNLI setup** only, not full-network fine-tuning.

I checked it against:
- the local paper PDF: `GLUEX_paperpdf.pdf`
- the paper repo: [YangLinyi/GLUE-X](https://github.com/YangLinyi/GLUE-X)
- this repo's implementation: `scripts/roberta_mnli.py` and the sweep files under `bllarse_sweeps/`

## 1) What GLUE-X means by "linear probing"

In the GLUE-X paper, the three tuning strategies are:
- standard fine-tuning
- fine-tuning only the head (**linear probing**)
- linear probing then fine-tuning (**LP-FT**)

For MNLI, GLUE-X uses:
- source / training task: **MNLI**
- ID selection/evaluation split: **`validation_matched`**
- OOD targets: **`validation_mismatched`**, **SNLI**, and **SICK**

Appendix B says they do a **grid search**, keep the **best-performing checkpoint on the ID dataset**, and then evaluate that checkpoint on the OOD datasets.

One thing the current note was missing: the paper's own Figure 2 says that for **RoBERTa-base on MNLI**, plain linear probing is **worse than** standard fine-tuning and LP-FT on both ID and OOD accuracy. So if you run only the probe head, do not expect it to match the full fine-tuning result from the paper.

## 2) What the public GLUE-X repo actually does for LP

The public torch code is more specific than the paper text.

For encoder models such as RoBERTa, the LP path does **not** train a single `d -> 3` layer directly on cached features. It does this:

1. Load the pretrained encoder with `AutoModel`.
2. Take the first-token hidden state `hidden[:, 0, :]`.
3. Pass it through a trainable head:
   `Linear(d, d) -> ReLU -> Dropout(0.1) -> Linear(d, num_labels)`.
4. Freeze only the pretrained backbone parameters (`l1.*`).
5. Train the head with cross-entropy and **Adam**.

The public LP defaults in `evaluation/utils_lp.py` are:
- `training_epoch_lp = 120`
- `lr_list = [2e-5, 3e-5]`
- `batch_size_per_device_list = [8, 4]`
- `max_len = 512`
- `dropout = 0.1`
- `random_seed = 2022`

Important detail: the optimizer helper freezes the backbone for LP and uses **Adam without weight decay**. There is no scheduler in the LP code path.

Another important detail: these `8/4` LP batch sizes come from the **public repo code**, not from an explicit statement in the paper PDF. The paper's Table 9 gives a single LR / batch-size pair per model-task, and for **RoBERTa-base on MNLI** it shows **`2e-05 / 32`**, but that table is not clearly labeled as LP-only. For LP reproduction, the repo defaults above are the safer source of truth.

## 3) What this repo currently implements

This repo's `scripts/roberta_mnli.py` is **GLUE-X-inspired**, but it is **not a byte-for-byte reproduction** of the public GLUE-X LP head.

What it does:
- extract frozen **final-layer CLS features** from `glue/mnli`
- cache features for `train`, `validation_matched`, and `validation_mismatched`
- train only a **linear softmax head** `xW + b` for deterministic baselines
- optionally apply dropout to the cached features before the linear map
- select the best checkpoint on **`validation_matched` accuracy**, with NLL and ECE tie-breaks
- log **ACC / NLL / ECE** on matched and mismatched

What it does **not** currently do:
- it does not reproduce the GLUE-X torch head `Linear(d,d) -> ReLU -> Dropout -> Linear(d,3)`
- it does not currently evaluate **SNLI** or **SICK**

So the right wording for the current local implementation is:

> **cached-feature, head-only MNLI baseline aligned with GLUE-X selection/eval conventions**

and not:

> exact reproduction of the public GLUE-X linear-probing code

## 4) Public GLUE-X repo LP defaults, adapted in this repo

If the goal is to follow the **public GLUE-X LP code defaults** inside this repo's cached-feature approximation, use:
- backbone: `FacebookAI/roberta-base`
- optimizer: `adam`
- learning rate: `{2e-5, 3e-5}`
- train batch size: `{4, 8}`
- epochs: `120`
- dropout rate: `0.1`
- seed: `2022`
- max length: `512`
- train only the probe head
- select best epoch by `validation_matched` accuracy
- report at least `validation_matched` and `validation_mismatched`

One local gotcha: `scripts/roberta_mnli.py` defaults to `--max-length 256`, but the GLUE-X-aligned sweep files override that to **512**. Use the sweep values, not the script default, if you want GLUE-X alignment.

## 5) What to run in this repo

### Smoke check

```bash
python src/bllarse/tools/run_config.py bllarse_sweeps/mnli_roberta_gluex_linear_probe_smoke.py 0
```

### Full GLUE-X-style sweep

```bash
python src/bllarse/tools/run_sweep.py \
  bllarse_sweeps/mnli_roberta_gluex_linear_probe_sweep.py \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

### Single direct run

```bash
python scripts/roberta_mnli.py \
  --stage all \
  --backbone FacebookAI/roberta-base \
  --reuse-cache \
  --optimizer adam \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --epochs 120 \
  --dropout-rate 0.1 \
  --seed 2022 \
  --max-length 512
```

If the goal is "linear probe only", avoid:
- `scripts/finetuning.py`
- any full-network fine-tuning path
- `--optimizer cavi`

`adamw` is reasonable as an additional baseline in this repo, but it is an **extension beyond literal GLUE-X**, not the exact public LP setup.

## 6) If you want an exact GLUE-X LP reproduction later

To match the public GLUE-X torch implementation more closely, the local MNLI probe path would need two changes:

1. Replace the cached-feature `xW + b` head with the GLUE-X trainable head:
   `Linear(d,d) -> ReLU -> Dropout -> Linear(d,3)`.
2. Add the remaining GLUE-X NLI OOD evaluations:
   **SNLI** and **SICK**, in addition to `validation_mismatched`.

Until then, the current local path should be described as a **faithful head-only baseline using GLUE-X hyperparameters and selection rules**, but not as the exact GLUE-X codepath.

## 7) Source pointers

- Paper:
  [GLUE-X PDF in this repo](./GLUEX_paperpdf.pdf)
- Public repo:
  [evaluation/utils_lp.py](https://github.com/YangLinyi/GLUE-X/blob/main/evaluation/utils_lp.py)
- Public repo:
  [evaluation/eval_multi_class.py](https://github.com/YangLinyi/GLUE-X/blob/main/evaluation/eval_multi_class.py)
- Public repo:
  [evaluation/top_OOD_lp.py](https://github.com/YangLinyi/GLUE-X/blob/main/evaluation/top_OOD_lp.py)

## 8) Current cluster recipe in this repo

### Canonical feature extraction

For the current local pipeline, the recommended extraction job is:

- backbone: `FacebookAI/roberta-base`
- max length: `512`
- cache dtype: `float16`
- extract batch size: `256`
- resources: `1x A100 40GB`, `8 CPUs`, `BLLARSE_DOCKER_SHM_SIZE=16g`
- HF sync: `pull_push`
- MLflow: disabled for extraction jobs

The canonical sweep file is:

- `bllarse_sweeps/mnli_roberta_len512_extract.py`

Launch it from the login node with:

```bash
source .venv_bllarse_new/bin/activate
export HF_TOKEN="<token>"
export BLLARSE_DOCKER_SHM_SIZE=16g

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/mnli_roberta_len512_extract.py \
  --venv .venv_bllarse_new \
  --max-concurrent 1 \
  --cpus-per-task 8 \
  --job-name mnli_len512_extract_bs256 \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

The expected HF destination is:

```text
roberta_activations/mnli_roberta_cls/FacebookAI__roberta_base_len512_float16_5c31d846204b
```

Important operational detail:

- the canonical cache key does **not** encode extraction batch size
- only backbone / max length / dtype / sample caps affect the cache key
- so benchmarking different batch sizes should use sample-capped jobs with `hf_sync=none`

### Len256 Adam / AdamW baseline training

For the current internal deterministic baseline, we reuse the existing full len256 HF cache and train the cached-feature linear softmax head.

Quick 5-epoch sanity sweep:

- `bllarse_sweeps/mnli_roberta_len256_linear_probe_baseline.py`

```bash
python src/bllarse/tools/run_config.py \
  bllarse_sweeps/mnli_roberta_len256_linear_probe_baseline.py 0
```

Full multiseed sweep:

- `bllarse_sweeps/mnli_roberta_len256_linear_probe_multiseed.py`

```bash
source .venv_bllarse_new/bin/activate
export MLFLOW_TRACKING_URI="https://mlflow.markov.icu"
export BLLARSE_DOCKER_SHM_SIZE=16g

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/mnli_roberta_len256_linear_probe_multiseed.py \
  --venv .venv_bllarse_new \
  --max-concurrent 24 \
  --job-name mnli_len256_multiseed \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

That sweep currently uses:

- optimizers: `adam`, `adamw`
- learning rates: `2e-5`, `3e-5`
- batch sizes: `8`, `4`
- seeds: `2022`, `2023`, `2024`
- epochs: `120`
- dropout rate: `0.1`
- `weight_decay = 0.0` for `adam`
- `weight_decay = 1e-2` for `adamw`

Per-epoch `acc`, `nll`, and `ece` are logged to MLflow for these `train_eval` jobs.
