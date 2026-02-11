# VBLL PyTorch Finetuning Experiments

PyTorch-based finetuning experiments using [VBLL](https://github.com/VectorInstitute/vbll) (Variational Bayesian Last Layers) with [scaling_mlps](https://github.com/gregorbachmann/scaling_mlps) pretrained models.

This module is integrated with the main `bllarse` MLflow/sweep flow.

## Setup

```bash
git submodule update --init --recursive scripts/vbll_pytorch/scaling_mlps
uv pip install -e scripts/vbll_pytorch
```

## Usage

Basic finetuning on CIFAR-10:
```bash
python scripts/vbll_pytorch/finetuning_vbll.py --dataset cifar10 --epochs 100
```

Full network finetuning with MLflow logging:
```bash
python scripts/vbll_pytorch/finetuning_vbll.py \
    --dataset cifar10 \
    --tune-mode full_network \
    --optimizer adamw \
    --learning-rate 1e-4 \
    --epochs 100 \
    --enable-mlflow
```

Using Lion optimizer:
```bash
uv pip install lion-pytorch
python scripts/vbll_pytorch/finetuning_vbll.py --optimizer lion --learning-rate 1e-4
```

Smoke run using the shared sweep tooling:
```bash
python src/bllarse/tools/run_config.py bllarse_sweeps/vbll_smoke_sweep.py 0
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | cifar10 | Dataset (cifar10, cifar100) |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--optimizer` | adamw | Optimizer (adamw, lion) |
| `--learning-rate` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | Weight decay |
| `--num-blocks` | 6 | MLP blocks (6, 12) |
| `--embed-dim` | 512 | Embedding dim (512, 1024) |
| `--pretrained` | in21k_cifar | Checkpoint (in21k, in21k_cifar) |
| `--tune-mode` | last_layer | last_layer or full_network |
| `--prior-scale` | 1.0 | VBLL prior scale |
| `--wishart-scale` | 0.1 | VBLL Wishart scale |
| `--nodataaug` | false | Disable data augmentation |
| `--enable-mlflow` | false | Enable MLflow tracking |

## Comparison with JAX Script

This script mirrors `scripts/finetuning.py` but:
- Uses PyTorch instead of JAX/Equinox
- Uses VBLL instead of IBProbit for Bayesian last layer
- Uses MLflow instead of W&B for experiment tracking
- Uses scaling_mlps PyTorch models instead of mlpox
