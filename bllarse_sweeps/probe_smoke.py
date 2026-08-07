"""One config per arm and per figure, for shaking out the cluster path.

Not a results sweep. Exercises: Apptainer launch, dump paths, both the Fig 1
(no OOD) and Fig 2 (with OOD) code paths, both arms, and -- since both now set
`enable_mlflow` -- the full MLflow path including run_config's nesting gate and
the parent/child attachment across array tasks.
"""

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_smoke"

_MLFLOW = dict(enable_mlflow=True, mlflow_experiment="bllarse-probing")

_JAX = dict(**_MLFLOW, arm="adamw_ce", epochs=4, n_dumps=3, batch_size=512,
            learning_rate=1e-3, weight_decay=1e-4, val_fraction=0.1,
            model="dinov3_small", seed=137, group_id=GROUP_ID)
_TORCH = dict(**_MLFLOW, epochs=4, n_dumps=3, batch_size=512,
              learning_rate=1e-3, weight_decay=1e-4, val_fraction=0.1,
              parameterization="diagonal",
              vbll_type="discriminative", device="cuda",
              model="dinov3_small", seed=137, group_id=GROUP_ID)

CONFIGS = [
    dict(**_JAX, dataset="cifar10", num_classes=10, img_size=224, uid="jax_fig1"),
    dict(**_JAX, dataset="cifar10", num_classes=10, img_size=224,
         ood_dataset="svhn", uid="jax_fig2"),
    dict(**_TORCH, dataset="cifar10", num_classes=10, img_size=224, uid="torch_fig1"),
    dict(**_TORCH, dataset="cifar10", num_classes=10, img_size=224,
         ood_dataset="svhn", uid="torch_fig2"),
]


def create_configs():
    return CONFIGS


def run(config):
    script = ("scripts/vbll_pytorch/probing_vbll.py"
              if "arm" not in config else "scripts/dinov3/probe.py")
    run_script_from_config(script, config)
