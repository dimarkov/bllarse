import pytest

from bllarse.tools.module import get_module_from_source_path

SWEEPS = [
    "bllarse_sweeps/probe_adamw_ce.py",
    "bllarse_sweeps/probe_vbll.py",
    "bllarse_sweeps/probe_ood_adamw_ce.py",
    "bllarse_sweeps/probe_ood_vbll.py",
]

LADDER = {"dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"}


@pytest.mark.parametrize("path", SWEEPS)
def test_sweep_exposes_the_expected_contract(path):
    mod = get_module_from_source_path(path)
    assert callable(mod.create_configs)
    assert callable(mod.run)


@pytest.mark.parametrize("path", SWEEPS)
def test_configs_are_unique_and_well_formed(path):
    configs = get_module_from_source_path(path).create_configs()
    assert len(configs) > 0
    uids = [c["uid"] for c in configs]
    assert len(uids) == len(set(uids)), "uids must be unique -- they key the dumps"
    for c in configs:
        assert c["model"] in LADDER, "dinov3_max is out of scope"
        assert "group_id" in c and "seed" in c


@pytest.mark.parametrize("path", SWEEPS)
def test_all_configs_share_one_group_id(path):
    configs = get_module_from_source_path(path).create_configs()
    assert len({c["group_id"] for c in configs}) == 1


def test_fig1_sweep_sizes_match_the_plan():
    # 4 backbones x 6 datasets x 10 seeds
    assert len(get_module_from_source_path(SWEEPS[0]).create_configs()) == 240


@pytest.mark.parametrize("path", [SWEEPS[2], SWEEPS[3]])
def test_fig2_sweep_sizes_match_the_plan(path):
    # 4 backbones x 3 (ID, OOD) pairs x 10 seeds.
    # Three pairs, not two ID datasets: CIFAR-10 appears twice, against SVHN
    # (far-OOD) and against CIFAR-100 (near-OOD).
    assert len(get_module_from_source_path(path).create_configs()) == 120


@pytest.mark.parametrize("path", [SWEEPS[2], SWEEPS[3]])
def test_fig2_sweeps_pair_each_id_dataset_with_an_ood_dataset(path):
    for c in get_module_from_source_path(path).create_configs():
        assert c["ood_dataset"] in {"svhn", "cifar100"}
        assert c["ood_dataset"] != c["dataset"]


def test_probe_smoke_does_not_claim_or_enable_mlflow():
    """probe_smoke.py's docstring used to claim it exercises "MLflow
    nesting", but neither scripts/alternatives/probe.py nor
    scripts/vbll_pytorch/probing_vbll.py reference MLflow, and
    run_config._config_uses_mlflow only fires when a config sets
    enable_mlflow=True -- which none of probe_smoke's configs do. Guard both
    facts: the docstring must not repeat the stale claim, and no config here
    may silently start requesting a gate that was never wired through."""
    mod = get_module_from_source_path("bllarse_sweeps/probe_smoke.py")
    doc = (mod.__doc__ or "").lower()
    # The exact stale claim this docstring used to make. Mentioning MLflow to
    # explain why it is NOT exercised is fine (and present); claiming it IS
    # exercised, as part of an "Exercises: ..." list, is the regression.
    assert "exercises: apptainer launch, mlflow nesting" not in doc
    for c in mod.create_configs():
        assert not c.get("enable_mlflow", False)
