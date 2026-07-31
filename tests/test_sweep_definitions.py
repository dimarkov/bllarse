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


def test_every_probe_sweep_enables_mlflow():
    """Inverted from its original form. This test used to assert probe_smoke did
    NOT enable MLflow, because neither arm referenced it -- true until both arms
    grew a child_run and every sweep was switched on. It stays as the guard on
    the other direction: a sweep that quietly loses enable_mlflow would run
    ~720 unattended array tasks whose only signal is dump-file presence.

    probe_smoke keeps it too, so the cluster smoke test exercises the MLflow
    path (including cross-process parent attachment) before the real sweeps do.
    """
    for name in ("probe_smoke", "probe_adamw_ce", "probe_vbll",
                 "probe_ood_adamw_ce", "probe_ood_vbll"):
        mod = get_module_from_source_path(f"bllarse_sweeps/{name}.py")
        configs = mod.create_configs()
        assert configs, name
        for config in configs:
            assert config.get("enable_mlflow") is True, f"{name}: {config.get('uid')}"
            assert config.get("mlflow_experiment"), f"{name}: no experiment name"
