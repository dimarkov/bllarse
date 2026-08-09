"""The fine-tuning worker's config assembly and study identity.

Everything here runs without a GPU: what it checks is that a trial's config is
the one the design specifies, because a wrong value costs ~6.3 A100-hours per
trial and is invisible in the result.
"""

import pytest

from bllarse.finetune import protocol
from bllarse.tuning import finetune_space as space
from bllarse.tuning import finetune_worker as fw


@pytest.fixture(autouse=True)
def pinned_micro_batch(monkeypatch):
    # MICRO_BATCH ships empty on purpose (it is measured per GPU), so every
    # config test pins it here rather than depending on a checked-in value.
    monkeypatch.setattr(space, "MICRO_BATCH", {"ce": 64, "ibprobit": 32, "vbll": 64})


def _params(arm):
    base = {"batch_size": 1024, "lr": 1e-4, "weight_decay": 1e-3}
    if arm.startswith("ibprobit"):
        base |= {"q_init": 1e-4, "num_update_iters": 192}
        if arm == "ibprobit":
            base["eps_init"] = 1e-3
    else:
        base["head_lr_ratio"] = 100.0
        if arm == "vbll":
            base |= {"kl_ratio": 1e-6, "prior_scale": 10.0}
    return base


# --- The protocol constants every arm must carry ----------------------------


@pytest.mark.parametrize("arm", fw.SCRIPTS)
def test_every_arm_carries_the_tuning_protocol(arm):
    config = fw.trial_config(arm, params=_params(arm))
    assert config["max_samples"] == 2_000_000
    assert config["eval_quantum"] == 16_384
    assert config["img_size"] == 224
    assert config["model"] == "dinov3_large"
    assert config["val_fraction"] == 0.2
    assert config["bf16"] is True


@pytest.mark.parametrize("arm", fw.SCRIPTS)
def test_tuning_evaluates_val_only_and_dumps_only_the_argmin(arm):
    # Forwarding the 30k test partition at all 122 evaluations would roughly
    # double the campaign for a number nothing reads, and 122 trajectory dumps
    # would be 4.9 GB per run.
    config = fw.trial_config(arm, params=_params(arm))
    assert config["eval_splits"] == "val"
    assert config["dump_every_k_evals"] == 0
    assert config["ood_every_k_dumps"] == 0


@pytest.mark.parametrize("arm", fw.SCRIPTS)
def test_the_micro_batch_divides_the_sampled_batch(arm):
    # protocol.accum_plan raises otherwise, at the top of a ~6 h trial.
    config = fw.trial_config(arm, params=_params(arm))
    assert config["batch_size"] % config["micro_batch_size"] == 0
    protocol.accum_plan(config["batch_size"], config["micro_batch_size"])


def test_an_unmeasured_micro_batch_fails_at_config_time(monkeypatch):
    # Loudly, at submit time -- not three hours into the first trial.
    monkeypatch.setattr(space, "MICRO_BATCH", {})
    with pytest.raises(KeyError, match="probe_micro_batch"):
        fw.trial_config("ce", params=_params("ce"))


# --- Arm -> script flag -----------------------------------------------------


def test_the_noeps_study_maps_to_the_ibprobit_flag_with_eps_pinned():
    # finetune_jax's argparser accepts only {ce, ibprobit}; passing the study
    # name through would abort every trial at argparse.
    config = fw.trial_config("ibprobit_noeps", params=_params("ibprobit_noeps"))
    assert config["arm"] == "ibprobit"
    assert config["eps_init"] == space.IBPROBIT_NOEPS_INIT


def test_the_searched_ibprobit_arm_keeps_its_sampled_eps():
    config = fw.trial_config("ibprobit", params=_params("ibprobit"))
    assert config["arm"] == "ibprobit"
    assert config["eps_init"] == 1e-3


def test_both_jax_arms_and_vbll_route_to_their_own_scripts():
    assert fw.SCRIPTS["ce"] == fw.SCRIPTS["ibprobit"] == fw.SCRIPTS["ibprobit_noeps"]
    assert fw.SCRIPTS["vbll"] != fw.SCRIPTS["ce"]


def test_arms_match_the_search_space():
    # An arm in one table and not the other gives a worker that accepts --arm
    # and then dies inside its first trial.
    assert set(fw.SCRIPTS) == set(space.ARMS)


# --- Head learning rate -----------------------------------------------------


@pytest.mark.parametrize("arm", ["ce", "vbll"])
def test_head_lr_is_the_trunk_rate_times_the_sampled_ratio(arm):
    config = fw.trial_config(arm, params=_params(arm))
    assert config["head_lr"] == pytest.approx(1e-4 * 100.0)
    # The ratio itself is not a script flag.
    assert "head_lr_ratio" not in config


@pytest.mark.parametrize("arm", ["ibprobit", "ibprobit_noeps"])
def test_neither_ibprobit_arm_gets_a_head_lr(arm):
    # finetune_jax.check_arm_flags rejects --head-lr for this arm outright:
    # the head is assimilated in closed form and has no optimizer.
    config = fw.trial_config(arm, params=_params(arm))
    assert "head_lr" not in config
    assert "head_lr_ratio" not in config


# --- Fixed knobs ------------------------------------------------------------


@pytest.mark.parametrize("arm", ["ibprobit", "ibprobit_noeps"])
def test_ibprobit_runs_px_with_a_static_q(arm):
    config = fw.trial_config(arm, params=_params(arm))
    assert config["px"] is True
    assert config["no_learn_q"] is True


def test_vbll_pins_wishart_scale_rather_than_searching_it():
    config = fw.trial_config("vbll", params=_params("vbll"))
    assert config["wishart_scale"] == space.WISHART_SCALE
    assert config["parameterization"] == "diagonal"


def test_trial_config_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        fw.trial_config("ivon", params={})


# --- Study identity and the pruner ------------------------------------------


def test_study_name_is_namespaced_by_campaign():
    # open_study loads by name, so an un-suffixed name would RESUME a study
    # whose search space no longer exists.
    assert fw.study_name("ce") == "ft_ce"
    assert fw.study_name("ce", "imagenet_ft") == "ft_ce_imagenet_ft"
    assert fw.study_name("ibprobit_noeps", "v2") == "ft_ibprobit_noeps_v2"


def test_pruner_engages_far_earlier_than_the_probing_one():
    # 20 was a fifth of ~80 probing trials; at ~30 trials it would barely
    # engage, and pruning is where the saving is at ~6 h a trial.
    assert fw.PRUNER_STARTUP_TRIALS == 5
    assert fw.PRUNER_WARMUP_EVALS == 10


def test_patience_is_on_and_can_actually_fire():
    # The probing campaign's `--patience 20` needed the argmin at dump <= 19 on
    # a bounded 40-point geometric grid. Here there are 122 uniformly
    # sample-spaced evaluations and the earliest stop is eval 30, so the
    # mechanism has room to work.
    assert fw.PATIENCE == 20
    assert fw.WARMUP_EVALS + fw.PATIENCE < space.eval_count()
