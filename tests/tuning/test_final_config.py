"""Turning a finished study into the final campaign's command.

The defect this guards is not a crash. A config transferred wrongly -- or not
transferred at all -- runs to completion and produces figures; it just answers
a question about a different operating point than the one reported. So most of
these assert on the arithmetic reaching the emitted FLAGS, not on the helper
returning a plausible dict.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

from bllarse.finetune import protocol  # noqa: E402
from bllarse.tools import adapters  # noqa: E402
from bllarse.tuning import final_config as fc  # noqa: E402
from bllarse.tuning import finetune_space as fs  # noqa: E402
from bllarse.tuning import finetune_worker  # noqa: E402

ROOT = math.sqrt(512)
# The fixb512 campaign's ce winner, its budget, and the factor it takes. The
# budget is a LITERAL: these trials were run at 2M and stay run at 2M whatever
# protocol.MAX_SAMPLES becomes for the next campaign.
TUNE_SAMPLES = 2_000_000
CE_BEST = {"lr": 1.17e-5, "weight_decay": 8.13e-6, "head_lr_ratio": 31.7}
DURATION_FACTOR = math.sqrt(TUNE_SAMPLES / 5_124_668)


class FakeTrial:
    def __init__(self, params, **user_attrs):
        self.params = dict(params)
        self.user_attrs = dict(user_attrs)
        self.number = 19


class FakeStudy:
    def __init__(self, trial, value=0.44426):
        self.best_trial = trial
        self.best_value = value


# --- absolute_params --------------------------------------------------------


def test_it_undoes_the_per_sqrt_batch_reparameterisation():
    out = fc.absolute_params(
        {"lr_per_sqrt_batch": 1e-6, "weight_decay_per_sqrt_batch": 1e-7},
        effective_batch=512)
    assert out == {"lr": pytest.approx(1e-6 * ROOT),
                   "weight_decay": pytest.approx(1e-7 * ROOT)}


def test_it_passes_through_a_study_that_never_had_the_rename():
    """Not defensive coding -- a requirement.

    Both campaigns that exist (ft_*_shards2m and ft_*_fixb512) searched `lr`
    directly, and they are the studies this gets pointed at first. A helper
    that only understood the new names would be useless on every study we have.
    """
    assert fc.absolute_params(CE_BEST, effective_batch=512) == CE_BEST


def test_it_refuses_a_study_carrying_both_spellings():
    with pytest.raises(ValueError, match="ambiguous"):
        fc.absolute_params({"lr": 1e-5, "lr_per_sqrt_batch": 1e-6},
                           effective_batch=512)


def test_the_rename_is_not_spelled_twice():
    # A second independent spelling of the suffix drifts the moment one moves.
    assert fc.SEARCH_SUFFIX is fs.PER_SQRT_BATCH_SUFFIX


# --- final_config -----------------------------------------------------------


def _ce_config(**kwargs):
    kwargs.setdefault("from_samples", TUNE_SAMPLES)
    return fc.final_config("ce", CE_BEST, effective_batch=512, **kwargs)


def test_the_learning_rate_takes_the_duration_factor():
    assert DURATION_FACTOR == pytest.approx(0.6247, abs=1e-4)
    assert _ce_config()["lr"] == pytest.approx(1.17e-5 * DURATION_FACTOR)
    assert _ce_config()["weight_decay"] == pytest.approx(8.13e-6 * DURATION_FACTOR)


def test_the_head_rate_is_formed_from_the_TRANSFERRED_trunk_rate():
    """The ordering that matters, and the easy one to get backwards.

    head_lr_ratio is invariant, so the head rate has to be the ratio times the
    rate AFTER transfer. Multiplying by the pre-transfer rate would leave the
    head 1.6x too fast while every other number looked right.
    """
    config = _ce_config()
    assert config["head_lr"] == pytest.approx(config["lr"] * 31.7)


def test_the_budget_swaps_from_samples_to_epochs():
    # Both scripts take these as alternatives and --max-samples wins when
    # nonzero, so it has to be CLEARED rather than merely joined by --epochs.
    config = _ce_config()
    assert config["epochs"] == 4.0
    assert config["max_samples"] == 0


def test_the_betas_are_the_final_duration_not_the_tuning_one():
    config = _ce_config()
    assert config["beta1"] == pytest.approx(0.96097, abs=1e-5)
    assert config["beta2"] == pytest.approx(0.99961, abs=1e-5)
    assert (config["beta1"], config["beta2"]) != protocol.REFERENCE_BETAS


def test_total_decoupled_shrinkage_survives_the_assembly():
    # The same invariant test_transfer checks on the rule, re-checked at the
    # level that actually ships: trial_config sits between the two and could
    # overwrite either factor.
    config = _ce_config()
    before = (protocol.steps_for_samples(TUNE_SAMPLES, 512)
              * CE_BEST["lr"] * CE_BEST["weight_decay"])
    after = (protocol.steps_for_samples(5_124_668, 512)
             * config["lr"] * config["weight_decay"])
    assert after == pytest.approx(before, rel=1e-3)


def test_the_final_run_evaluates_the_test_split():
    # Tuning forwards `val` alone -- the objective is val_nll and forwarding
    # the 30k test partition 122 times would roughly double the campaign. The
    # final run is what the figures are made from.
    assert _ce_config()["eval_splits"] == "val+test"


def test_the_backbone_is_not_silently_corrected_for_width():
    """The transplant stays visible.

    Tuning runs dinov3_large and the paper reports dinov3_huge. CompleteP's
    width rule would say lr x m_N^-1 = x0.8, but it is unvalidated for
    fine-tuning a pretrained backbone and its depth companion cannot be applied
    to one at all. Applying half a rule silently would be worse than applying
    none, so the default carries the tuning model through and --model is how
    the transplant gets stated.
    """
    assert _ce_config()["model"] == fs.TUNING_MODEL == "dinov3_large"
    assert _ce_config(model="dinov3_huge")["model"] == "dinov3_huge"


def test_a_batch_change_is_refused_on_the_ibprobit_arms():
    params = {"lr": 2.62e-5, "weight_decay": 3.30e-5, "q_init": 3.72e-5,
              "num_update_iters": 512}
    # Duration-only is fine, which is the case that actually runs.
    assert fc.final_config("ibprobit_noeps", params, effective_batch=512,
                           from_samples=TUNE_SAMPLES)
    with pytest.raises(ValueError, match="q_init"):
        fc.final_config("ibprobit_noeps", params, effective_batch=512,
                        from_samples=TUNE_SAMPLES, to_batch=1024)


def test_a_batch_change_applies_the_square_root_rule_on_the_other_arms():
    config = _ce_config(to_batch=2048)
    # m_B = 4, m_D = 5124668/2000000: sqrt(4 / 2.562) = 1.2494.
    assert config["batch_size"] == 2048
    assert config["lr"] == pytest.approx(1.17e-5 * math.sqrt(4) * DURATION_FACTOR)


# --- best_config ------------------------------------------------------------


def test_it_reads_the_batch_off_the_user_attr_not_the_params():
    # batch_size is pinned and therefore no longer a param at all; the worker
    # has always written the effective batch as a user attr.
    study = FakeStudy(FakeTrial(CE_BEST, effective_batch=512,
                                max_samples=TUNE_SAMPLES))
    assert fc.best_config(study, "ce")["lr"] == pytest.approx(
        1.17e-5 * DURATION_FACTOR)


def test_it_honours_the_trials_own_tuning_budget():
    """A study run at a shortened budget must transfer from THAT budget.

    max_samples is a per-trial user attr precisely because it is overridable
    (--max-samples, for smoke runs). Assuming protocol.MAX_SAMPLES would
    mis-scale every config from a study that used anything else.
    """
    study = FakeStudy(FakeTrial(CE_BEST, effective_batch=512,
                                max_samples=500_000))
    config = fc.best_config(study, "ce")
    assert config["lr"] == pytest.approx(
        1.17e-5 * math.sqrt(500_000 / 5_124_668))


def test_a_study_recording_the_new_coordinate_round_trips():
    tuned = {"lr_per_sqrt_batch": 1.17e-5 / ROOT,
             "weight_decay_per_sqrt_batch": 8.13e-6 / ROOT,
             "head_lr_ratio": 31.7}
    study = FakeStudy(FakeTrial(tuned, effective_batch=512,
                                max_samples=TUNE_SAMPLES))
    assert fc.best_config(study, "ce")["lr"] == pytest.approx(
        1.17e-5 * DURATION_FACTOR)


# --- The emitted command ----------------------------------------------------


@pytest.mark.parametrize("arm", ["ce", "ibprobit_noeps"])
def test_the_emitted_flags_are_accepted_by_the_script_that_runs_them(arm):
    """The check that a dict-level test cannot make.

    Every value here reaches the training loop as a command-line flag, so a
    config key with no matching argument is a campaign that dies at argparse
    after the submission has taken a node.
    """
    import finetune_jax as fj

    params = dict(CE_BEST)
    if arm.startswith("ibprobit"):
        params = {"lr": 2.62e-5, "weight_decay": 3.30e-5,
                  "q_init": 3.72e-5, "num_update_iters": 512}
    config = fc.final_config(arm, params, effective_batch=512,
                             from_samples=TUNE_SAMPLES)
    argv = adapters.config_to_argv(config)
    args = fj.build_argparser().parse_args(argv)

    assert args.beta1 == pytest.approx(config["beta1"])
    assert args.beta2 == pytest.approx(config["beta2"])
    assert args.lr == pytest.approx(config["lr"])
    assert args.epochs == 4.0
    assert args.max_samples == 0


def test_the_vbll_script_takes_the_same_two_beta_flags():
    # Parsed rather than grepped, so a flag that exists but is spelled
    # differently (--beta-1) still fails. No torch needed: finetune_vbll
    # imports it inside the functions that use it, precisely so its contract
    # can be checked on a machine that only has the JAX venv.
    import finetune_vbll as fv

    params = {"lr": 1.47e-5, "weight_decay": 1.38e-3, "head_lr_ratio": 326.0,
              "kl_ratio": 3.06e-7, "prior_scale": 1.66e-3}
    config = fc.final_config("vbll", params, effective_batch=512,
                             from_samples=TUNE_SAMPLES)
    args = fv.build_argparser().parse_args(adapters.config_to_argv(config))
    assert args.beta1 == pytest.approx(config["beta1"])
    assert args.beta2 == pytest.approx(config["beta2"])
    assert args.lr == pytest.approx(1.47e-5 * DURATION_FACTOR)


# --- Searched betas ---------------------------------------------------------


def test_a_trial_that_TUNED_its_betas_transfers_them_not_the_defaults():
    """The gap that let a real bug through.

    Every other test here uses a fixb512 trial, which recorded no betas because
    the axis did not exist yet -- so the fallback path was exercised everywhere
    and the searched path nowhere. final_config's own consistency check was
    comparing against the transfer of REFERENCE_BETAS rather than of the
    trial's values, and rejected correct configs from any study that tuned
    them.
    """
    tune_steps = protocol.steps_for_samples(TUNE_SAMPLES, 512)
    final_steps = protocol.steps_for_samples(5_124_668, 512)
    h1, h2 = 120.0, 12.0
    params = {
        **CE_BEST,
        "beta1": protocol.beta_from_horizons(h1, tune_steps),
        "beta2": protocol.beta_from_horizons(h2, tune_steps),
    }
    config = fc.final_config("ce", params, effective_batch=512,
                             from_samples=TUNE_SAMPLES)
    # The horizon survives; the beta does not, and must not.
    assert protocol.horizons(config["beta1"], final_steps) == pytest.approx(h1)
    assert protocol.horizons(config["beta2"], final_steps) == pytest.approx(h2)
    assert config["beta1"] != pytest.approx(params["beta1"])
    # ... and it is emphatically not the transfer of the library defaults.
    assert config["beta1"] != pytest.approx(protocol.betas_for(final_steps)[0])


def test_the_whole_search_space_survives_the_round_trip():
    """suggest -> final_config, on every arm, with the real search space.

    The end-to-end path: a drawn trial's horizons must equal the horizons its
    final-run command implies. Nothing else in this file exercises suggest().
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    tune_steps = protocol.steps_for_samples(protocol.MAX_SAMPLES,
                                            fs.FIXED_BATCH_SIZE)
    final_steps = protocol.steps_for_samples(5_124_668, fs.FIXED_BATCH_SIZE)

    for arm in fs.ARMS:
        params = fs.suggest(study.ask(), arm)
        config = fc.final_config(arm, params,
                                 effective_batch=fs.FIXED_BATCH_SIZE)
        for name in ("beta1", "beta2"):
            assert protocol.horizons(config[name], final_steps) == pytest.approx(
                protocol.horizons(params[name], tune_steps)), (arm, name)
        # The learning rate takes the duration factor over the SAME budgets.
        assert config["lr"] == pytest.approx(
            params["lr"] * math.sqrt(protocol.MAX_SAMPLES / 5_124_668))


# --- MLflow reaches both campaigns ------------------------------------------


def test_the_final_run_logs_to_its_own_experiment():
    """One integration, two campaigns -- which was the requirement.

    Both the tuning trials and the reported 4-epoch runs reach MLflow through
    the same --enable-mlflow flag on the same training scripts, because both
    render a config dict into that script's argv. Only the experiment differs,
    so a search trial and a reported run are never averaged together.
    """
    config = _ce_config()
    assert config["enable_mlflow"] is True
    assert config["mlflow_experiment"] == "bllarse-finetuning"
    assert fc.MLFLOW_EXPERIMENT != finetune_worker.MLFLOW_EXPERIMENT


def test_the_experiment_names_mirror_the_probing_conventions():
    # bllarse-probing / bllarse-probing-tuning already exist on the server, so
    # the four read as two campaigns x two phases rather than four names.
    assert fc.MLFLOW_EXPERIMENT == "bllarse-finetuning"
    assert finetune_worker.MLFLOW_EXPERIMENT == "bllarse-finetuning-tuning"


@pytest.mark.parametrize("arm", ["ce", "ibprobit_noeps"])
def test_the_mlflow_flags_are_accepted_by_the_script(arm):
    import finetune_jax as fj

    params = dict(CE_BEST)
    if arm.startswith("ibprobit"):
        params = {"lr": 2.62e-5, "weight_decay": 3.30e-5,
                  "q_init": 3.72e-5, "num_update_iters": 512}
    config = fc.final_config(arm, params, effective_batch=512,
                             from_samples=TUNE_SAMPLES)
    args = fj.build_argparser().parse_args(adapters.config_to_argv(config))
    assert args.enable_mlflow is True
    assert args.mlflow_experiment == "bllarse-finetuning"


def test_the_vbll_script_takes_them_too():
    import finetune_vbll as fv

    params = {"lr": 1.47e-5, "weight_decay": 1.38e-3, "head_lr_ratio": 326.0,
              "kl_ratio": 3.06e-7, "prior_scale": 1.66e-3}
    config = fc.final_config("vbll", params, effective_batch=512,
                             from_samples=TUNE_SAMPLES)
    args = fv.build_argparser().parse_args(adapters.config_to_argv(config))
    assert args.enable_mlflow is True
    assert args.mlflow_experiment == "bllarse-finetuning"


def test_tracking_can_be_turned_off_for_a_local_run():
    # config_to_argv omits a False store_true flag entirely, so the script
    # defaults to off and a laptop run never reaches for the server.
    config = _ce_config(enable_mlflow=False)
    assert "--enable-mlflow" not in adapters.config_to_argv(config)


# --- The final campaign's per-seed identity ---------------------------------


def test_it_emits_a_seed_so_the_five_runs_are_not_identical():
    """Nothing upstream supplies one.

    trial_config never sets a seed -- objective.evaluate passes it for tuning
    trials -- and the final runs have no such wrapper. Without this every seed
    of every arm ran at the scripts' default 137: five identical runs reported
    as an error bar.
    """
    assert _ce_config(seed=141)["seed"] == 141
    assert _ce_config()["seed"] == fs.REPORTED_SEEDS[0]
    assert "--seed" in adapters.config_to_argv(_ce_config(seed=141))


def test_it_emits_one_group_for_the_whole_final_campaign():
    # The uid already carries model/dataset/arm/seed, so a single group puts
    # every reported run in one dump directory -- which is what Fig 5 reads.
    assert _ce_config()["group_id"] == fc.FINAL_GROUP_ID == "ftfinal"


@pytest.mark.parametrize("arm,mod_name", [
    ("ce", "finetune_jax"), ("ibprobit_noeps", "finetune_jax"),
    ("vbll", "finetune_vbll"),
])
def test_every_arm_and_seed_gets_a_distinct_run_name(arm, mod_name):
    """The property MLflow actually needs, checked through the real parsers.

    Run names collided during tuning because trial numbers restart per study --
    63 names were shared across arms, and matching on one grafted another arm's
    curve onto VBLL. The final campaign's uid is keyed on arm and seed, so this
    pins that it stays that way in BOTH frameworks.
    """
    import importlib
    mod = importlib.import_module(mod_name)
    params = {"ce": CE_BEST,
              "ibprobit_noeps": {"lr": 2.6e-5, "weight_decay": 3.3e-5,
                                 "q_init": 3.7e-5, "num_update_iters": 512},
              "vbll": {"lr": 1.5e-5, "weight_decay": 1.4e-3,
                       "head_lr_ratio": 326.0, "kl_ratio": 3e-7,
                       "prior_scale": 1.7e-3}}[arm]
    names = set()
    for seed in fs.REPORTED_SEEDS[:5]:
        config = fc.final_config(arm, params, effective_batch=512,
                                 from_samples=TUNE_SAMPLES,
                                 model="dinov3_huge", seed=seed)
        args = mod.build_argparser().parse_args(adapters.config_to_argv(config))
        names.add(mod.resolve_uid(args))
    assert len(names) == 5, names
    assert all("dinov3_huge" in n and "seed" in n for n in names)


# --- The searched betas must survive to the final run -----------------------


def _study_shaped(**over):
    """A study's params as OPTUNA RECORDS THEM.

    This is the shape every earlier test got wrong. `fs.suggest` RETURNS
    `beta1`/`beta2`, but what a trial stores is `beta1_horizons` -- so tests
    built from suggest()'s return value, or from hand-written `beta1` keys,
    exercised a shape no real study has.
    """
    p = {"lr_per_sqrt_batch": 1.25e-06, "weight_decay_per_sqrt_batch": 2.2e-06,
         "beta1_horizons": 72.59376434, "beta2_horizons": 32.37392373}
    p.update(over)
    return p


def test_the_searched_betas_reach_the_final_run_not_the_library_defaults():
    """The bug this exists to prevent, and it shipped once.

    `protocol.transfer` reads betas by KEY. A study records them under
    `*_horizons`, so transfer saw no betas at all and fell back to
    REFERENCE_BETAS -- every arm landing on the same (0.9, 0.999)-derived pair
    and the entire beta search, the reason the campaign was re-run, discarded.
    """
    cfg = fc.final_config("ibprobit_noeps", _study_shaped(), effective_batch=512,
                          from_samples=1_000_000)
    tune_steps = protocol.steps_for_samples(1_000_000, 512)
    final_steps = protocol.steps_for_samples(5_124_668, 512)
    # The horizon is the invariant, and it is what must survive.
    assert protocol.horizons(cfg["beta1"], final_steps) == pytest.approx(72.59376434)
    assert protocol.horizons(cfg["beta2"], final_steps) == pytest.approx(32.37392373)
    # ... and the result is emphatically NOT the transfer of the defaults.
    default = protocol.betas_for(final_steps, from_steps=tune_steps)
    assert cfg["beta1"] != pytest.approx(default[0])
    assert cfg["beta2"] != pytest.approx(default[1])


def test_arms_with_different_searched_betas_get_different_final_betas():
    # The symptom that exposed it: all three arms transferred to an identical
    # pair, which no per-arm search could ever produce.
    a = fc.final_config("ce", _study_shaped(beta1_horizons=40.8, beta2_horizons=0.92,
                                            head_lr_ratio=30.8),
                        effective_batch=512, from_samples=1_000_000)
    b = fc.final_config("ibprobit_noeps", _study_shaped(),
                        effective_batch=512, from_samples=1_000_000)
    assert a["beta1"] != pytest.approx(b["beta1"])
    assert a["beta2"] != pytest.approx(b["beta2"])


def test_a_horizon_without_a_step_count_is_refused():
    # A beta converted at the wrong step count is a plausible number with no
    # error attached, so this raises rather than defaulting.
    with pytest.raises(ValueError, match="step count"):
        fc.absolute_params(_study_shaped(), effective_batch=512)


def test_the_horizon_suffix_is_not_spelled_twice():
    assert fc.HORIZONS_SUFFIX is fs.HORIZONS_SUFFIX


def test_a_real_suggest_trial_round_trips_through_the_recorded_shape():
    """End to end on the actual search space: suggest -> journal shape -> final.

    Uses the RECORDING trial's params (what optuna stores), not suggest()'s
    return value, which is the distinction the shipped bug lived in.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    tune_steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE)
    final_steps = protocol.steps_for_samples(5_124_668, fs.FIXED_BATCH_SIZE)
    for arm in fs.ARMS:
        t = study.ask()
        returned = fs.suggest(t, arm)
        recorded = t.params                      # what the journal holds
        assert "beta1_horizons" in recorded and "beta1" not in recorded
        cfg = fc.final_config(arm, recorded, effective_batch=fs.FIXED_BATCH_SIZE)
        for b in ("beta1", "beta2"):
            assert protocol.horizons(cfg[b], final_steps) == pytest.approx(
                protocol.horizons(returned[b], tune_steps)), (arm, b)
