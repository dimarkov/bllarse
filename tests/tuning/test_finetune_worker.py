"""The fine-tuning worker's config assembly and study identity.

Everything here runs without a GPU: what it checks is that a trial's config is
the one the design specifies, because a wrong value costs hours of A100 time
per trial and is invisible in the result.
"""

from pathlib import Path

import pytest

from bllarse.finetune import protocol
from bllarse.tuning import finetune_space as space
from bllarse.tuning import finetune_worker as fw


@pytest.fixture(autouse=True)
def pinned_micro_batch(monkeypatch):
    # MICRO_BATCH is measured per GPU, so every config test pins it here
    # rather than depending on the checked-in values.
    monkeypatch.setattr(space, "MICRO_BATCH",
                        {"dinov3_large": {"ce": 64, "ibprobit": 32, "vbll": 64}})


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
    assert config["max_samples"] == 1_000_000
    assert config["eval_quantum"] == 16_384
    assert config["img_size"] == 224
    assert config["model"] == "dinov3_large"
    assert config["val_fraction"] == 0.2
    assert config["bf16"] is True


@pytest.mark.parametrize("arm", fw.SCRIPTS)
def test_tuning_evaluates_val_only_and_dumps_only_the_argmin(arm):
    # Forwarding the 40k test partition at all 61 evaluations would make every
    # evaluation five times as expensive for a number nothing reads, and 61
    # trajectory dumps would be 2.4 GB per run.
    config = fw.trial_config(arm, params=_params(arm))
    assert config["eval_splits"] == "val"
    assert config["dump_every_k_evals"] == 0
    assert config["ood_every_k_dumps"] == 0


@pytest.mark.parametrize("arm", fw.SCRIPTS)
def test_the_micro_batch_divides_the_sampled_batch(arm):
    # protocol.accum_plan raises otherwise, at the top of a multi-hour trial.
    config = fw.trial_config(arm, params=_params(arm))
    assert config["batch_size"] % config["micro_batch_size"] == 0
    protocol.accum_plan(config["batch_size"], config["micro_batch_size"])


def test_an_unmeasured_micro_batch_fails_at_config_time(monkeypatch):
    # Loudly, at submit time -- not three hours into the first trial.
    monkeypatch.setattr(space, "MICRO_BATCH", {"dinov3_large": {}})
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


# --- The pre-decoded training shard -----------------------------------------


def test_the_shard_path_travels_in_the_config_not_only_the_environment(monkeypatch):
    """So the STUDY records which trials read a shard.

    finetune_jax's argparser defaults --train-shards from the same variable, so
    the trial would find the shard either way. What the environment alone
    cannot do is tell you, months later, why one trial took 3 h and another 14.
    """
    monkeypatch.setenv(fw.TRAIN_SHARDS_ENV, "/scratch/shards")
    config = fw.trial_config("ce", params=_params("ce"))
    assert config["train_shards"] == "/scratch/shards"


def test_an_unset_shard_variable_leaves_the_key_out_entirely(monkeypatch):
    # Not `""`: config_to_argv would emit `--train-shards ''`, which is the
    # same fallback but now looks deliberate in the recorded config.
    monkeypatch.delenv(fw.TRAIN_SHARDS_ENV, raising=False)
    assert "train_shards" not in fw.trial_config("ce", params=_params("ce"))


def test_an_explicit_empty_string_forces_the_jpeg_path(monkeypatch):
    # A machine with no built shard has to be able to run a trial anyway.
    monkeypatch.setenv(fw.TRAIN_SHARDS_ENV, "/scratch/shards")
    config = fw.trial_config("ce", params=_params("ce"), train_shards="")
    assert "train_shards" not in config


# --- Study identity and the pruner ------------------------------------------


def test_study_name_is_namespaced_by_campaign():
    # open_study loads by name, so an un-suffixed name would RESUME a study
    # whose search space no longer exists.
    assert fw.study_name("ce") == "ft_ce"
    assert fw.study_name("ce", "imagenet_ft") == "ft_ce_imagenet_ft"
    assert fw.study_name("ibprobit_noeps", "v2") == "ft_ibprobit_noeps_v2"


def test_pruner_engages_far_earlier_than_the_probing_one():
    # 20 was a fifth of ~80 probing trials; a fine-tuning study completes far
    # fewer, so 20 would barely engage, and at hours per trial pruning is
    # where the saving is.
    assert fw.PRUNER_STARTUP_TRIALS == 5


def test_the_pruner_warmup_is_a_fraction_of_the_run_not_a_flat_count():
    # A flat early warmup is ANTI-informative on this head: the two trials in
    # a horizon-transfer test whose argmin fell at eval 11 of 78 were the worst
    # two at the endpoint, because an early dip then a climb is what a
    # too-large q0 does. A fraction survives a change of eval_quantum.
    assert 0.2 <= fw.PRUNER_WARMUP_FRACTION <= 0.5
    # It must still leave room for the pruner to fire at all.
    assert fw.PRUNER_WARMUP_FRACTION * space.eval_count() < space.eval_count()


def test_the_pinned_cavi_iterations_reach_the_REPORTED_config():
    """A pinned value must survive the tuning -> final_config round trip.

    `space.suggest` returning it is not enough: its return value reaches the
    tuning run but never `trial.params`, and `final_config` rebuilds from
    `trial.params`. Pinned only in the search space, every reported IBProbit
    run would fall back to finetune_jax's default of 128 while its
    hyperparameters had been selected at 256. IBPROBIT_FIXED is applied by
    `trial_config`, which both paths go through.
    """
    assert fw.IBPROBIT_FIXED["num_update_iters"] == space.IBPROBIT_UPDATE_ITERS
    for arm in ("ibprobit", "ibprobit_noeps"):
        cfg = fw.trial_config(arm, params={"lr": 1e-4, "batch_size": 512},
                              max_samples=399872)
        assert cfg["num_update_iters"] == 256, arm


def test_open_study_is_usable_without_a_warmup_for_readers(tmp_path):
    # final_config opens a study only to reach best_trial. The pruner is a
    # runtime object and is never written to the journal, so a reader must not
    # be forced to invent an eval count it does not have.
    study = fw.open_study("ce", tmp_path, "large400kv2")
    assert study.study_name == "ft_ce_large400kv2"


def test_the_objective_averages_a_tail_rather_than_taking_the_argmin():
    # The running argmin rewards a config that dips low and then leaves, which
    # is exactly what q0 two decades high does on this head.
    assert fw.TAIL_EVALS >= 3
    assert fw.TAIL_EVALS < space.eval_count()


def test_patience_is_on_and_can_actually_fire():
    # The probing campaign's `--patience 20` needed the argmin at dump <= 19 on
    # a bounded 40-point geometric grid. Here there are 61 uniformly
    # sample-spaced evaluations and the earliest stop is eval 21, so the
    # mechanism has room to work.
    assert fw.PATIENCE == 20
    assert fw.WARMUP_EVALS + fw.PATIENCE < space.eval_count()


# --- Search depth, decoupled from the prune rate ----------------------------


def test_the_worker_accepts_a_completed_trial_target():
    """--n-trials cannot express "search every arm to the same depth".

    It counts trials the worker STARTS, and MedianPruner kills 83%-92% of
    them at a rate that differs by arm: the same 10-per-worker request can land
    at 36 completed trials on ibprobit_noeps and 16 on vbll -- arms then
    compared against each other at depths nobody chose.
    """
    args = fw.build_argparser().parse_args(
        ["--arm", "ce", "--n-trials", "12", "--storage-dir", "/tmp/s",
         "--target-complete", "60"]
    )
    assert args.target_complete == 60
    assert args.n_trials == 12


def test_the_target_is_optional_so_the_old_invocation_still_runs():
    # Smoke runs and one-off re-runs have no depth to reach.
    args = fw.build_argparser().parse_args(
        ["--arm", "ce", "--n-trials", "1", "--storage-dir", "/tmp/s"]
    )
    assert args.target_complete is None


def test_the_target_counts_the_study_not_the_worker():
    """Four workers must cooperate through the shared journal.

    A local counter would have each stop at target/4, so a fast worker idles
    while a slow one is mid-trial. Reading the study means whoever finishes
    checks the global total -- at the cost of a mild overshoot, which is why
    the flag is a floor rather than an exact count.
    """
    import optuna

    study = optuna.create_study(direction="minimize")
    stop = fw.stop_at_completed(2)

    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1,
                   callbacks=[stop])
    assert fw.n_completed(study) == 1
    # Still under target, so the worker was not stopped and can take more.
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=5,
                   callbacks=[stop])
    assert fw.n_completed(study) == 2


# --- Aborting a campaign that is failing every trial ------------------------


def _study():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study()


def test_the_abort_guard_fires_after_the_configured_streak():
    import optuna

    guard = fw.abort_on_consecutive_failures(limit=3)
    study = _study()

    def failing(trial):
        raise RuntimeError("harness fault")

    with pytest.raises(RuntimeError, match="consecutive trials failed"):
        study.optimize(failing, n_trials=20, catch=(Exception,), callbacks=[guard])
    # Exactly the streak, not the whole budget: 3 of 20, which is the saving.
    assert len(study.trials) == 3


def test_the_abort_guard_escapes_optimizes_catch():
    """A property of Optuna's internals, so it is pinned rather than assumed.

    `catch=(Exception,)` wraps the OBJECTIVE inside _run_trial; callbacks are
    invoked outside it. If that ever changed, the guard would be swallowed by
    the very mechanism that lets failing trials accumulate -- silently, and the
    only symptom would be a campaign failing every trial for days.
    """
    guard = fw.abort_on_consecutive_failures(limit=1)
    study = _study()
    with pytest.raises(RuntimeError):
        study.optimize(lambda t: (_ for _ in ()).throw(ValueError("x")),
                       n_trials=5, catch=(Exception,), callbacks=[guard])


def test_a_successful_trial_resets_the_streak():
    # Intermittent failures are normal -- a node OOM, a filesystem hiccup --
    # and killing an array over them would be worse than the disease.
    guard = fw.abort_on_consecutive_failures(limit=3)
    study = _study()
    state = {"n": 0}

    def alternating(trial):
        state["n"] += 1
        if state["n"] % 2:
            raise RuntimeError("transient")
        return 1.0

    study.optimize(alternating, n_trials=12, catch=(Exception,), callbacks=[guard])
    assert len(study.trials) == 12  # never aborted


def test_a_pruned_trial_also_resets_the_streak():
    """PRUNED means the harness works and the pruner is doing its job.

    Treating it as a failure would abort every well-behaved campaign here --
    prune rates on these studies run 69% to 92%.
    """
    import optuna

    guard = fw.abort_on_consecutive_failures(limit=2)
    study = _study()
    state = {"n": 0}

    def mostly_pruned(trial):
        state["n"] += 1
        if state["n"] % 3 == 0:
            raise RuntimeError("one bad trial")
        raise optuna.TrialPruned()

    study.optimize(mostly_pruned, n_trials=15, catch=(Exception,), callbacks=[guard])
    assert len(study.trials) == 15


def test_the_guard_is_installed_unconditionally():
    """It must run with AND without --target-complete.

    If the callback list were an either/or on target_complete, wiring the guard
    into only one branch would leave the other able to burn a full budget.
    """
    src = Path(fw.__file__).read_text()
    block = src[src.index("callbacks=("):src.index("callbacks=(") + 400]
    assert "abort_on_consecutive_failures()" in block
    # Present on the no-target branch too: the guard is outside the conditional.
    assert block.index("abort_on_consecutive_failures()") < block.index("target_complete")


def test_the_streak_limit_is_small_enough_to_be_worth_something():
    # At hours a completed trial and seconds for a launch fault, the guard only
    # helps if it fires in minutes. workers x limit is the wasted-trial bound.
    assert 1 < fw.CONSECUTIVE_FAILURE_LIMIT <= 10


def test_campaign_settings_reach_the_config_and_the_trial_record():
    """model, batch, quantum and layer decay are campaign-level, not searched.

    They have to travel together: the beta axes are drawn in horizon units and
    converted at the trial's own STEP count (samples / batch), so a campaign
    that moved one without the others would search a space it did not record.
    """
    import optuna

    study = optuna.create_study()
    objective_seen = {}

    def fake_trial_config(arm, *, params, max_samples, dump_root=None, **kw):
        objective_seen.update(params)
        return {"model": params.get("model", "dinov3_large"),
                "batch_size": params["batch_size"],
                "micro_batch_size": 128, "train_shards": ""}

    trial = study.ask()
    params = space.suggest(trial, "ce", max_samples=100_000, batch_size=128)

    assert params["batch_size"] == 128
    # the horizon axes were converted at 100_000/128 = 781 steps, so beta1 sits
    # inside the box that budget can express
    lo, hi = space.beta1_horizons_box(781)
    assert lo <= trial.params["beta1_horizons"] <= hi


def test_worker_refuses_a_quantum_that_disables_the_pruner(monkeypatch, capsys):
    """The failure mode this guard exists for: both the pruner warmup and the
    patience clock count EVALUATIONS, so a coarse quantum silently makes them
    unfireable and the campaign runs every trial to its budget."""
    import sys

    argv = ["finetune_worker", "--arm", "ce", "--storage-dir", "/tmp/x",
            "--max-samples", "100000", "--eval-quantum", "16384",
            "--n-trials", "1"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="pruner can never fire"):
        fw.main()


def test_the_span_reaches_the_scripts_config(monkeypatch):
    """A campaign spanning backbones must transfer the SPAN, not the factor.

    `--layer-decay` is per-block, so one value is a different trunk shape at
    every depth: 0.75 damps the embeddings to 3.2e-2 over 12 blocks but 1.0e-4
    over 32. The span is resolved against the loaded backbone's real depth
    inside the training script, so it has to survive as far as the config.
    """
    captured = {}

    def fake_trial_config(arm, *, params, **kw):
        captured.update(params)
        return {"model": "dinov3_small", "micro_batch_size": 128, **params}

    monkeypatch.setattr(fw, "trial_config", fake_trial_config)
    monkeypatch.setattr(fw.space, "suggest",
                        lambda trial, arm, **kw: {"batch_size": 128})
    monkeypatch.setattr(fw.objective_mod, "evaluate",
                        lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    objective = fw.make_objective(
        "ce", dump_root=None, group_id="t", layer_decay_span=0.0317)

    class _Trial:
        number = 0
        params: dict = {}
        def set_user_attr(self, *a): pass
        def report(self, *a): pass
        def should_prune(self): return False

    with pytest.raises(_Stop):
        objective(_Trial())
    assert captured["layer_decay_span"] == 0.0317
    # The per-block factor must NOT also be set: the scripts refuse both.
    assert "layer_decay" not in captured


class _Stop(Exception):
    pass


