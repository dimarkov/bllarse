"""The VBLL fine-tuning arm's contract with the rest of the campaign.

This arm is the only one in PyTorch, so nothing about its agreement with the
JAX arms is enforced by sharing code paths -- it has to be pinned. Everything
here runs without torch or a GPU: the script imports torch inside the functions
that need it, precisely so this file can check the contract on any machine.

What is NOT checked here is the trunk itself. That is
scripts/finetune/trunk_parity.py, which needs both frameworks and a downloaded
checkpoint, and which found the one bug this arm was most likely to ship with.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"))

import finetune_vbll as fv  # noqa: E402

from bllarse.finetune import protocol  # noqa: E402
from bllarse.tools import adapters  # noqa: E402
from bllarse.tuning import finetune_space as space  # noqa: E402
from bllarse.tuning import finetune_worker as fw  # noqa: E402


def _params():
    return {"batch_size": 1024, "lr": 1e-4, "weight_decay": 1e-3,
            "head_lr_ratio": 100.0, "kl_ratio": 1e-6, "prior_scale": 10.0}


@pytest.fixture
def vbll_config(monkeypatch):
    # MICRO_BATCH has no vbll entry until one is MEASURED on the target GPU --
    # micro_batch() raises rather than borrowing the JAX value, since a
    # different framework has a different activation profile. Pinned here so
    # the config assembles.
    monkeypatch.setattr(space, "MICRO_BATCH", dict(space.MICRO_BATCH, vbll=32))
    return fw.trial_config("vbll", params=_params(), train_shards="/ptmp/shards")


# --- The join between the worker and this script ----------------------------


def test_the_worker_config_is_accepted_by_this_script(vbll_config):
    """The one failure that would abort every trial at argparse.

    finetune_worker builds a config dict; adapters.config_to_argv turns it into
    flags; this script's parser has to accept all of them. A key the parser
    does not know is not a wrong number, it is an immediate SystemExit on every
    trial of the arm -- and it would only show up on the cluster.
    """
    args = adapters.parse_config(fv, vbll_config)
    assert args.batch_size == 1024
    assert args.kl_ratio == 1e-6
    assert args.prior_scale == 10.0
    assert args.wishart_scale == space.WISHART_SCALE


def test_head_lr_is_the_trunk_rate_times_the_sampled_ratio(vbll_config):
    # The ratio is the searched axis, not two independent rates: a fresh head
    # at the trunk's rate stays near its init for a whole epoch.
    assert vbll_config["head_lr"] == pytest.approx(1e-4 * 100.0)
    args = adapters.parse_config(fv, vbll_config)
    assert args.head_lr == pytest.approx(1e-2)
    assert args.lr == pytest.approx(1e-4)


def test_the_micro_batch_must_be_given(vbll_config):
    """Never auto-probed. Design A8 pins it so num_accum is exact."""
    del vbll_config["micro_batch_size"]
    with pytest.raises(SystemExit):
        adapters.parse_config(fv, vbll_config)


# --- Agreement with the JAX arms --------------------------------------------


@pytest.mark.parametrize("batch", space.BATCH_SIZES)
def test_the_evaluation_grid_matches_the_jax_arms_at_every_rung(batch, vbll_config):
    """Eval index i must mean the same samples_seen in every arm.

    It is MedianPruner's key and the patience clock. If this arm evaluated on
    its own grid, the pruner would compare a VBLL trial's index 30 against a CE
    trial's index 30 at different amounts of training -- silently, and in the
    pruner's favour or against it depending on the batch.
    """
    import finetune_jax as fj

    # Each arm's OWN config, not the vbll one reused: finetune_jax's parser
    # rejects --kl-ratio and friends, so passing the vbll config through it
    # would fail at argparse and never reach the thing being compared.
    ce_params = {"batch_size": batch, "lr": 1e-4, "weight_decay": 1e-3,
                 "head_lr_ratio": 100.0}
    v = fv.resolve_plan(
        adapters.parse_config(fv, dict(vbll_config, batch_size=batch)), 32)
    j = fj.resolve_plan(
        adapters.parse_config(fj, fw.trial_config("ce", params=ce_params)), 32)
    for key in ("effective_batch", "steps", "eval_every", "n_evals"):
        assert v[key] == j[key], f"{key} differs at batch {batch}"


def test_the_budget_is_the_campaign_budget(vbll_config):
    args = adapters.parse_config(fv, vbll_config)
    assert args.max_samples == protocol.MAX_SAMPLES
    assert args.eval_quantum == protocol.EVAL_QUANTUM
    assert fv.resolve_plan(args, 32)["n_evals"] == space.eval_count()


def test_tuning_evaluates_val_only_and_writes_one_dump(vbll_config):
    args = adapters.parse_config(fv, vbll_config)
    assert args.eval_splits == "val"
    # dump_every_k_evals 0 is what routes writing through _write_best, so the
    # run produces its buffered argmin and nothing else.
    assert args.dump_every_k_evals == 0
    assert args.ood_every_k_dumps == 0


# --- Things that are silently wrong rather than loudly broken ---------------


def test_the_uid_is_shaped_exactly_like_the_jax_arms():
    """Both arms' uids have to be groupable by one piece of code.

    What this does NOT assert is that the dataset field comes out clean.
    fig1_curves.UID_RE is {model}_(?P<dataset>.+)_seed{seed} with a GREEDY
    dataset group, so dinov3_large_imagenet1k_vbll_seed137 parses as
    dataset="imagenet1k_vbll" -- on this arm and on the JAX ones alike, whose
    comment used to claim otherwise. Harmless for a campaign with one dataset,
    but pinned here so the next person reads the truth rather than the claim.
    """
    from fig1_curves import UID_RE

    uid = "dinov3_large_imagenet1k_vbll_seed137"
    match = UID_RE.match(uid)
    assert match is not None
    assert match["model"] == "dinov3_large"
    assert int(match["seed"]) == 137
    assert match["dataset"] == "imagenet1k_vbll"


def test_the_shard_is_required():
    # Without it the arm decodes JPEGs: ~40 img/s against ~220, so a 3 h trial
    # becomes 14 h. A slow failure, not a visible one -- so it is rejected
    # before torch is even imported.
    args = fv.build_argparser().parse_args(["--micro-batch-size", "32"])
    args.train_shards = ""
    with pytest.raises(SystemExit, match="train-shards"):
        fv.check_required_flags(args)


def test_a_shard_path_satisfies_the_flag_check():
    args = fv.build_argparser().parse_args(
        ["--micro-batch-size", "32", "--train-shards", "/ptmp/shards"])
    fv.check_required_flags(args)


def test_every_hf_checkpoint_has_a_registry_entry():
    # The two maps are keyed by the same model names on purpose: main() cross
    # checks transformers' hidden_size against the registry's embed_dim, and a
    # name present in one and absent from the other would KeyError mid-run.
    from bllarse.finetune.registry import EQUIMO_MODELS

    for name in fv.HF_CHECKPOINTS:
        assert name in EQUIMO_MODELS


# --- OOD parity with the JAX arms -------------------------------------------


def test_vbll_masks_r_and_a_exactly_as_the_jax_arms_do():
    """Without this the arms are scored on different predictives.

    R and A cover 200 classes, so the JAX arms mask the other 800 before
    scoring. A VBLL arm scored on the full 1000-way predictive over the same
    images would produce OOD numbers that are not comparable -- and nothing
    downstream would show it, because both look like valid dumps.
    """
    import inspect

    import numpy as np

    from bllarse.finetune import imagenet as im

    sig = inspect.signature(fv.evaluate_split)
    assert "subset" in sig.parameters
    assert sig.parameters["subset"].default is None
    # The sentinel is shared, not re-spelled, so the two arms cannot drift.
    assert im.MASK_NEG == -1e4
    src = inspect.getsource(fv.evaluate_split)
    assert "imagenet.MASK_NEG" in src
    assert "np.where" in src


def test_both_arms_load_the_same_ood_suite_from_one_place():
    """The loader used to live in finetune_jax.

    That left the torch arm with no way to reach it without importing jax, so
    VBLL would silently have had no OOD at all.
    """
    from bllarse.finetune import imagenet as im

    assert hasattr(im, "load_ood_sets")
    assert hasattr(fv, "load_ood_sets")


def test_the_ood_registry_imports_without_equinox():
    """The torch OOD worker died on this before consuming one checkpoint.

    imagenet.load_ood_sets imports precompute_ood_features, which imported
    vit_classification at module scope, which imports equinox -- absent from
    the torch venv by design (jax[cuda] and torch cannot share an
    environment). Moving the OOD loader into a shared module made it look
    framework-agnostic while it still pulled equinox in transitively.
    """
    import ast
    from pathlib import Path

    src = Path("scripts/dinov3/precompute_ood_features.py").read_text()
    tree = ast.parse(src)
    top_level = {
        n.module for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module
    } | {
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names
    }
    for forbidden in ("vit_classification", "jax", "equinox"):
        assert not any(m == forbidden or m.startswith(forbidden + ".")
                       for m in top_level), (
            f"{forbidden} imported at module scope; the torch venv cannot "
            f"build the OOD registry")
