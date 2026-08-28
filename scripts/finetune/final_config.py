#!/usr/bin/env python3
"""Print the final campaign's command line for a finished tuning study.

The tuning campaign is capped at 2,000,000 samples and the final campaign at 4
epochs (5,124,668). Those are different operating points, and the optimum moves
between them: under the AdamW SDE transfer rule the learning rate and weight
decay both take a factor sqrt(m_B / m_D) = 0.625, and the AdamW betas move so
that (1 - beta) * steps is held invariant. See `bllarse.finetune.protocol` for
the rule and `bllarse.tuning.final_config` for the assembly.

This exists so that the transfer is DERIVED and recorded rather than done by
hand in a shell history. Usage:

    scripts/finetune/final_config.py --arm ce --storage-dir /ptmp/$USER/ft_studies \\
        --campaign fixb512

    # every arm at once, as a shell script
    scripts/finetune/final_config.py --arm ce --arm ibprobit_noeps --arm vbll \\
        --campaign fixb512 --storage-dir /ptmp/$USER/ft_studies --format sh

It prints only; nothing is submitted. The width transplant (tuned on
dinov3_large, reported on dinov3_huge) is NOT corrected -- pass --model to make
it explicit in the emitted command, and see final_config.final_config's
docstring for why no correction is applied.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse.tools import adapters  # noqa: E402
from bllarse.tuning import final_config as fc  # noqa: E402
from bllarse.tuning import finetune_worker  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True,
                   choices=sorted(finetune_worker.SCRIPTS),
                   help="Repeatable. One study is read per arm.")
    p.add_argument("--storage-dir", required=True,
                   help="Directory holding the studies' journal files.")
    p.add_argument("--campaign", default=None,
                   help="Campaign suffix the studies were created with.")
    p.add_argument("--epochs", type=float, default=fc.FINAL_EPOCHS)
    p.add_argument("--seed", type=int, default=None,
                   help="Reported seed for this run. One process per seed; the "
                        "uid the scripts derive is keyed on it, so omitting it "
                        "makes every seed the same run.")
    p.add_argument("--eval-quantum", type=int, default=None,
                   help="Samples between evaluations. Coarser than tuning's by "
                        "default in the launcher: at protocol.EVAL_QUANTUM the "
                        "val+test forwards roughly double a 4-epoch run.")
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Target effective batch. Defaults to the tuned batch, i.e. a pure "
             "duration transfer. Changing it also applies the batch rule, and "
             "is refused on the IBProbit arms (q0 = alpha^2 / B is a second "
             "batch relation with no rule to reconcile it).",
    )
    p.add_argument(
        "--model", default=None,
        help="Override the backbone. The final campaign reports dinov3_huge "
             "while tuning ran on dinov3_large; no width correction is applied "
             "either way.",
    )
    p.add_argument("--dump-root", default=None)
    p.add_argument("--dump-every-k-evals", type=int, default=None,
                   help="Trajectory dumps for Fig 5. Omitted leaves "
                        "trial_config's value, which is 0 -- one dump per run.")
    p.add_argument("--ood-checkpoint-root", default=None,
                   help="Delegate OOD to the worker: write checkpoints at the "
                        "scheduled steps instead of evaluating inline.")
    p.add_argument("--ood-max-pending", type=int, default=None)
    p.add_argument("--ood-every-k-dumps", type=int, default=None,
                   help="OOD dumps, as a stride of the dump grid. Needs "
                        "--dump-every-k-evals nonzero; finetune_jax gates on "
                        "both.")
    p.add_argument("--eval-splits", default=fc.FINAL_EVAL_SPLITS,
                   choices=["val", "val+test"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override ONE config key after the transfer, e.g. "
                        "--set q_init=1.2e-4 or --set no_learn_q=false. For "
                        "deliberate, recorded deviations from the reported "
                        "config -- an ablation sweep -- so the run is still "
                        "the study's config plus exactly the named change, "
                        "with nothing hand-copied. The key must already exist "
                        "in the transferred config, or a typo would silently "
                        "add a flag the script ignores.")
    p.add_argument("--group-id", default=None,
                   help="Dump group. Set it for a sweep so the dumps land "
                        "beside, not inside, the reported campaign.")
    p.add_argument("--uid", default=None,
                   help="Run identity. REQUIRED when a sweep varies something "
                        "other than (model, dataset, arm, seed), since the "
                        "derived uid is keyed on those alone and two sweep "
                        "points would otherwise share a directory.")
    p.add_argument("--format", default="argv", choices=["argv", "sh", "json", "exec"],
                   help="`exec` replaces this process with the training run, "
                        "so a SLURM task resolves its own config from the "
                        "study instead of having one templated into a script.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    for arm in args.arm:
        study = finetune_worker.open_study(
            arm, args.storage_dir, campaign=args.campaign)
        kw = {} if args.seed is None else {"seed": args.seed}
        if args.eval_quantum is not None:
            kw["eval_quantum"] = args.eval_quantum
        config = fc.best_config(
            study, arm, **kw,
            epochs=args.epochs,
            to_batch=args.batch_size,
            model=args.model,
            dump_root=args.dump_root,
            eval_splits=args.eval_splits,
            dump_every_k_evals=args.dump_every_k_evals,
            ood_every_k_dumps=args.ood_every_k_dumps,
            ood_checkpoint_root=args.ood_checkpoint_root,
            ood_max_pending=args.ood_max_pending,
        )
        if args.group_id is not None:
            config["group_id"] = args.group_id
        if args.uid is not None:
            config["uid"] = args.uid
        for item in args.set:
            key, _, raw = item.partition("=")
            if not _:
                raise SystemExit(f"[final] --set wants KEY=VALUE, got {item!r}")
            if key not in config:
                raise SystemExit(
                    f"[final] --set {key}: not in the transferred config. "
                    f"Known keys: {', '.join(sorted(config))}")
            current = config[key]
            if isinstance(current, bool) or raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            print(f"[final] override {key}: {current!r} -> {value!r}", flush=True)
            config[key] = value

        script = finetune_worker.SCRIPTS[arm]
        argv = adapters.config_to_argv(config)

        if args.format == "exec":
            if len(args.arm) != 1:
                raise SystemExit("[final] --format exec takes exactly one --arm")
            repo = str(Path(__file__).resolve().parents[2])
            print(f"[final] exec {script} ({arm}, seed {config['seed']}, "
                  f"uid derived from model/dataset/arm/seed)", flush=True)
            os.execv(sys.executable,
                     [sys.executable, str(Path(repo) / script)] + argv)

        if args.format == "json":
            print(json.dumps({"arm": arm, "script": script, "config": config},
                             indent=2, sort_keys=True))
            continue

        best = study.best_trial
        print(f"# {arm}: trial {best.number}, val_nll {study.best_value:.5f}, "
              f"tuned at batch "
              f"{best.user_attrs.get('effective_batch', '?')} / "
              f"{best.user_attrs.get('max_samples', '?')} samples")
        if args.format == "sh":
            # Group each flag with its value, tolerating store_true flags
            # (--bf16) which config_to_argv emits with no value after them.
            lines, i = [], 0
            while i < len(argv):
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    lines.append(f"{argv[i]} {argv[i + 1]}")
                    i += 2
                else:
                    lines.append(argv[i])
                    i += 1
            print(f"python {script} \\\n    " + " \\\n    ".join(lines))
        else:
            print(f"python {script} " + " ".join(argv))
        print()


if __name__ == "__main__":
    main()
