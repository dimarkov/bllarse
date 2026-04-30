import os
import argparse
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from jax import config
from datasets import load_dataset
from dotenv import load_dotenv
from functools import partial
import warnings
import time

load_dotenv()
load_dotenv(".env.secrets", override=False)

# Imports from project
from mlpox.load_models import load_model
from bllarse.losses import IBProbit, CrossEntropy
from bllarse.utils import (
    run_training,
    resize_images,
    augmentdata,
    evaluate_model,
    extract_features,
    MEAN_DICT,
    STD_DICT,
)
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as compute_ece,
)

warnings.filterwarnings("ignore")


def get_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name).with_format("jax")
    label_key = "fine_label" if dataset_name == "cifar100" else "label"
    train_ds = {
        "image": ds["train"]["img"][:].astype(jnp.float32),
        "label": ds["train"][label_key][:],
    }
    test_ds = {
        "image": ds["test"]["img"][:].astype(jnp.float32),
        "label": ds["test"][label_key][:],
    }
    return train_ds, test_ds


def run_experiment(args):
    # --- Configuration Grid ---
    datasets = args.datasets

    # Large batch size limit
    batch_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    # Update iterations
    update_iters_list = [2, 4, 8, 16, 32, 64]

    # Pretrained starting points
    pretrained_types = args.pretrained_types

    # Model variants: (num_blocks, embed_dim)
    model_variants = [tuple(int(x) for x in v.split(",")) for v in args.model_variants]

    repeats = 25
    epochs = 1

    # Data Augmentation Status
    use_data_aug = not args.nodataaug

    # --- Test Mode Override ---
    if args.test_run:
        print("!!! RUNNING IN TEST MODE (Reduced Grid) !!!")
        datasets = ["cifar10"]
        batch_sizes = [512, 16384]  # Test one small, one large
        update_iters_list = [2, 64]  # Test limits
        pretrained_types = ["in21k_cifar"]
        model_variants = [(6, 512)]
        repeats = 2  # Small number of repeats

    output_file = args.output
    # Create CSV with header if it doesn't exist
    if not os.path.exists(output_file):
        df_header = pd.DataFrame(
            columns=[
                "dataset",
                "num_blocks",
                "embed_dim",
                "pretrained_source",
                "type",
                "batch_size",
                "update_iters",
                "repeat_idx",
                "data_aug",
                "acc",
                "ece",
                "nll",
                "time_taken",
                "temperature",
                "test_fraction",
            ]
        )
        df_header.to_csv(output_file, index=False)

    for dataset_name in datasets:
        # Load and resize data once per dataset
        train_ds, test_ds = get_dataset(dataset_name)
        num_classes = 100 if dataset_name == "cifar100" else 10
        img_size = 64

        print(f"Resizing images for {dataset_name}...")
        train_ds["image"] = resize_images(train_ds["image"], img_size)
        test_ds["image"] = resize_images(test_ds["image"], img_size)

        mean = MEAN_DICT[dataset_name]
        std = STD_DICT[dataset_name]
        augdata_base = partial(augmentdata, mean=mean, std=std)

        # Determine Augmentation Function
        if use_data_aug:
            print("  Data Augmentation: ENABLED (Random Crop/Flip)")
            augdata_fn = augdata_base
        else:
            print("  Data Augmentation: DISABLED (Normalization Only)")
            # Force key=None to disable random augmentations
            augdata_fn = lambda img, key=None, **kwargs: augdata_base(
                img, key=None, **kwargs
            )

        # Evaluation always uses deterministic mode (effectively same as nodataaug)
        _augdata_eval = lambda img, key=None, **kwargs: augdata_base(
            img, key=None, **kwargs
        )

        for num_blocks, embed_dim in model_variants:
            print(f"--- Model: Blocks={num_blocks}, Dim={embed_dim} ---")

            # 1. Evaluate Baseline (in21k_cifar) once per model/dataset combo
            baseline_name = f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k_{dataset_name}"
            print(f"  Evaluating Baseline: {baseline_name}")
            try:
                baseline_nnet = eqx.nn.inference_mode(load_model(baseline_name), True)
                loss_fn_baseline = CrossEntropy(0.0, num_classes)

                start_t = time.time()
                acc, nll, ece = evaluate_model(
                    _augdata_eval,
                    loss_fn_baseline,
                    baseline_nnet,
                    test_ds["image"],
                    test_ds["label"],
                )
                elapsed = time.time() - start_t

                print(
                    f"    Baseline Results: Acc={acc:.4f}, ECE={ece:.4f}, NLL={nll:.4f}"
                )

                # Save baseline result
                row = {
                    "dataset": dataset_name,
                    "num_blocks": num_blocks,
                    "embed_dim": embed_dim,
                    "pretrained_source": "in21k_cifar",
                    "type": "baseline",
                    "batch_size": -1,
                    "update_iters": -1,
                    "repeat_idx": 0,
                    "data_aug": False,
                    "acc": float(acc),
                    "ece": float(ece),
                    "nll": float(nll),
                    "time_taken": elapsed,
                    "temperature": float("nan"),
                    "test_fraction": float("nan"),
                }
                pd.DataFrame([row]).to_csv(
                    output_file, mode="a", header=False, index=False
                )

                # Optional: temperature scaling using a fraction of the test set
                # (calibrate beta on f*N_test, evaluate on the remaining (1-f)*N_test).
                # Repeated `repeats` times per fraction with different random splits to
                # match the IBProbit linear-probing repeat statistics.
                if args.temperature_scaling:
                    print(
                        f"    Running TS on test fractions {args.ts_test_fractions} "
                        f"(Newton iters={args.ts_iters}, {repeats} repeats per fraction)"
                    )
                    aug_test = _augdata_eval(test_ds["image"], key=None)
                    test_logits_all = extract_features(baseline_nnet, aug_test)
                    test_lbls_all = test_ds["label"]
                    n_test = test_logits_all.shape[0]

                    ts_rows = []
                    for f in args.ts_test_fractions:
                        n_cal = int(round(f * n_test))
                        accs, eces, nlls, betas = [], [], [], []
                        for r_idx in range(repeats):
                            ts_seed = (
                                args.seed
                                + r_idx
                                + int(round(f * 100000))
                                + (hash(("ts", dataset_name, num_blocks, embed_dim))
                                   % 100000)
                            )
                            ts_key = jr.PRNGKey(ts_seed % (2**32))
                            perm = jr.permutation(ts_key, n_test)
                            cal_idx = perm[:n_cal]
                            eval_idx = perm[n_cal:]

                            cal_logits = test_logits_all[cal_idx]
                            cal_lbls = test_lbls_all[cal_idx]
                            eval_logits = test_logits_all[eval_idx]
                            eval_lbls = test_lbls_all[eval_idx]

                            start_t = time.time()
                            loss_fn_ts = CrossEntropy(0.0, num_classes).calibrate(
                                cal_logits, cal_lbls, num_iters=args.ts_iters
                            )
                            beta_ts = float(loss_fn_ts.beta)
                            nll_ts_arr, scaled_eval = loss_fn_ts(
                                eval_logits, eval_lbls, with_logits=True
                            )
                            preds_ts = jnp.argmax(scaled_eval, axis=-1)
                            acc_ts = float(jnp.mean(preds_ts == eval_lbls))
                            ece_ts = float(compute_ece(
                                20,
                                logits=scaled_eval,
                                labels_true=eval_lbls,
                                labels_predicted=preds_ts,
                            ))
                            nll_ts = float(nll_ts_arr.mean())
                            elapsed_ts = time.time() - start_t

                            accs.append(acc_ts)
                            eces.append(ece_ts)
                            nlls.append(nll_ts)
                            betas.append(beta_ts)

                            ts_rows.append({
                                "dataset": dataset_name,
                                "num_blocks": num_blocks,
                                "embed_dim": embed_dim,
                                "pretrained_source": "in21k_cifar",
                                "type": "baseline_ts",
                                "batch_size": -1,
                                "update_iters": args.ts_iters,
                                "repeat_idx": r_idx,
                                "data_aug": False,
                                "acc": acc_ts,
                                "ece": ece_ts,
                                "nll": nll_ts,
                                "time_taken": elapsed_ts,
                                "temperature": 1.0 / beta_ts,
                                "test_fraction": float(f),
                            })

                        accs_arr = jnp.asarray(accs)
                        eces_arr = jnp.asarray(eces)
                        nlls_arr = jnp.asarray(nlls)
                        betas_arr = jnp.asarray(betas)
                        Ts_arr = 1.0 / betas_arr
                        print(
                            f"      f={f:.3f}  n_cal={n_cal}  n_eval={n_test - n_cal}  "
                            f"T={float(Ts_arr.mean()):.3f}±{float(Ts_arr.std()):.3f}  "
                            f"Acc={float(accs_arr.mean()):.4f}±{float(accs_arr.std()):.4f}  "
                            f"ECE={float(eces_arr.mean()):.4f}±{float(eces_arr.std()):.4f}  "
                            f"NLL={float(nlls_arr.mean()):.4f}±{float(nlls_arr.std()):.4f}"
                        )

                    pd.DataFrame(ts_rows).to_csv(
                        output_file, mode="a", header=False, index=False
                    )

            except Exception as e:
                print(f"    Failed to load/eval baseline {baseline_name}: {e}")

            if args.skip_ibprobit:
                continue

            # 2. Run IBProbit Experiments
            for p_source in pretrained_types:
                # Load the appropriate backbone
                if p_source == "in21k_cifar":
                    model_name_to_load = (
                        f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k_{dataset_name}"
                    )
                else:  # in21k
                    model_name_to_load = f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k"

                print(f"  Loading Backbone from: {model_name_to_load} ({p_source})")
                try:
                    loaded_nnet = eqx.nn.inference_mode(
                        load_model(model_name_to_load), True
                    )
                except Exception as e:
                    print(f"    Error loading {model_name_to_load}: {e}")
                    continue

                for batch_size in batch_sizes:
                    for num_update_iters in update_iters_list:
                        print(
                            f"    Config: Source={p_source}, BS={batch_size}, Iters={num_update_iters}"
                        )

                        # Collect repeats
                        rows_buffer = []

                        for r_idx in range(repeats):
                            # Seed handling: ensure variation across repeats
                            run_seed = (
                                args.seed
                                + r_idx
                                + (hash(p_source) % 1000)
                                + batch_size
                                + num_update_iters
                            )
                            key = jr.PRNGKey(run_seed % (2**32))
                            key_init, key_train = jr.split(key)

                            # Initialize fresh IBProbit head
                            loss_fn = IBProbit(embed_dim, num_classes, key=key_init)

                            start_t = time.time()
                            trained_loss_fn, trained_nnet, _, _ = run_training(
                                key_train,
                                loaded_nnet,
                                loss_fn,
                                augdata_fn,
                                train_ds,
                                test_ds,
                                optimizer=None,
                                tune_last_layer_only=True,
                                loss_type=3,
                                num_epochs=epochs,
                                batch_size=batch_size,
                                num_update_iters=num_update_iters,
                                mc_samples=1,
                                log_to_wandb=False,
                            )

                            # Eval
                            acc, nll, ece = evaluate_model(
                                _augdata_eval,
                                trained_loss_fn,
                                trained_nnet,
                                test_ds["image"],
                                test_ds["label"],
                                loss_type=3
                            )
                            elapsed = time.time() - start_t

                            rows_buffer.append(
                                {
                                    "dataset": dataset_name,
                                    "num_blocks": num_blocks,
                                    "embed_dim": embed_dim,
                                    "pretrained_source": p_source,
                                    "type": "ibprobit",
                                    "batch_size": batch_size,
                                    "update_iters": num_update_iters,
                                    "repeat_idx": r_idx,
                                    "data_aug": use_data_aug,
                                    "acc": float(acc),
                                    "ece": float(ece),
                                    "nll": float(nll),
                                    "time_taken": elapsed,
                                    "temperature": float("nan"),
                                    "test_fraction": float("nan"),
                                }
                            )

                            if (r_idx + 1) % 10 == 0:
                                print(
                                    f"      Repeat {r_idx + 1}/{repeats} done. (Acc: {acc:.3f})"
                                )

                        # Save buffer to CSV
                        pd.DataFrame(rows_buffer).to_csv(
                            output_file, mode="a", header=False, index=False
                        )

    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="results_ibprobit_last_layer.csv",
        help="Output CSV file",
    )
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifar10", "cifar100"],
        default=["cifar100"],
        help="Datasets to run experiments on",
    )
    parser.add_argument(
        "--pretrained-types",
        nargs="+",
        choices=["in21k", "in21k_cifar"],
        default=["in21k"],
        dest="pretrained_types",
        help="Pretrained backbone sources to use",
    )
    parser.add_argument(
        "--model-variants",
        nargs="+",
        default=["6,512", "12,512", "6,1024", "12,1024"],
        dest="model_variants",
        metavar="BLOCKS,DIM",
        help="Model variants as BLOCKS,DIM pairs (e.g. 6,512 12,1024)",
    )
    parser.add_argument(
        "--test-run", action="store_true", help="Run with reduced grid for testing"
    )
    parser.add_argument(
        "--nodataaug",
        action="store_true",
        help="Disable data augmentation (normalization only)",
    )
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit beta on a fraction of the test set and evaluate on the rest. "
             "Sweeps over --ts-test-fractions, with `repeats` random splits per "
             "fraction; emits one 'baseline_ts' row per (fraction, repeat).",
    )
    parser.add_argument(
        "--ts-iters",
        type=int,
        default=4,
        help="Newton iterations per TS calibration (default: 4 — convex CE "
             "converges fast).",
    )
    parser.add_argument(
        "--ts-test-fractions",
        nargs="+",
        type=float,
        default=[0.10],
        help="Fractions of the test set used as TS calibration data.",
    )
    parser.add_argument(
        "--skip-ibprobit",
        action="store_true",
        help="Skip the IBProbit sweep (only run baseline / temperature scaling).",
    )
    args = parser.parse_args()

    run_experiment(args)
