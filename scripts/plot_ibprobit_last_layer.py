"""Plot results from the IBProbit large-batch experiment.

Produces one figure per dataset. Each figure has a grid of subplots:
  - Rows: metrics (Accuracy, ECE, NLL)
  - Columns: model sizes (num_blocks x embed_dim)

Within each subplot the x-axis is batch size and each line corresponds
to a different number of CAVI update iterations.  Shaded bands show
+/- 1 standard deviation over the 100 repeats.  The pretrained baseline
is shown as a horizontal dashed line.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


METRICS = ["acc", "ece", "nll"]
METRIC_LABELS = {
    "acc": "Accuracy",
    "ece": "ECE",
    "nll": "NLL",
}
# Higher is better for acc; lower is better for ece/nll
METRIC_BETTER = {"acc": "higher", "ece": "lower", "nll": "lower"}

def make_figure(df_dataset, dataset_name, data_aug, pretrained_sources, output_dir):
    """Create and save one figure for a dataset, combining all sources."""

    df_ib = df_dataset[df_dataset["type"] == "ibprobit"]
    df_bl = df_dataset[
        (df_dataset["type"] == "baseline")
        & (df_dataset["pretrained_source"] == "in21k_cifar")
    ]

    if df_ib.empty:
        return

    # Determine available model variants (sorted)
    model_variants = (
        df_ib.groupby(["num_blocks", "embed_dim"])
        .size()
        .reset_index()[["num_blocks", "embed_dim"]]
        .values.tolist()
    )
    model_variants.sort(key=lambda x: (x[0], x[1]))

    # Defined column structure
    # Cols 0-2: in21k_cifar
    # Cols 3-5: in21k
    source_columns = ["in21k_cifar", "in21k"]
    source_labels = {"in21k_cifar": "in21k+cifar", "in21k": "in21k"}
    
    n_rows = len(model_variants)
    n_cols = len(source_columns) * len(METRICS) # 2 * 3 = 6

    # 4 rows x 6 columns
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 2.5 * n_rows),
        squeeze=False,
        sharex=True,
    )

    batch_sizes = sorted(df_ib["batch_size"].unique())
    update_iters_list = sorted(df_ib["update_iters"].unique())

    # Divergent colormap
    cmap = plt.cm.RdYlBu
    colors = {
        it: cmap(i / max(len(update_iters_list) - 1, 1))
        for i, it in enumerate(update_iters_list)
    }

    for row_idx, (nb, ed) in enumerate(model_variants):
        for source_idx, source in enumerate(source_columns):
            # Filter for this specific row's model & source
            df_model = df_ib[
                (df_ib["num_blocks"] == nb)
                & (df_ib["embed_dim"] == ed)
                & (df_ib["pretrained_source"] == source)
            ]
            
            # Baseline (in21k_cifar)
            df_bl_model = df_bl[(df_bl["num_blocks"] == nb) & (df_bl["embed_dim"] == ed)]

            for m_idx, metric in enumerate(METRICS):
                col_idx = source_idx * 3 + m_idx
                ax = axes[row_idx, col_idx]

                # Baseline horizontal line (always plot if available)
                if not df_bl_model.empty:
                    bl_val = df_bl_model[metric].values[0]
                    ax.axhline(bl_val, color="grey", ls="--", lw=1.2, label="baseline")

                # One line per update_iters value
                if not df_model.empty:
                    for ui in update_iters_list:
                        df_ui = df_model[df_model["update_iters"] == ui]
                        if df_ui.empty:
                            continue

                        stats = df_ui.groupby("batch_size")[metric].agg(["mean", "std"])
                        stats = stats.reindex(batch_sizes).dropna()

                        x = np.array(stats.index)
                        y = stats["mean"].values
                        yerr = stats["std"].values

                        ax.plot(x, y, marker="o", ms=3.5, color=colors[ui], label=f"iters={ui}")
                        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=colors[ui])

                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.set_major_locator(ticker.FixedLocator(batch_sizes))
                ax.tick_params(axis="x", rotation=45)

                # Titles on top row
                if row_idx == 0:
                    display_source = source_labels.get(source, source)
                    ax.set_title(f"{display_source}\n{METRIC_LABELS[metric]}", fontsize=10, fontweight="bold")
                
                # X-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Batch size")

                # Y-label / Row label on left column
                if col_idx == 0:
                    label_str = f"B={nb}, D={ed}"
                    ax.set_ylabel(label_str, fontsize=10)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        ncol=1,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
    )

    fig.suptitle(
        "CIFAR-10" if dataset_name == 'cifar10' else 'CIFAR-100',
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()

    fname = f"{output_dir}/linear_probing_ibprobit_{dataset_name}_{'aug' if data_aug else 'noaug'}.pdf"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="results_ibprobit_last_layer.csv",
        help="Path to the results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--data-aug",
        action="store_true",
        help="Use results with data augmentation",
    )
    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    datasets = df["dataset"].unique()
    pretrained_sources = df.loc[df["type"] == "ibprobit", "pretrained_source"].unique()

    for dataset_name in datasets:
        df_ds = df[df["dataset"] == dataset_name]
        loc1 = df_ds["data_aug"] == args.data_aug
        loc2 = df_ds['type'] == 'baseline'
        df_ds = df_ds[loc1 | loc2]
        make_figure(df_ds, dataset_name, args.data_aug, pretrained_sources, args.output_dir)


if __name__ == "__main__":
    main()
