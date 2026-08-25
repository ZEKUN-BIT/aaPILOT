"""Plot four LigandMPNN checkpoint evaluations on shared visual scales."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("evaluation_runs")
OUTPUT = ROOT / "plots"
DPI = 1600
MODELS = [
    "ligandmpnn_v_32_005_25",
    "ligandmpnn_v_32_010_25",
    "ligandmpnn_v_32_020_25",
    "ligandmpnn_v_32_030_25",
]
LABELS = {model: model.split("_")[-2] for model in MODELS}
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
INK = "#202124"


def set_style():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
    })


def load_structure_means(dataset, model):
    path = ROOT / dataset / model / f"{model}_per_row.csv"
    raw = pd.read_csv(path)
    keys = ["name", "target", "structure_state", "cluster_id"]
    metrics = [
        "recovery", "recovery_vanilla", "bs_recovery",
        "bs_recovery_vanilla", "perplexity", "perplexity_vanilla",
    ]
    return raw.groupby(keys, as_index=False, dropna=False)[metrics].mean()


def load_effects():
    rows = []
    for dataset in ("pathway", "original_test"):
        unit = "cluster" if dataset == "pathway" else "row"
        for model in MODELS:
            summary = pd.read_csv(ROOT / dataset / model / "stratified_summary.csv")
            selected = summary[
                (summary.bootstrap_unit == unit) & (summary.stratum == "all")
            ]
            for metric in ("recovery", "pocket_recovery"):
                values = selected[selected.metric == metric]
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "label": LABELS[model],
                    "metric": metric,
                    "baseline": values.vanilla_mean.mean() * 100,
                    "finetuned": values.finetuned_mean.mean() * 100,
                    "delta": values.mean_delta.mean() * 100,
                    "low": values.ci95_low.min() * 100,
                    "high": values.ci95_high.max() * 100,
                    "bootstrap_unit": unit,
                    "n_rows": int(values.n_rows.iloc[0]),
                })
    return pd.DataFrame(rows)


def clean_axes(ax):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def absolute_comparison(effects):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharex=False)
    for row, dataset in enumerate(("pathway", "original_test")):
        for col, metric in enumerate(("recovery", "pocket_recovery")):
            ax = axes[row, col]
            data = effects[(effects.dataset == dataset) & (effects.metric == metric)]
            y = np.arange(len(data))
            for yi, record in zip(y, data.itertuples()):
                ax.plot([record.baseline, record.finetuned], [yi, yi],
                        color=LIGHT_GRAY, linewidth=2, zorder=1)
            ax.scatter(data.baseline, y, s=48, facecolors="white", edgecolors=GRAY,
                       linewidths=1.5, label="Vanilla", zorder=2)
            ax.scatter(data.finetuned, y, s=52, color=BLUE, edgecolors="white",
                       linewidths=.6, label="Fine-tuned", zorder=3)
            for yi, record in zip(y, data.itertuples()):
                label_x = max(record.baseline, record.finetuned) + .45
                ax.text(label_x, yi, f"{record.finetuned:.1f}", va="center", fontsize=8)
            ax.set_yticks(y, data.label)
            ax.invert_yaxis()
            ax.set_xlabel("Recovery (%)")
            ax.set_title(
                f"{'Pathway test' if dataset == 'pathway' else 'Original test'} · "
                f"{'Overall' if metric == 'recovery' else 'Ligand pocket'}"
            )
            clean_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, -.01))
    fig.suptitle("Model Recovery Comparison", fontsize=17, y=1.01)
    fig.subplots_adjust(bottom=.10)
    save(fig, "all_models_absolute_recovery")


def effect_comparison(effects):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    for row, dataset in enumerate(("pathway", "original_test")):
        for col, metric in enumerate(("recovery", "pocket_recovery")):
            ax = axes[row, col]
            data = effects[(effects.dataset == dataset) & (effects.metric == metric)]
            y = np.arange(len(data))
            errors = np.vstack([data.delta - data.low, data.high - data.delta])
            ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1.2, zorder=0)
            ax.errorbar(data.delta, y, xerr=errors, fmt="o", markersize=6,
                        color=BLUE, ecolor=GRAY, elinewidth=1.4, capsize=3, zorder=2)
            for yi, value in zip(y, data.delta):
                ax.annotate(f"{value:+.2f}", (value, yi), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8)
            ax.set_yticks(y, data.label)
            ax.invert_yaxis()
            ax.set_xlabel("Fine-tuned − vanilla (percentage points)")
            ax.set_title(
                f"{'Pathway test' if dataset == 'pathway' else 'Original test'} · "
                f"{'Overall' if metric == 'recovery' else 'Ligand pocket'}"
            )
            clean_axes(ax)
    fig.suptitle("Recovery Change with Paired-bootstrap 95% CI", fontsize=17, y=1.01)
    fig.text(.5, -.01,
             "Pathway: cluster bootstrap; original test: row bootstrap; "
             "intervals span the three seed-specific 95% CIs.",
             ha="center", fontsize=8, color=GRAY)
    save(fig, "all_models_recovery_delta_ci")


def violin_grid(dataset, metric):
    baseline = "recovery_vanilla" if metric == "overall" else "bs_recovery_vanilla"
    finetuned = "recovery" if metric == "overall" else "bs_recovery"
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        data = load_structure_means(dataset, model)[[baseline, finetuned]].dropna() * 100
        long = data.rename(columns={baseline: "Vanilla", finetuned: "Fine-tuned"}).melt(
            var_name="Version", value_name="Recovery"
        )
        sns.violinplot(data=long, x="Version", y="Recovery", hue="Version",
                       palette={"Vanilla": GRAY, "Fine-tuned": BLUE}, inner="box",
                       cut=0, linewidth=1.0, legend=False, ax=ax)
        ax.set_title(f"Noise {LABELS[model]}")
        ax.set_xlabel("")
        ax.set_ylabel("Recovery (%)" if ax in axes[:, 0] else "")
        ax.set_ylim(0, 100)
        clean_axes(ax)
    scope = "Pathway" if dataset == "pathway" else "Original LigandMPNN"
    kind = "Overall Protein" if metric == "overall" else "Ligand-pocket"
    fig.suptitle(f"{scope} Test · {kind} Recovery", fontsize=17, y=1.01)
    save(fig, f"{dataset}_{metric}_recovery_violin_all_models")


def paired_scatter_grid(dataset, metric):
    baseline = "recovery_vanilla" if metric == "overall" else "bs_recovery_vanilla"
    finetuned = "recovery" if metric == "overall" else "bs_recovery"
    datasets = []
    for model in MODELS:
        data = load_structure_means(dataset, model)[[baseline, finetuned]].dropna() * 100
        datasets.append(data)
    low = np.floor((min(d.min().min() for d in datasets) - 3) / 5) * 5
    high = np.ceil((max(d.max().max() for d in datasets) + 3) / 5) * 5

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4), sharex=True, sharey=True)
    for ax, model, data in zip(axes.flat, MODELS, datasets):
        delta = data[finetuned] - data[baseline]
        improved = delta > 0
        degraded = delta < 0
        same = ~(improved | degraded)
        style = {"s": 8, "alpha": .35, "linewidths": .15, "edgecolors": "white"}
        ax.scatter(data.loc[improved, baseline], data.loc[improved, finetuned],
                   color=BLUE, label=f"Improved ({improved.sum():,})", **style)
        ax.scatter(data.loc[same, baseline], data.loc[same, finetuned],
                   color=GRAY, label="_nolegend_", **style)
        ax.scatter(data.loc[degraded, baseline], data.loc[degraded, finetuned],
                   color=ORANGE, label=f"Degraded ({degraded.sum():,})", **style)
        ax.plot([low, high], [low, high], color="#9CA3AF", linestyle="--",
                linewidth=1.0, zorder=0)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title(f"Noise {LABELS[model]}")
        ax.set_xlabel("Vanilla recovery (%)" if ax in axes[1, :] else "")
        ax.set_ylabel("Fine-tuned recovery (%)" if ax in axes[:, 0] else "")
        ax.legend(loc="lower right", frameon=True, facecolor="white",
                  edgecolor=LIGHT_GRAY, framealpha=.88, fontsize=7)
        clean_axes(ax)
    scope = "Pathway" if dataset == "pathway" else "Original LigandMPNN"
    kind = "Overall Protein" if metric == "overall" else "Ligand-pocket"
    fig.suptitle(f"{scope} Test · Paired {kind} Recovery", fontsize=17, y=1.01)
    save(fig, f"{dataset}_{metric}_recovery_paired_scatter_all_models")


def main():
    global ROOT, OUTPUT, DPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="evaluation root containing pathway/ and original_test/")
    parser.add_argument("--output-dir", type=Path,
                        help="figure directory (default: <root>/plots)")
    parser.add_argument("--summary-csv", type=Path,
                        help="compact figure source table (default: <output-dir>/all_models_plot_summary.csv)")
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()
    ROOT = args.root
    OUTPUT = args.output_dir or ROOT / "plots"
    DPI = args.dpi
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_style()
    effects = load_effects()
    summary_csv = args.summary_csv or OUTPUT / "all_models_plot_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(summary_csv, index=False)
    absolute_comparison(effects)
    effect_comparison(effects)
    for dataset in ("pathway", "original_test"):
        for metric in ("overall", "pocket"):
            violin_grid(dataset, metric)
            paired_scatter_grid(dataset, metric)
    print(f"saved plots to {OUTPUT}")


if __name__ == "__main__":
    main()
