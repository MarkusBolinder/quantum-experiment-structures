from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aggregate(df, metric):
    """Aggregate mean and std by n, stage, and mode."""
    grouped = (
        df.groupby(["n_measurements", "object_name", "stage", "mode"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped


def _format_label(object_name, stage, mode):
    """Format a compact legend label."""
    label_map = {
        ("ccs", "to_spacetime", "none"): "Converting CCS to spacetime game",
        (
            "game",
            "adds",
            "both",
        ): "Adding (complete) histories and (reduced) strategies to spacetime game",
        ("game", "checks", "both"): "Running all correctness checks on spacetime game",
        ("game", "to_extensive", "none"): "Converting spacetime game to extensive form game",
        ("generator", "generate", "none"): "Generating (causally secured) CCS",
        ("ccs", "size", "both"): "Size of (causally secured) CCS",
        ("extensive", "size", "nan"): "Size of extensive form game",
        (
            "game",
            "final_size",
            "both",
        ): "Size of spacetime game after adding histories and strategies",
        ("game", "final_size", "none"): "Size of spacetime game (without histories and strategies)",
        ("game", "initial_size", "none"): "Initial size of spacetime game",
    }
    label = label_map.get((object_name, stage, str(mode)))
    if label is not None:
        return label
    else:
        return input(f"Label for {(object_name, stage, mode)=}: ")


def plot_metric(df, metric, title, ylabel, output_path, logy=False):
    """Plot mean and std of a metric across measurement counts."""
    df = df[df[metric].notna() & (~df["killed"])].copy()
    agg = _aggregate(df, metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    labels = []

    for (object_name, stage, mode), group in agg.groupby(
        ["object_name", "stage", "mode"], dropna=False
    ):
        group = group.sort_values("n_measurements")
        label = _format_label(object_name, stage, mode)
        x = group["n_measurements"].to_numpy()
        y = group["mean"].to_numpy()
        yerr = group["std"].fillna(0.0).to_numpy()

        container = ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        handles.append(container)
        labels.append(label)

    ax.set_title(title)
    ax.set_xlabel("Number of measurements")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_size_time_correlation(df, output_path, metric="wall_time_s", log_axes=True):
    """Create scatter plot of JSON size vs runtime."""

    plot_df = df[
        df["size_bytes"].notna()
        & df[metric].notna()
        & (~df["killed"])
        & (df["size_bytes"] > 0)
        & (df[metric] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    size_label_map = {
        ("generator", "generate"): "ccs",
        ("ccs", "to_spacetime"): "game_initial",
        ("game", "checks"): "game_final",
        ("game", "adds"): "game_final",
        ("game", "to_extensive"): "extensive",
        ("ccs", "size"): "ccs_final",
        ("game", "initial_size"): "game_initial",
        ("game", "final_size"): "game_final",
        ("extensive", "size"): "extensive",
    }

    for (object_name, stage, mode), group in plot_df.groupby(
        ["object_name", "stage", "mode"],
        dropna=False,
    ):
        if stage == "checks":
            continue
        ax.scatter(
            group["size_bytes"],
            group[metric],
            label=size_label_map[(object_name, stage)],
            alpha=0.7,
        )

        if len(group) >= 2:
            x = group["size_bytes"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)

            if log_axes:
                x_fit = np.log10(x)
                y_fit = np.log10(y)
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = 10 ** (intercept + slope * np.log10(x_line))
            else:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = intercept + slope * x_line

            ax.plot(x_line, y_line, linewidth=1)

    ax.set_xlabel("JSON size (bytes)")
    ax.set_ylabel(metric)
    ax.set_title(f"JSON size vs {metric}")

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_results(csv_path, output_dir):
    """Create standard profiling plots from a benchmark CSV."""
    df = pd.read_csv(csv_path)
    corr_df = pd.read_csv(Path(*csv_path.parts[:-1]) / "sizes.csv")

    numeric_rows = df[df["wall_time_s"].notna()].copy()
    plot_metric(
        numeric_rows,
        "wall_time_s",
        "Wall time scaling",
        "Wall time (s)",
        output_dir / "wall_time_scaling.png",
        logy=True,
    )

    plot_metric(
        numeric_rows,
        "cpu_time_s",
        "CPU time scaling",
        "CPU time (s)",
        output_dir / "cpu_time_scaling.png",
        logy=True,
    )

    size_rows = df[df["json_size_bytes"].notna()].copy()
    plot_metric(
        size_rows,
        "json_size_bytes",
        "Serialized size scaling",
        "JSON size bytes",
        output_dir / "json_size_scaling.png",
        logy=True,
    )

    plot_size_time_correlation(
        corr_df,
        output_dir / "json_size_vs_wall_time.png",
        metric="wall_time_s",
        log_axes=True,
    )
    plot_size_time_correlation(
        corr_df,
        output_dir / "json_size_vs_cpu_time.png",
        metric="cpu_time_s",
        log_axes=True,
    )
