from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# omit brown from being used
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(  # type: ignore
    color=[
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
)


LABEL_MAP = {
    # ccs
    ("ccs", "to_spacetime", "none", "cpu_time_s"): "Converting CCS to spacetime game",
    ("ccs", "size", "both", "json_size_bytes"): "Size of (causally secured) CCS",
    # only the cleanliness check is not performed by the generator
    ("ccs", "checks", "both", "cpu_time_s"): "Checking CCS cleanliness",
    ("ccs", "is_scenario_clean", "both", "cpu_time_s"): "Checking CCS is clean",
    (
        "ccs",
        "transitively_close_enabling_relations",
        "both",
        "cpu_time_s",
    ): "Transitively closing CCS",
    # spacetime games
    ("game", "add_histories", "both", "cpu_time_s"): "Adding histories",
    ("game", "add_played_information_sets", "both", "cpu_time_s"): "Adding played information sets",
    (
        "game",
        "add_histories_and_played_info_sets",
        "both",
        "json_size_bytes",
    ): "Size of spacetime game after adding only histories",
    (
        "game",
        "add_reduced_strategies",
        "both",
        "json_size_bytes",
    ): "Size of spacetime game after adding histories and strategies",
    ("game", "check_histories_consistency", "both", "cpu_time_s"): "Checking history consistency",
    (
        "game",
        "check_information_sets_consistency",
        "both",
        "cpu_time_s",
    ): "Checking information set consistency",
    (
        "game",
        "check_reduced_strategies_consistency",
        "both",
        "cpu_time_s",
    ): "Checking reduced strategies consistency",
    ("game", "check_no_cycles", "both", "cpu_time_s"): "Checking no cycles",
    ("game", "check_node_graph_integrity", "both", "cpu_time_s"): "Checking spacetime game DAG",
    ("game", "check_totality_and_cototality", "both", "cpu_time_s"): "Checking totality/cototality",
    ("game", "adds", "both", "cpu_time_s"): "Adding histories and strategies",
    ("game", "checks", "both", "cpu_time_s"): "All spacetime game checks",
    ("game", "to_extensive", "none", "cpu_time_s"): "Converting spacetime to extensive form game",
    (
        "game",
        "final_size",
        "none",
        "json_size_bytes",
    ): "Size of spacetime game (w/o histories and strategies)",
    ("game", "initial_size", "none", "json_size_bytes"): "Initial size of spacetime game",
    ("game", "add_final_size", "both", "json_size_bytes"): "Final size of spacetime game",
    ("game", "final_size", "both", "json_size_bytes"): "Final size of spacetime game",
    ("game", "adds", "both", "json_size_bytes"): "Final size of spacetime game",
    (
        "game",
        "add_reduced_strategies_isolated",
        "both",
        "json_size_bytes",
    ): "Size of spacetime game after adding only strategies",
    (
        "game",
        "add_histories_and_played_info_sets",
        "both",
        "cpu_time_s",
    ): "Adding histories and played information sets",
    ("game", "add_reduced_strategies_isolated", "both", "cpu_time_s"): "Adding reduced strategies",
    # generator
    ("generator", "generate", "none", "cpu_time_s"): "Generating (causally secured) CCS",
    # extensive form games
    ("extensive", "size", "nan", "json_size_bytes"): "Size of extensive form game",
    ("extensive", "size", "", "json_size_bytes"): "Size of extensive form game",
    # random thing
    (
        "game",
        "aggregated_alternating_checks",
        "both",
        "cpu_time_s",
    ): "Aggregated alternating checks",
    ("game", "some_aggregated_game_checks", "both", "cpu_time_s"): "Other fast checks",
    # checks for AlternatingSpacetimeGame
    ("game", "check_2_players", "both", "cpu_time_s"): "Checking two players",
    ("game", "check_ab1", "both", "cpu_time_s"): "Checking AB1",
    ("game", "check_ab2", "both", "cpu_time_s"): "Checking AB2",
    ("game", "check_ba1", "both", "cpu_time_s"): "Checking BA1",
    ("game", "check_ba2", "both", "cpu_time_s"): "Checking BA2",
    ("game", "check_ba3", "both", "cpu_time_s"): "Checking BA3",
    ("game", "check_bipartite", "both", "cpu_time_s"): "Checking bipartiteness",
    ("game", "check_bob_a", "both", "cpu_time_s"): "Checking Bob-A",
    ("game", "check_roots_and_leaves", "both", "cpu_time_s"): "Checking roots and leaves",
    (
        "game",
        "check_singleton_bob_info_sets",
        "both",
        "cpu_time_s",
    ): "Checking singleton Bob information sets",
    ("game", "check_even_height", "both", "cpu_time_s"): "Checking even height",
}


def _aggregate(df, metric):
    """Aggregate mean and std by n, stage, and mode."""
    grouped = (
        df.groupby(["n_measurements", "object_name", "stage", "mode"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped


def _format_label(object_name, stage, mode, metric):
    """Format a compact legend label safely without blocking on stdin."""
    # Normalize potential NaN values or empty strings to "nan"
    mode_str = "nan" if pd.isna(mode) or str(mode).strip() in ("", "nan", "None") else str(mode)

    label = LABEL_MAP.get((object_name, stage, mode_str, metric))
    if label is not None:
        return label
    return input(f"Label for {(object_name, stage, mode, metric)=}: ")

    # Check safe fallbacks to bypass subtle naming mismatches
    for fallback in ["both", "none", "nan", ""]:
        label = LABEL_MAP.get((object_name, stage, fallback))
        if label is not None:
            return label

    return f"{object_name} ({stage})"


def plot_metric(df, metric, title, ylabel, output_path, logy=False, granular=False):
    """Plot mean and std of a metric across measurement counts."""
    df = df[df[metric].notna() & (~df["killed"])].copy()

    alternating_checks = {
        "check_2_players",
        "check_ab1",
        "check_ab2",
        "check_ba1",
        "check_ba2",
        "check_ba3",
        "check_bipartite",
        "check_bob_a",
        "check_even_height",
        "check_roots_and_leaves",
        "check_singleton_bob_info_sets",
    }
    other_fast_checks = {
        "check_information_sets_consistency",
        "check_no_cycles",
        "check_node_graph_integrity",
        "check_totality_and_cototality",
    }

    if granular:
        # 1. Eliminate redundant stages for size plots to keep visual output clean
        if metric == "json_size_bytes":
            allowed_stages = {
                "initial_size",
                "add_histories",  # Will be merged to add_histories_and_played_info_sets
                "add_played_information_sets",  # Will be merged to add_histories_and_played_info_sets
                "add_reduced_strategies_isolated",
                "add_reduced_strategies",
                "final_size",
                "size",  # Extensive game size
            }
            df = df[
                (~df["object_name"].isin(["game", "extensive"]))
                | (df["stage"].isin(allowed_stages))
            ]

        # 2. Construct group indices to track matching execution runs
        groupby_cols = ["object_name", "stage", "mode"]
        if "n_measurements" in df.columns:
            groupby_cols.insert(0, "n_measurements")

        df["run_id"] = df.groupby(groupby_cols, dropna=False).cumcount()

        # 3. Aggregate checks and combine history / info set stages
        df.loc[df["stage"].isin(alternating_checks), "stage"] = "aggregated_alternating_checks"
        df.loc[df["stage"].isin(other_fast_checks), "stage"] = "some_aggregated_game_checks"
        df.loc[df["stage"].isin(["add_histories", "add_played_information_sets"]), "stage"] = (
            "add_histories_and_played_info_sets"
        )

        # 4. Collapse metrics appropriately (sum times / max sizes)
        if metric == "json_size_bytes":
            df = df.groupby(groupby_cols + ["run_id"], dropna=False)[metric].max().reset_index()
        else:
            df = df.groupby(groupby_cols + ["run_id"], dropna=False)[metric].sum().reset_index()

    if granular:
        plot_groups = [
            (df[df["object_name"].isin(["generator", "ccs"])], "_generation_ccs"),
            (df[df["object_name"].isin(["game", "extensive"])], "_spacetime_extensive"),
        ]
    else:
        plot_groups = [(df, "")]

    for sub_df, suffix in plot_groups:
        if sub_df.empty:
            continue
        agg = _aggregate(sub_df, metric)
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        handles = []
        labels = []
        min_n = 10**9
        max_n = 0
        MAX_TICKS = 25

        for (object_name, stage, mode), group in agg.groupby(
            ["object_name", "stage", "mode"], dropna=False
        ):
            group = group.sort_values("n_measurements")
            label = _format_label(object_name, stage, mode, metric)
            x = group["n_measurements"].to_numpy()
            max_n = np.max(x)
            min_n = np.min(x)
            y = group["mean"].to_numpy()
            yerr = group["std"].fillna(0.0).to_numpy()

            container = ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
            handles.append(container)
            labels.append(label)

        ax.set_title(title)
        ax.set_xlabel("Number of measurements")
        n_in_range = int(max_n - min_n + 1)
        step_size = max(n_in_range // MAX_TICKS, 1)
        plt.xticks(range(int(min_n), int(max_n) + 1, step_size))
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        ax.legend(handles, labels, fontsize=8, loc="best")
        fig.tight_layout()

        actual_output_path = (
            output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
            if suffix
            else output_path
        )
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(actual_output_path, dpi=200)
        plt.close(fig)


def plot_size_time_correlation(
    df, output_path, metric="wall_time_s", log_axes=True, granular=False
):
    """Create scatter plot of JSON size vs runtime."""

    plot_df = df[
        df["size_bytes"].notna()
        & df[metric].notna()
        & (~df["killed"])
        & (df["size_bytes"] > 0)
        & (df[metric] > 0)
    ].copy()

    alternating_checks = {
        "check_2_players",
        "check_ab1",
        "check_ab2",
        "check_ba1",
        "check_ba2",
        "check_ba3",
        "check_bipartite",
        "check_bob_a",
        "check_even_height",
        "check_roots_and_leaves",
        "check_singleton_bob_info_sets",
    }

    if not granular:
        groupby_cols = ["object_name", "stage", "mode"]
        if "n_measurements" in plot_df.columns:
            groupby_cols.insert(0, "n_measurements")

        plot_df["run_id"] = plot_df.groupby(groupby_cols, dropna=False).cumcount()
        plot_df.loc[plot_df["stage"].isin(alternating_checks), "stage"] = (
            "aggregated_alternating_checks"
        )

        # sum both the file sizes and metrics for the same run iteration
        plot_df = (
            plot_df.groupby(groupby_cols + ["run_id"], dropna=False)[["size_bytes", metric]]
            .sum()
            .reset_index()
        )

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
        ("game", "aggregated_alternating_checks"): "game_final",
        ("game", "add_histories_and_played_info_sets"): "game_history",
        ("game", "add_reduced_strategies_isolated"): "game_strategies_only",
        ("game", "add_reduced_strategies"): "game_final",
    }

    for (object_name, stage, mode), group in plot_df.groupby(
        ["object_name", "stage", "mode"],
        dropna=False,
    ):
        if stage == "checks":
            continue

        label = size_label_map.get((object_name, stage), f"{object_name}_{stage}")
        ax.scatter(
            group["size_bytes"],
            group[metric],
            label=label,
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


def plot_results(csv_path, output_dir, granular=False):
    """Create standard profiling plots from a benchmark CSV."""
    df = pd.read_csv(csv_path)
    corr_df = pd.read_csv(Path(*csv_path.parts[:-1]) / "sizes.csv")

    numeric_rows = df[df["wall_time_s"].notna()].copy()

    plot_metric(
        numeric_rows,
        "cpu_time_s",
        "CPU time scaling",
        "CPU time (s)",
        output_dir / "cpu_time_scaling.png",
        logy=True,
        granular=granular,
    )

    size_rows = df[df["json_size_bytes"].notna()].copy()
    plot_metric(
        size_rows,
        "json_size_bytes",
        "Serialized size scaling",
        "JSON size bytes",
        output_dir / "json_size_scaling.png",
        logy=True,
        granular=granular,
    )

    plot_size_time_correlation(
        corr_df,
        output_dir / "json_size_vs_cpu_time.png",
        metric="cpu_time_s",
        log_axes=True,
        granular=granular,
    )
