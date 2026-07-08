import csv
import json
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import quantum_experiment_structures as qes
from quantum_experiment_structures.utils import utils

import profiling_utils

CHECK_CANDIDATES = ("all_checks",)
ADD_CANDIDATES = ("all_adds",)

DEFAULT_BASE_GENERATOR_SETTINGS = {
    "n_values_range": [2, 5],
    "n_contexts_range": [2, 30],
    "context_size_range": [2, 15],
    "n_alternatives_range": [1, 1],
    "enabling_relation_size_range": [1, 3],
    "n_samples_per_causal_structure": 1,
    "p_has_enabled": 0.7,
    "enabling_relation_size_mean": 1.5,
    "causally_secured": True,
    "n_scenarios": 1,
}


@dataclass
class BenchmarkRow:
    """A single row in the benchmark output table."""

    n_measurements: int
    repeat_index: int
    seed: int
    object_name: str
    stage: str
    mode: str
    method_name: str | None
    wall_time_s: float | None
    cpu_time_s: float | None
    json_size_bytes: int | None
    error: str | None = None
    extra_json: str = ""
    killed: bool = False
    timeout_budget_s: float | None = None
    attempt_index: int = 0
    observed_elapsed_s: float | None = None
    termination_reason: str | None = None

    def as_dict(self):
        """Converts the row to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TimedCallResult:
    value: Any = None
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    killed: bool = False
    error: str | None = None


def run_generator_case(generator):
    return list(generator.generate())


def run_ccs_checks(ccs):
    return ccs.all_checks()


def run_ccs_adds(ccs):
    return ccs.all_adds()


def run_ccs_to_spacetime(ccs):
    return ccs.to_spacetime_game()


def run_game_checks(game):
    return game.all_checks()


def run_game_adds(game):
    return game.all_adds()


def run_game_to_extensive(game):
    return game.to_extensive_game()


def _mode_to_operations(mode):
    if mode == "none":
        return []
    if mode == "checks":
        return ["checks"]
    if mode == "adds":
        return ["adds"]
    if mode == "both":
        return ["checks", "adds"]
    raise ValueError(f"Unknown mode: {mode}")


def _profile_method(
    object_name,
    stage,
    mode,
    obj,
    candidate_names,
    config,
    n_measurements,
    repeat_index,
    seed,
    timeout_trackers,
):
    method_name, bound_method = profiling_utils.resolve_first_callable(obj, candidate_names)

    prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key=f"{object_name}.{stage}",
        profile_name=f"{object_name}.{stage}",
        callable_factory=bound_method,
        config=config,
        timeout_trackers=timeout_trackers,
    )

    return BenchmarkRow(
        n_measurements=n_measurements,
        repeat_index=repeat_index,
        seed=seed,
        object_name=object_name,
        stage=stage,
        mode=mode,
        method_name=method_name,
        wall_time_s=prof.wall_time_s,
        cpu_time_s=prof.cpu_time_s,
        json_size_bytes=None,
        error=prof.error,
        killed=killed,
        timeout_budget_s=budget_s,
        attempt_index=attempt_index,
        observed_elapsed_s=prof.wall_time_s,
        termination_reason="timeout" if killed else "exception" if prof.error else None,
    )


def _record_size_row(
    n_measurements, repeat_index, seed, object_name, stage, mode, obj, data_attr="data"
):
    data = getattr(obj, data_attr, obj)
    return BenchmarkRow(
        n_measurements=n_measurements,
        repeat_index=repeat_index,
        seed=seed,
        object_name=object_name,
        stage=stage,
        mode=mode,
        method_name=None,
        wall_time_s=None,
        cpu_time_s=None,
        json_size_bytes=utils.json_size_bytes(data),
        error=None,
    )


def _run_adaptive_profile(stage_key, profile_name, callable_factory, config, timeout_trackers):
    """Run a callable with optional adaptive percentile-based timeout."""

    tracker = timeout_trackers.get(stage_key)
    if tracker is None:
        tracker = profiling_utils.RollingPercentileTracker(
            percentile=config.timeout_percentile,
            min_samples=config.timeout_min_samples,
            base_time_s=config.timeout_base_s,
            history_size=config.timeout_history_size,
            multiplier=config.timeout_multiplier,
        )
        timeout_trackers[stage_key] = tracker

    use_timeout = config.adaptive_timeout_enabled and config.timeout_percentile is not None

    if not use_timeout:
        prof = profiling_utils.profile_callable(
            profile_name,
            callable_factory,
            collect_garbage=config.collect_garbage,
        )
        if prof.error is None:
            tracker.observe_success(prof.wall_time_s)
        return prof, False, None, 0

    budget_s = tracker.budget_s()
    attempt_index = 0

    while True:
        attempt_index += 1
        timed_result = profiling_utils.profile_callable_with_timeout(
            callable_factory,
            timeout_s=budget_s,
        )

        if not timed_result.killed:
            tracker.observe_success(timed_result.wall_time_s)
            return timed_result, False, budget_s, attempt_index

        tracker.observe_kill()

        if not config.timeout_retry_same_input:
            return timed_result, True, budget_s, attempt_index


def run_single_case(config, n_measurements, repeat_index, timeout_trackers):
    """Run one scenario size and one repeat."""
    seed = config.base_seed + 10_000 * repeat_index + n_measurements
    generator_settings = dict(DEFAULT_BASE_GENERATOR_SETTINGS)
    generator_settings.update(config.generator.build_for_n(n_measurements, seed))

    rows = []

    generator = qes.CCSGenerator(**generator_settings)
    gen_prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key="generator.generate",
        profile_name="generator.generate",
        callable_factory=partial(run_generator_case, generator),
        config=config,
        timeout_trackers=timeout_trackers,
    )
    generated = getattr(gen_prof, "value", None)

    rows.append(
        BenchmarkRow(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="generator",
            stage="generate",
            mode="none",
            method_name=None,
            wall_time_s=gen_prof.wall_time_s,
            cpu_time_s=gen_prof.cpu_time_s,
            json_size_bytes=None,
            error=gen_prof.error,
            killed=killed,
            timeout_budget_s=budget_s,
            attempt_index=attempt_index,
            observed_elapsed_s=gen_prof.wall_time_s,
            termination_reason="timeout" if killed else "exception" if gen_prof.error else None,
        )
    )
    if generated is None:
        return rows

    # TODO: should we try to generate more data to get an average for the generation time too?
    ccs_data = generated[0].data
    ccs = qes.CausallySecuredScenario(ccs_data)

    # the ccs from the generator have been funneled through ccs.everything(), so no need for it here
    rows.append(
        _record_size_row(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="ccs",
            stage="size",
            mode=config.ccs_mode,
            obj=ccs,
        )
    )

    ccs_to_st_prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key="ccs.to_spacetime",
        profile_name="ccs.to_spacetime",
        callable_factory=partial(run_ccs_to_spacetime, ccs),
        config=config,
        timeout_trackers=timeout_trackers,
    )

    spacetime_raw = getattr(ccs_to_st_prof, "value", None)

    rows.append(
        BenchmarkRow(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="ccs",
            stage="to_spacetime",
            mode="none",
            method_name="to_spacetime_game",
            wall_time_s=ccs_to_st_prof.wall_time_s,
            cpu_time_s=ccs_to_st_prof.cpu_time_s,
            json_size_bytes=None,  # will be recorded as initial_size
            error=ccs_to_st_prof.error,
            killed=killed,
            timeout_budget_s=budget_s,
            attempt_index=attempt_index,
            observed_elapsed_s=gen_prof.wall_time_s,
            termination_reason="timeout" if killed else "exception" if gen_prof.error else None,
        )
    )
    if spacetime_raw is None:
        return rows

    game = qes.AlternatingSpacetimeGame(spacetime_raw)
    rows.append(
        _record_size_row(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="game",
            stage="initial_size",
            mode="none",
            obj=game,
        )
    )

    for operation in _mode_to_operations(config.game_mode):
        if operation == "checks":
            rows.append(
                _profile_method(
                    object_name="game",
                    stage="checks",
                    mode=config.game_mode,
                    obj=game,
                    candidate_names=CHECK_CANDIDATES,
                    config=config,
                    n_measurements=n_measurements,
                    repeat_index=repeat_index,
                    seed=seed,
                    timeout_trackers=timeout_trackers,
                )
            )
        elif operation == "adds":
            rows.append(
                _profile_method(
                    object_name="game",
                    stage="adds",
                    mode=config.game_mode,
                    obj=game,
                    candidate_names=ADD_CANDIDATES,
                    config=config,
                    n_measurements=n_measurements,
                    repeat_index=repeat_index,
                    seed=seed,
                    timeout_trackers=timeout_trackers,
                )
            )

    rows.append(
        _record_size_row(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="game",
            stage="final_size",
            mode=config.game_mode,
            obj=game,
        )
    )

    game_to_ext_prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key="game.to_extensive",
        profile_name="game.to_extensive",
        callable_factory=partial(run_game_to_extensive, game),
        config=config,
        timeout_trackers=timeout_trackers,
    )

    extensive = getattr(game_to_ext_prof, "value", None)

    rows.append(
        BenchmarkRow(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="game",
            stage="to_extensive",
            mode="none",
            method_name="to_extensive_game",
            wall_time_s=game_to_ext_prof.wall_time_s,
            cpu_time_s=game_to_ext_prof.cpu_time_s,
            json_size_bytes=None,
            error=game_to_ext_prof.error,
            killed=killed,
            timeout_budget_s=budget_s,
            attempt_index=attempt_index,
            observed_elapsed_s=gen_prof.wall_time_s,
            termination_reason="timeout" if killed else "exception" if gen_prof.error else None,
        )
    )
    if extensive is None:
        return rows

    rows.append(
        BenchmarkRow(
            n_measurements=n_measurements,
            repeat_index=repeat_index,
            seed=seed,
            object_name="extensive",
            stage="size",
            mode="",
            method_name=None,
            wall_time_s=None,
            cpu_time_s=None,
            json_size_bytes=utils.json_size_bytes(extensive),
            error=None,
        )
    )

    return rows


def compute_correlations(rows):
    """Compute correlations between JSON size and runtime."""
    df = pd.DataFrame(row.as_dict() for row in rows)

    run_cols = ["n_measurements", "repeat_index", "seed"]

    size_source_map = {
        ("generator", "generate"): "ccs_final",
        ("ccs", "to_spacetime"): "game_initial",
        ("game", "checks"): "game_final",
        ("game", "adds"): "game_final",
        ("game", "to_extensive"): "extensive",
    }

    size_key_map = {
        ("ccs", "size"): "ccs_final",
        ("game", "initial_size"): "game_initial",
        ("game", "final_size"): "game_final",
        ("extensive", "size"): "extensive",
    }

    size_df = df[df["json_size_bytes"].notna()].copy()
    size_df["size_source"] = size_df.apply(
        lambda row: size_key_map.get((row["object_name"], row["stage"])),
        axis=1,
    )
    size_df = size_df[size_df["size_source"].notna()].copy()

    size_df = size_df.groupby(run_cols + ["size_source"], dropna=False, as_index=False)[
        "json_size_bytes"
    ].first()

    runtime_df = df[df["wall_time_s"].notna() & (~df["killed"])].copy()
    runtime_df["size_source"] = runtime_df.apply(
        lambda row: size_source_map.get((row["object_name"], row["stage"])),
        axis=1,
    )
    runtime_df = runtime_df[runtime_df["size_source"].notna()].copy()

    runtime_df = runtime_df.merge(
        size_df,
        on=run_cols + ["size_source"],
        how="left",
        suffixes=("", "_added"),
    )

    runtime_df = runtime_df.rename(columns={"json_size_bytes_added": "size_bytes"})

    runtime_df["size_bytes"] = pd.to_numeric(runtime_df["size_bytes"], errors="coerce")
    runtime_df["wall_time_s"] = pd.to_numeric(runtime_df["wall_time_s"], errors="coerce")
    runtime_df["cpu_time_s"] = pd.to_numeric(runtime_df["cpu_time_s"], errors="coerce")

    runtime_df = runtime_df.replace([np.inf, -np.inf], np.nan)
    runtime_df = runtime_df.dropna(subset=["size_bytes", "wall_time_s", "cpu_time_s"]).copy()

    def _pearson(x, y):
        if len(x) < 2:
            return float("nan")
        if np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(x, y):
        if len(x) < 2:
            return float("nan")
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        if np.all(rx == rx[0]) or np.all(ry == ry[0]):
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    summaries = []

    for (object_name, stage, mode), group in runtime_df.groupby(
        ["object_name", "stage", "mode"],
        dropna=False,
        sort=False,
    ):
        x_size = group["size_bytes"].to_numpy(dtype=float)
        wall = group["wall_time_s"].to_numpy(dtype=float)
        cpu = group["cpu_time_s"].to_numpy(dtype=float)

        # Spearman is probably better here because ther is very high variance (outliers)
        # Spearman > Pearson => monotonic but non-linear correlation
        pearson_wall = _pearson(x_size, wall)
        spearman_wall = _spearman(x_size, wall)
        pearson_cpu = _pearson(x_size, cpu)
        spearman_cpu = _spearman(x_size, cpu)

        # log-log to see how the scaling is correlated
        log_mask_wall = (x_size > 0) & (wall > 0)
        log_mask_cpu = (x_size > 0) & (cpu > 0)

        log_pearson_wall = (
            _pearson(np.log10(x_size[log_mask_wall]), np.log10(wall[log_mask_wall]))
            if log_mask_wall.sum() >= 2
            else float("nan")
        )
        log_spearman_wall = (
            _spearman(np.log10(x_size[log_mask_wall]), np.log10(wall[log_mask_wall]))
            if log_mask_wall.sum() >= 2
            else float("nan")
        )

        log_pearson_cpu = (
            _pearson(np.log10(x_size[log_mask_cpu]), np.log10(cpu[log_mask_cpu]))
            if log_mask_cpu.sum() >= 2
            else float("nan")
        )
        log_spearman_cpu = (
            _spearman(np.log10(x_size[log_mask_cpu]), np.log10(cpu[log_mask_cpu]))
            if log_mask_cpu.sum() >= 2
            else float("nan")
        )

        summaries.append(
            {
                "object_name": object_name,
                "stage": stage,
                "mode": mode,
                "samples": len(group),
                "killed_count": int(group["killed"].sum()),
                "size_source": group["size_source"].iloc[0],
                "pearson_size_wall": pearson_wall,
                "spearman_size_wall": spearman_wall,
                "pearson_size_cpu": pearson_cpu,
                "spearman_size_cpu": spearman_cpu,
                "log_pearson_size_wall": log_pearson_wall,
                "log_spearman_size_wall": log_spearman_wall,
                "log_pearson_size_cpu": log_pearson_cpu,
                "log_spearman_size_cpu": log_spearman_cpu,
            }
        )

    return pd.DataFrame(summaries), runtime_df


def run_benchmark(config):
    """Run the full benchmark sweep."""
    config.validate()
    config.ensure_dirs()

    all_rows = []
    timeout_trackers = dict()

    try:
        for n_measurements in range(config.n_min, config.n_max + 1, config.n_step):
            print(f"Profiling scenarios with {n_measurements=} variables.")
            for repeat_index in range(config.repeats):
                print(f"Repetition {repeat_index + 1}/{config.repeats}.")
                rows = run_single_case(config, n_measurements, repeat_index, timeout_trackers)
                all_rows.extend(rows)
    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user (SIGINT). Returning collected results so far...")

    return all_rows


def save_results(rows, config):
    """Save results to CSV and JSONL."""
    config.ensure_dirs()

    fieldnames = list(rows[0].as_dict().keys()) if rows else []

    with config.output_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())

    with config.output_jsonl.open("w", encoding="utf-8") as f_jsonl:
        for row in rows:
            f_jsonl.write(json.dumps(row.as_dict()) + "\n")

    correlation_df, sizes_merged_df = compute_correlations(rows)
    sizes_merged_df.to_csv(config.output_dir / "sizes.csv", index=False)
    correlation_df.to_csv(config.output_dir / "correlations.csv", index=False)
