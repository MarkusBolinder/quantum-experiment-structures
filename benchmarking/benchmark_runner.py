from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from functools import partial
import gc
from typing import Any

import numpy as np
import pandas as pd
import quantum_experiment_structures as qes
from quantum_experiment_structures.utils import utils

import profiling_utils


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


def run_ccs_to_spacetime(ccs):
    return ccs.to_spacetime_game()


def run_game_to_extensive(game):
    return game.to_extensive_game(match_utility=False)


def _mode_to_operations(mode):
    if mode == "none":
        return []
    if mode == "checks":
        return ["checks"]
    if mode == "adds":
        return ["adds"]
    if mode == "both":
        # adds should happen before checks
        return ["adds", "checks"]
    raise ValueError(f"Unknown mode: {mode}")


def _get_methods_to_profile(obj, operation_type, granular=False):
    """Discover methods to profile on an object dynamically.

    Returns:
        a list of tuples: (method_name, bound_method)
    """
    methods = []
    if operation_type == "none":
        return methods
    names = []
    is_scenario = obj.__class__.__name__ == "CausallySecuredScenario"
    # always aggregate checks
    if operation_type == "checks":
        if is_scenario:
            methods.append(("is_scenario_clean", getattr(obj, "is_scenario_clean")))
        elif granular:
            names = [name for name in dir(obj) if name.startswith("check_")]
            # TODO: somehow aggregate all checks for AlternatingSpacetimeGame
        else:
            methods.append(("all_checks", getattr(obj, "all_checks")))
    elif granular:
        if not is_scenario:
            # force order: histories, played info sets, reduced strategies
            names = sorted([name for name in dir(obj) if name.startswith("add_")])
        else:
            names = ["transitively_close_enabling_relations"]
    elif not is_scenario:
        # aggregated behavior
        methods.append(("all_adds", getattr(obj, "all_adds")))
    # add all actual methods
    for name in names:
        attr = getattr(obj, name)
        if callable(attr):
            methods.append((name, attr))

    return methods


def _profile_method(
    object_name,
    stage,
    mode,
    obj,
    method_name,
    bound_method,
    config,
    n_measurements,
    repeat_index,
    seed,
    timeout_trackers,
    lock=None,
    record_size=False,
):
    # unique stage_key allows independent adaptive timeout ceilings per method
    prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key=f"{object_name}.{stage}.{method_name}",
        profile_name=f"{object_name}.{stage}.{method_name}",
        callable_factory=bound_method,
        config=config,
        timeout_trackers=timeout_trackers,
        lock=lock,
    )

    json_size = None
    if record_size and not killed and prof.error is None:
        data = getattr(obj, "data", obj)
        json_size = utils.json_size_bytes(data)

    wall_time_extracted = prof.wall_time_s
    cpu_time_extracted = prof.cpu_time_s
    error_extracted = prof.error

    del prof
    if config.collect_garbage:
        gc.collect()

    return BenchmarkRow(
        n_measurements=n_measurements,
        repeat_index=repeat_index,
        seed=seed,
        object_name=object_name,
        stage=stage,
        mode=mode,
        method_name=method_name,
        wall_time_s=wall_time_extracted,
        cpu_time_s=cpu_time_extracted,
        json_size_bytes=json_size,
        error=error_extracted,
        killed=killed,
        timeout_budget_s=budget_s,
        attempt_index=attempt_index,
        observed_elapsed_s=wall_time_extracted,
        termination_reason="timeout" if killed else "exception" if error_extracted else None,
    )


def _run_adaptive_profile(
    stage_key, profile_name, callable_factory, config, timeout_trackers, lock=None
):
    """Run a callable with optional adaptive percentile-based timeout."""
    if lock:
        lock.acquire()
    try:
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
        budget_s = tracker.budget_s()
    finally:
        if lock:
            lock.release()

    use_timeout = config.adaptive_timeout_enabled and config.timeout_percentile is not None

    if not use_timeout:
        prof = profiling_utils.profile_callable(
            profile_name,
            callable_factory,
            collect_garbage=config.collect_garbage,
        )
        if prof.error is None:
            if lock:
                lock.acquire()
            try:
                tracker = timeout_trackers.get(stage_key)
                tracker.observe_success(prof.wall_time_s)
                timeout_trackers[stage_key] = tracker
            finally:
                if lock:
                    lock.release()
        return prof, False, None, 0

    attempt_index = 0
    while True:
        attempt_index += 1
        timed_result = profiling_utils.profile_callable_with_timeout(
            callable_factory,
            timeout_s=budget_s,
        )

        if lock:
            lock.acquire()
        try:
            tracker = timeout_trackers.get(stage_key)
            if not timed_result.killed:
                tracker.observe_success(timed_result.wall_time_s)
                timeout_trackers[stage_key] = tracker
                return timed_result, False, budget_s, attempt_index

            tracker.observe_kill()
            timeout_trackers[stage_key] = tracker
        finally:
            if lock:
                lock.release()

        if not config.timeout_retry_same_input:
            return timed_result, True, budget_s, attempt_index


def run_single_case(config, n_measurements, repeat_index, timeout_trackers, lock=None):
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
        lock=lock,
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
    gen_prof = None
    if config.collect_garbage:
        gc.collect()
    if generated is None:
        return rows

    ccs_data = generated[0].data
    ccs = qes.CausallySecuredScenario(ccs_data)
    del generated
    if config.collect_garbage:
        gc.collect()

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

    # profile CCS active operations based on config.ccs_mode
    for operation in _mode_to_operations(config.ccs_mode):
        methods = _get_methods_to_profile(ccs, operation, granular=config.granular)
        for method_name, bound_method in methods:
            rows.append(
                _profile_method(
                    object_name="ccs",
                    stage=method_name if config.granular else operation,
                    mode=config.ccs_mode,
                    obj=ccs,
                    method_name=method_name,
                    bound_method=bound_method,
                    config=config,
                    n_measurements=n_measurements,
                    repeat_index=repeat_index,
                    seed=seed,
                    timeout_trackers=timeout_trackers,
                    lock=lock,
                    record_size=False,
                )
            )

    ccs_to_st_prof, killed, budget_s, attempt_index = _run_adaptive_profile(
        stage_key="ccs.to_spacetime",
        profile_name="ccs.to_spacetime",
        callable_factory=partial(run_ccs_to_spacetime, ccs),
        config=config,
        timeout_trackers=timeout_trackers,
        lock=lock,
    )

    spacetime_raw = getattr(ccs_to_st_prof, "value", None)
    ccs = None
    if config.collect_garbage:
        gc.collect()

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
            json_size_bytes=None,
            error=ccs_to_st_prof.error,
            killed=killed,
            timeout_budget_s=budget_s,
            attempt_index=attempt_index,
            observed_elapsed_s=ccs_to_st_prof.wall_time_s,
            termination_reason="timeout"
            if killed
            else "exception"
            if ccs_to_st_prof.error
            else None,
        )
    )
    ccs_to_st_prof = None
    if config.collect_garbage:
        gc.collect()
    if spacetime_raw is None:
        return rows

    game = qes.AlternatingSpacetimeGame(spacetime_raw)
    del spacetime_raw
    if config.collect_garbage:
        gc.collect()

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
        granular_adding = config.granular and operation == "adds"

        if granular_adding:
            methods_dict = dict(_get_methods_to_profile(game, operation, granular=True))
            # add histories and played info sets
            game_hist = deepcopy(game)

            if "add_histories" in methods_dict:
                rows.append(
                    _profile_method(
                        object_name="game",
                        stage="add_histories",
                        mode=config.game_mode,
                        obj=game_hist,
                        method_name="add_histories",
                        bound_method=getattr(game_hist, "add_histories"),
                        config=config,
                        n_measurements=n_measurements,
                        repeat_index=repeat_index,
                        seed=seed,
                        timeout_trackers=timeout_trackers,
                        lock=lock,
                        record_size=True,
                    )
                )

            if "add_played_information_sets" in methods_dict:
                rows.append(
                    _profile_method(
                        object_name="game",
                        stage="add_played_information_sets",
                        mode=config.game_mode,
                        obj=game_hist,
                        method_name="add_played_information_sets",
                        bound_method=getattr(game_hist, "add_played_information_sets"),
                        config=config,
                        n_measurements=n_measurements,
                        repeat_index=repeat_index,
                        seed=seed,
                        timeout_trackers=timeout_trackers,
                        lock=lock,
                        record_size=True,
                    )
                )

            # handle strategies separately
            game_strat = deepcopy(game)
            if "add_reduced_strategies" in methods_dict:
                rows.append(
                    _profile_method(
                        object_name="game",
                        stage="add_reduced_strategies_isolated",
                        mode=config.game_mode,
                        obj=game_strat,
                        method_name="add_reduced_strategies",
                        bound_method=getattr(game_strat, "add_reduced_strategies"),
                        config=config,
                        n_measurements=n_measurements,
                        repeat_index=repeat_index,
                        seed=seed,
                        timeout_trackers=timeout_trackers,
                        lock=lock,
                        record_size=True,
                    )
                )

            # combine strategies and histories into the same game
            game_combined = game_hist
            game_combined.data["rs"] = game_strat.data["rs"]
            game = game_combined
            del game_hist
            del game_strat
            if config.collect_garbage:
                gc.collect()

        else:
            # if not granular, just process normally
            methods = _get_methods_to_profile(game, operation, granular=config.granular)
            for method_name, bound_method in methods:
                rows.append(
                    _profile_method(
                        object_name="game",
                        stage=method_name if config.granular else operation,
                        mode=config.game_mode,
                        obj=game,
                        method_name=method_name,
                        bound_method=bound_method,
                        config=config,
                        n_measurements=n_measurements,
                        repeat_index=repeat_index,
                        seed=seed,
                        timeout_trackers=timeout_trackers,
                        lock=lock,
                        record_size=False,
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
        lock=lock,
    )

    extensive = getattr(game_to_ext_prof, "value", None)
    game = None
    if config.collect_garbage:
        gc.collect()

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
            observed_elapsed_s=game_to_ext_prof.wall_time_s,
            termination_reason="timeout"
            if killed
            else "exception"
            if game_to_ext_prof.error
            else None,
        )
    )
    game_to_ext_prof = None
    if config.collect_garbage:
        gc.collect()
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
    del extensive
    if config.collect_garbage:
        gc.collect()

    return rows


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


def compute_correlations(rows):
    """Compute correlations between JSON size and runtime."""
    df = pd.DataFrame(row.as_dict() for row in rows)

    run_cols = ["n_measurements", "repeat_index", "seed"]

    size_source_map = {
        ("generator", "generate"): "ccs_final",
        ("ccs", "checks"): "ccs_final",
        ("ccs", "adds"): "ccs_final",
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
    size_df = size_df[size_df["size_source"].notna()].copy()  # type: ignore

    size_df = size_df.groupby(  # type: ignore
        run_cols + ["size_source"], dropna=False, as_index=False
    )["json_size_bytes"].first()

    runtime_df = df[df["wall_time_s"].notna() & (~df["killed"])].copy()
    runtime_df["size_source"] = runtime_df.apply(
        lambda row: size_source_map.get((row["object_name"], row["stage"])),
        axis=1,
    )
    runtime_df = runtime_df[runtime_df["size_source"].notna()].copy()  # type: ignore

    runtime_df = runtime_df.merge(  # type: ignore
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

    for (object_name, stage, mode), group in runtime_df.groupby(  # type: ignore
        ["object_name", "stage", "mode"],
        dropna=False,
        sort=False,
    ):
        x_size = group["size_bytes"].to_numpy(dtype=float)
        wall = group["wall_time_s"].to_numpy(dtype=float)
        cpu = group["cpu_time_s"].to_numpy(dtype=float)

        pearson_wall = _pearson(x_size, wall)
        spearman_wall = _spearman(x_size, wall)
        pearson_cpu = _pearson(x_size, cpu)
        spearman_cpu = _spearman(x_size, cpu)

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
                "killed_count": int(group["killed"].sum()),  # type: ignore
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


def _worker_task(task_args):
    """Global level bridge function for ProcessPoolExecutor mapping."""
    config, n_measurements, repeat_index, timeout_trackers, lock = task_args
    return run_single_case(config, n_measurements, repeat_index, timeout_trackers, lock)


def run_benchmark(config, num_workers=1):
    """Run the full benchmark sweep sequentially or in parallel."""
    config.validate()
    config.ensure_dirs()

    all_rows = []

    if num_workers <= 1:
        timeout_trackers = dict()
        try:
            for n_measurements in range(config.n_min, config.n_max + 1, config.n_step):
                print(f"Profiling scenarios with {n_measurements=} variables.")
                for repeat_index in range(config.repeats):
                    print(f"Repetition {repeat_index + 1}/{config.repeats}.")
                    rows = run_single_case(config, n_measurements, repeat_index, timeout_trackers)
                    all_rows.extend(rows)
        except KeyboardInterrupt:
            print(
                "\n[!] Benchmark interrupted by user (SIGINT). Returning collected results so far..."
            )
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (n_measurements, repeat_index)
            for n_measurements in range(config.n_min, config.n_max + 1, config.n_step)
            for repeat_index in range(config.repeats)
        ]

        print(
            f"Running benchmark in parallel using {num_workers} workers across {len(tasks)} tasks."
        )

        with multiprocessing.Manager() as manager:
            timeout_trackers = manager.dict()
            lock = manager.Lock()

            worker_tasks = [
                (config, n_meas, rep_idx, timeout_trackers, lock) for n_meas, rep_idx in tasks
            ]

            completed_tasks = 0
            try:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_worker_task, task) for task in worker_tasks]

                    for future in as_completed(futures):
                        completed_tasks += 1
                        try:
                            rows = future.result()
                            all_rows.extend(rows)
                            if rows:
                                n_meas = rows[0].n_measurements
                                rep = rows[0].repeat_index
                                print(
                                    f"[{completed_tasks}/{len(tasks)}] Finished profiling: n={n_meas}, repeat={rep + 1}"
                                )
                        except Exception as e:
                            print(f"[!] Worker task raised an exception: {e}")
                            if not config.continue_on_error:
                                raise
            except KeyboardInterrupt:
                print(
                    "\n[!] Benchmark interrupted by user (SIGINT). Stopping workers and collecting data..."
                )

    return all_rows


def save_results(rows, config):
    """Save results to CSV."""
    config.ensure_dirs()

    fieldnames = list(rows[0].as_dict().keys()) if rows else []

    with config.output_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())

    correlation_df, sizes_merged_df = compute_correlations(rows)
    sizes_merged_df.to_csv(config.output_dir / "sizes.csv", index=False)
    correlation_df.to_csv(config.output_dir / "correlations.csv", index=False)
