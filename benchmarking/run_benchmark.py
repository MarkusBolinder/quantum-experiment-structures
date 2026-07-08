import argparse
from pathlib import Path

from benchmark_config import BenchmarkConfig, GeneratorConfig
from benchmark_runner import run_benchmark, save_results
from plot_results import plot_results
from quantum_experiment_structures.utils import utils


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile causal contextuality scenario conversions.",
        formatter_class=utils.ArgparseFormatter,
    )

    parser.add_argument(
        "--n-min",
        type=int,
        default=1,
        help="The minimum number of measurements to start the benchamark at.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=30,
        help="The maximum number of measurements to end the benchamark at.",
    )
    parser.add_argument("--n-step", type=int, default=1, help="Step size between scenario sizes.")
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of samples gathered for one scneario size."
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility.")

    # TODO: allow for finer granularity, it should be possible to specify individual methods
    parser.add_argument(
        "--ccs-mode",
        choices=["none", "checks", "adds", "both"],
        default="both",
        help="Which methods should be run for the scenarios.",
    )
    parser.add_argument(
        "--game-mode",
        choices=["none", "checks", "adds", "both"],
        default="both",
        help="Which methods should be run for the spacetime game.",
    )

    parser.add_argument(
        "--no-extensive", action="store_false", help="Do not convert to extensive games."
    )
    parser.add_argument("--no-gc", action="store_true", help="No garbage collection.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Produce plots of the results.")

    parser.add_argument(
        "--adaptive-timeout",
        action="store_true",
        help="Interrupt method calls that exceed a time limit.",
    )
    parser.add_argument(
        "--timeout-percentile",
        type=float,
        default=95,
        help="Percentile threshold at which to cancel calls.",
    )
    parser.add_argument(
        "--timeout-min-samples",
        type=int,
        default=8,
        help="Minimum amount of samples required before using the percentile threshold.",
    )
    parser.add_argument(
        "--timeout-base-s",
        type=float,
        default=0.25,
        help="Timeout time before relying on percentile (if percentile is given).",
    )
    parser.add_argument(
        "--timeout-history-size",
        type=int,
        default=50,
        help="Number of samples to use to determine the percentile threshold (sliding window).",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Leniency factor for avoiding buzzer beaters or whatever.",
    )
    parser.add_argument("--timeout-retry", action="store_true", help="Retry calls that timed out.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="Directory to save output to.",
    )

    parser.add_argument(
        "--produce-plots",
        type=Path,
        help="Path to a directory containing results to reproduce the plots. "
        "If this is specified it will override all other options. "
        "This path will also be the output destination of the new plots "
        "(it will overwrite existing).",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use. If 1 or omitted, runs sequentially.",
    )

    return parser.parse_args()


def main():
    """Run the benchmark and optionally create plots."""
    args = parse_args()

    if args.produce_plots:
        plot_results(args.produce_plots / "results.csv", args.produce_plots / "plots")
        exit()

    config = BenchmarkConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        n_step=args.n_step,
        repeats=args.repeats,
        base_seed=args.seed,
        ccs_mode=args.ccs_mode,
        game_mode=args.game_mode,
        extensive=args.no_extensive,
        collect_garbage=not args.no_gc,
        continue_on_error=not args.stop_on_error,
        output_dir=args.output_dir,
        output_csv=args.output_dir / "results.csv",
        output_jsonl=args.output_dir / "results.jsonl",
        plot_dir=args.output_dir / "plots",
        generator=GeneratorConfig(),
        adaptive_timeout_enabled=args.adaptive_timeout,
        timeout_percentile=args.timeout_percentile,
        timeout_min_samples=args.timeout_min_samples,
        timeout_base_s=args.timeout_base_s,
        timeout_history_size=args.timeout_history_size,
        timeout_multiplier=args.timeout_multiplier,
        timeout_retry_same_input=args.timeout_retry,
    )

    rows = run_benchmark(config, num_workers=args.workers)
    save_results(rows, config)

    if args.plot:
        plot_results(config.output_csv, config.plot_dir)

    print(f"Saved CSV file(s) to: {config.output_dir}")
    print(f"Saved JSONL to: {config.output_jsonl}")
    if args.plot:
        print(f"Saved plots to: {config.plot_dir}")


if __name__ == "__main__":
    main()
