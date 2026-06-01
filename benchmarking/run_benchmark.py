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

    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--ccs-mode",
        choices=["none", "checks", "adds", "both"],
        default="both",
    )
    parser.add_argument(
        "--game-mode",
        choices=["none", "checks", "adds", "both"],
        default="both",
    )

    parser.add_argument("--extensive", action="store_false")
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--adaptive-timeout", action="store_true")
    parser.add_argument("--timeout-percentile", type=float, default=95)
    parser.add_argument("--timeout-min-samples", type=int, default=8)
    parser.add_argument("--timeout-base-s", type=float, default=0.25)
    parser.add_argument("--timeout-history-size", type=int, default=50)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--timeout-retry", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))

    return parser.parse_args()


def main():
    """Run the benchmark and optionally create plots."""
    args = parse_args()

    config = BenchmarkConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        n_step=args.n_step,
        repeats=args.repeats,
        base_seed=args.seed,
        ccs_mode=args.ccs_mode,
        game_mode=args.game_mode,
        extensive=args.extensive,
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

    rows = run_benchmark(config)
    save_results(rows, config)

    if args.plot:
        plot_results(config.output_csv, config.plot_dir)

    print(f"Saved CSV file(s) to: {config.output_dir}")
    print(f"Saved JSONL to: {config.output_jsonl}")
    if args.plot:
        print(f"Saved plots to: {config.plot_dir}")


if __name__ == "__main__":
    main()
