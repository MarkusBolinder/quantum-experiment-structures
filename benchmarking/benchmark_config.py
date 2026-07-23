from dataclasses import dataclass, field
from pathlib import Path

from quantum_experiment_structures.utils import utils
from quantum_experiment_structures.data.schemas import CCS_GENERATOR_SETTINGS_SCHEMA


@dataclass
class GeneratorConfig:
    """Parameters passed to 'CCSGenerator'."""

    n_values_range: list[int] = field(default_factory=lambda: [2, 5])
    n_contexts_range: list[int] = field(default_factory=lambda: [2, 30])
    context_size_range: list[int] = field(default_factory=lambda: [2, 15])
    n_alternatives_range: list[int] = field(default_factory=lambda: [1, 1])  # unique causal bridges
    enabling_relation_size_range: list[int] = field(default_factory=lambda: [1, 3])
    n_samples_per_causal_structure: int = 1
    p_has_enabled: float = 0.7
    enabling_relation_size_mean: float = 1.5
    causally_secured: bool = True
    n_scenarios: int = 1

    def build_for_n(self, n_measurements, seed):
        """Build a generator settings dict for a fixed measurement count."""
        return {
            "n_measurements_range": [n_measurements, n_measurements],
            "n_values_range": self.n_values_range,
            "n_contexts_range": self.n_contexts_range,
            "context_size_range": self.context_size_range,
            "n_alternatives_range": self.n_alternatives_range,
            "enabling_relation_size_range": self.enabling_relation_size_range,
            "n_samples_per_causal_structure": self.n_samples_per_causal_structure,
            "p_has_enabled": self.p_has_enabled,
            "enabling_relation_size_mean": self.enabling_relation_size_mean,
            "causally_secured": self.causally_secured,
            "n_scenarios": self.n_scenarios,
            "seed": seed,
        }


@dataclass
class BenchmarkConfig:
    """Controls what is benchmarked and how results are saved."""

    n_min: int = 1
    n_max: int = 30
    n_step: int = 1
    repeats: int = 5
    base_seed: int = 1

    ccs_mode: str = "both"  # none | checks | adds | both
    game_mode: str = "both"

    extensive: bool = True
    collect_garbage: bool = True
    continue_on_error: bool = True
    granular: bool = False  # allows switching between aggregated and per-method reporting

    adaptive_timeout_enabled: bool = False
    timeout_percentile: float = 95.0
    timeout_min_samples: int = 8
    timeout_history_size: int = 50
    timeout_base_s: float = 0.25
    timeout_multiplier: float = 1.0
    timeout_retry_same_input: bool = False

    output_dir: Path = Path("benchmark_output")
    output_csv: Path = Path("benchmark_output/results.csv")
    output_jsonl: Path = Path("benchmark_output/results.jsonl")
    plot_dir: Path = Path("benchmark_output/plots")

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def validate(self):
        """Validate the configuration."""
        valid_modes = {"none", "checks", "adds", "both"}
        for name, mode in (
            ("ccs_mode", self.ccs_mode),
            ("game_mode", self.game_mode),
        ):
            if mode not in valid_modes:
                raise ValueError(f"{name} must be one of {sorted(valid_modes)}")

        if self.n_min <= 0:
            raise ValueError("n_min must be positive.")
        if self.n_max < self.n_min:
            raise ValueError("n_max must be >= n_min.")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive.")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive.")
        if not 0 < self.timeout_percentile <= 100:
            raise ValueError("timeout_percentile must be in (0,100]")
        if self.timeout_min_samples < 0:
            raise ValueError("timeout_min_samples must be non-negative.")
        if self.timeout_base_s <= 0:
            raise ValueError("timeout_base_s must be positive.")

        validator = utils.DefaultValuesValidator(CCS_GENERATOR_SETTINGS_SCHEMA)
        # TODO: generate more scenarios at once and get a distribution for the generation too
        validator.validate(self.generator.build_for_n(n_measurements=1, seed=1))

    def ensure_dirs(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
