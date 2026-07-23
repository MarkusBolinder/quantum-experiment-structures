import gc
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from quantum_experiment_structures.utils import utils


@dataclass
class ProfileResult:
    """Hold profiling output for one instrumented function call."""

    value: Any
    stage: str
    wall_time_s: float
    cpu_time_s: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def profile_callable(
    stage,
    func,
    collect_garbage=True,
):
    """Profile a zero-argument callable.

    Args:
      stage: Name of the stage being profiled.
      func: Zero-argument callable to execute.
      collect_garbage: Whether to run gc.collect() before timing.

    Returns:
      A tuple of (result, profile_result). If 'func' raises, result is None and
      the exception is recorded in 'profile_result.error'.
    """
    if collect_garbage:
        gc.collect()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    error = None
    result = None

    try:
        result = func()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    finally:
        wall_time_s = time.perf_counter() - wall_start
        cpu_time_s = time.process_time() - cpu_start

    profile_result = ProfileResult(
        value=result,
        stage=stage,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        error=error,
    )
    return profile_result


def resolve_first_callable(obj, candidate_names):
    """Find the first callable attribute on 'obj' from 'candidate_names'.

    Args:
      obj: Object to inspect.
      candidate_names: Possible method names.

    Returns:
      A pair of (method_name, bound_method).

    Raises:
      AttributeError: If none of the candidates exist and are callable.
    """
    for name in candidate_names:
        method = getattr(obj, name, None)
        if callable(method):
            return name, method

    candidates = ", ".join(candidate_names)
    raise AttributeError(f"None of these callables exist on {type(obj).__name__}: {candidates}")


@dataclass
class RollingPercentileTracker:
    percentile: float
    min_samples: int
    base_time_s: float
    history_size: int = 50
    multiplier: float = 1.0
    durations_s: deque[float] = field(default_factory=deque)
    killed_count: int = 0

    def __post_init__(self):
        self.durations_s = deque(maxlen=self.history_size)

    def budget_s(self):
        diff = len(self.durations_s) - self.min_samples
        if diff < 0 or (diff == self.min_samples == 0):
            return self.base_time_s
        xs = sorted(self.durations_s)
        idx = int(round((self.percentile / 100.0) * (len(xs) - 1)))
        return max(self.base_time_s, self.multiplier * xs[idx])

    def observe_success(self, duration_s):
        self.durations_s.append(duration_s)

    def observe_kill(self):
        self.killed_count += 1


@dataclass
class TimedCallResult:
    value: Any = None
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    killed: bool = False
    error: str | None = None


def profile_callable_with_timeout(func, timeout_s):
    """Run 'func' in a separate process and terminates it on timeout."""

    @utils.cancel_call(seconds=timeout_s)
    def timeout_func():
        return func()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        value = timeout_func()
        status, value, wall_time_s, cpu_time_s, error = (
            "ok",
            value,
            time.perf_counter() - wall_start,
            time.process_time() - cpu_start,
            None,
        )
    except TimeoutError as e:  # noqa: BLE001
        print("TIMEOUT")
        status, value, wall_time_s, cpu_time_s, error = (
            "timeout",
            None,
            time.perf_counter() - wall_start,
            time.process_time() - cpu_start,
            f"{type(e).__name__}: {e}",
        )

    return TimedCallResult(
        value=value,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        killed=(status != "ok"),
        error=error,
    )
