import copy
from functools import lru_cache
import itertools

import quantum_experiment_structures as qes
from quantum_experiment_structures.data.local_covers import LOCAL_COVERS


class CCSEnumerator:
    """Sister class to CCSGenerator that enumerates all possible CCSs for n measurements.

    With 'all', we mean that every possible cover combined with every possible enabling relation for
    scenarios will be enumerated by this class. This scales super-exponentially, and it is only
    feasible to run this on one computer for n <= 3, and using a cluster it is only feasible for
    n <= 4. For n = 5, we would need to iterate over ~10^21 scenarios (when duplicate measurements
    in enabling relations are not allowed (i.e. inconsistent set of events are not allowed)), which
    is infeasible to compute. For the enabling relations, a lexicographical order is assumed, i.e. A
    is always enabled by default, B can only be enabled by A, C can only be enabled by A or B, and
    so on. Furthermore, the enumerated enabling relations here are minimal, meaning that if we have
    {(A,0)} ⊢ C and {(A,0),(B,1)} ⊢ C, the second enabling relation is not enumerated, it is not
    minimal as the first one is a subset of it. All measurements have exactly two outcomes.
    """

    def __init__(self, n, names=None, covers=None, allow_duplicates=False):
        """Initialize the CCSEnumerator.
        Args:
            n: Number of measurements.
            names: Optional measurement names.
                If omitted, uses ["A", "B", ..., "Z", "AA", ..., "AZ", "BA", ... ].
            covers: Optional list of covers that the enumerator will use to create CCSs.
                Shall be a list[list[list[int]], where each cover is a list of contexts, where each
                context is a list of integers representing the measurements index in the
                linearization. E.g. [[0, 1], [1, 2]] is a cover with two contexts, where the first
                context contains the first and second measurement in the linearization; the second
                context contains the second and third measurement in the linearization.
            allow_duplicates: If True, allow duplicate-measurement contexts in enabling
                relations. If False, exclude them.

        """
        if n < 1:
            raise ValueError("n must be at least 1.")
        if covers is None:
            if n - 1 >= len(LOCAL_COVERS):
                raise ValueError(
                    f"No static covers available for n={n}. Expected LOCAL_COVERS to contain index {n - 1}."
                )
            self.covers = LOCAL_COVERS[n - 1]
        else:
            self.covers = covers

        if names is None:
            self.names = qes.CCSGenerator._generate_measurement_names(n)
        else:
            if len(names) != n:
                raise ValueError("names must have length n.")
            self.names = names
        self.n_measurements = n
        self.allow_duplicates = allow_duplicates

    def rename_cover(self, cover):
        """Rename an index-based cover using the provided measurement names."""
        idx_to_name = {i: name for i, name in enumerate(self.names)}
        return [[idx_to_name[i] for i in context] for context in cover]

    def _create_valid_enabling_relation(self, relation):
        """Convert an antichain of events into valid enabling relation."""
        return [
            [{"m": m, "v": v} for m, v in sorted(ctx)]
            for ctx in sorted(relation, key=self._context_key)
        ]

    @staticmethod
    def _get_all_nonempty_subsets(events):
        """Return all non-empty subsets of events as frozensets."""
        events = tuple(events)
        subsets = []
        for r in range(1, len(events) + 1):
            for combo in itertools.combinations(events, r):
                subsets.append(frozenset(combo))
        return tuple(subsets)

    @staticmethod
    def _context_key(ctx):
        """Stable sort key for a context."""
        return (len(ctx), tuple(sorted(ctx)))

    @staticmethod
    def _incomparable(a, b):
        """Return True iff neither context contains the other."""
        return not (a <= b or b <= a)

    @staticmethod
    def _has_duplicate_measurements(ctx):
        """Return True if a context contains more than one outcome of the same measurement."""
        measurements = [m for m, _ in ctx]
        return len(measurements) != len(set(measurements))

    @lru_cache(maxsize=None)
    def _candidate_contexts(self, prior_names):
        """Return all possible non-empty contexts over the prior outcome-events."""
        events = tuple((name, v) for name in prior_names for v in (0, 1))
        contexts = self._get_all_nonempty_subsets(events)

        if not self.allow_duplicates:
            contexts = tuple(ctx for ctx in contexts if not self._has_duplicate_measurements(ctx))

        return tuple(sorted(contexts, key=self._context_key))

    def _iter_antichains(self, candidates):
        """Yield all antichains of the given candidate contexts."""
        if not candidates:
            yield ()
            return

        first = candidates[0]
        rest = candidates[1:]

        # branch 1: do not include 'first'
        yield from self._iter_antichains(rest)

        # branch 2: include 'first', and remove anything comparable to it
        reduced = tuple(ctx for ctx in rest if self._incomparable(ctx, first))
        for tail in self._iter_antichains(reduced):
            yield (first,) + tail

    def iter_enabling_relations(self, prior_names):
        """Yield every valid enabling relation for one measurement.

        Args:
            prior_names: Names of the earlier measurements. For measurement index i,
                this should be names[:i].

        Yields:
            A list of contexts, where each context is a list of event
            objects {"m": X "v": 0|1}.
        """
        candidates = self._candidate_contexts(tuple(prior_names))
        for relation in self._iter_antichains(candidates):
            yield self._create_valid_enabling_relation(relation)

    def enumerate(self):
        """Yield every causal contextuality scenario for n variables.

        Yields:
            JSON-serializable CCS objects.
        """
        for base_cover in self.covers:
            cover = self.rename_cover(base_cover)

            def build(i, acc_ms):
                if i == self.n_measurements:
                    yield copy.deepcopy({"ms": acc_ms, "c": cover})
                    return

                prior_names = self.names[:i]
                for relation in self.iter_enabling_relations(prior_names):
                    ms_obj = {"m": self.names[i], "e": relation, "o": [{"v": 0}, {"v": 1}]}
                    yield from build(i + 1, acc_ms + [ms_obj])

            yield from build(0, [])
        print(self._candidate_contexts.cache_info())
