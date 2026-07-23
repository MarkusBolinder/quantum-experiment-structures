"""Collection of helpful functions and classes."""

import argparse
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import re
import signal
import sys

import jsonschema
import numpy as np
from quantum_experiment_structures.data.integer_sequences import DEDEKIND_NUMBERS


def minimal_antichain_enabling_relations(enabling_relations):
    """Return the subset-minimal enabling relations as an antichain.

    Each enabling relation is represented as a list of event dicts like
    {"m": "A", "v": 0}.

    If one relation is a strict subset of another, only the smaller relation is kept.
    Duplicate relations are removed as well.

    Args:
        enabling_relations: The raw enabling relations for a single measurement.

    Returns:
        A list of enabling relations forming a minimal antichain, sorted
        deterministically by size and then lexicographically.
    """
    if not enabling_relations:
        return []

    unique_relations = {
        frozenset((event["m"], event["v"]) for event in relation)
        for relation in enabling_relations
        if relation
    }

    minimal_relations = []
    for rel in unique_relations:
        is_minimal = True
        for other in unique_relations:
            if other is rel:
                continue
            if other < rel:
                is_minimal = False
                break
        if is_minimal:
            minimal_relations.append(rel)

    minimal_relations.sort(key=lambda rel: (len(rel), tuple(sorted(rel))))

    return [[{"m": m, "v": v} for m, v in sorted(relation)] for relation in minimal_relations]


def compute_transitive_closures(data, consistent=True):
    """Compute transitive closures of all enabling relations in a CCS.

    Args:
        data: A CCS-like dictionary containing at least the key "ms". Each
            measurement in data["ms"] must be a mapping with keys:
            - "m": measurement name (str)
            - "e": list of enabling relations, where each enabling relation is a
              list of event dicts with keys "m" and "v".
        consistent: A bool indicating if the computation should produce transitive closures that
            only contain consistent sets of events.

    Returns:
        A dictionary mapping each measurement name to the set of all distinct
        transitively closed enabling sets for that measurement. Each enabling
        set is represented as a frozenset of (measurement_name, value) tuples.

    Raises:
        KeyError: If a referenced measurement is missing from data["ms"].
        ValueError: If a cycle is detected in the enabling graph.
    """
    measurements = {m["m"]: m for m in data["ms"]}

    seen = dict()
    visiting = set()

    def merge_compatible(a, b):
        """Merge two event sets if they are consistent, else return None."""
        merged = dict(a)
        for m, v in b:
            if m in merged and merged[m] != v:
                return None
            merged[m] = v
        return frozenset(merged.items())

    def close_measurement(name):
        """Return all transitive closures for one measurement."""
        if name in seen:
            return seen[name]
        if name in visiting:
            raise ValueError(f"Cycle detected in enabling relations at {name!r}.")

        visiting.add(name)
        try:
            measurement = measurements[name]
            enabling_relations = measurement["e"]

            # root measurements are enabled by the empty set
            if not enabling_relations:
                result = set([frozenset()])
                seen[name] = result
                return result

            result = set()

            for relation in enabling_relations:
                # start with the empty closure for this relation, then expand event by event
                partial_closures = set([frozenset()])

                for event in relation:
                    parent = event["m"]
                    value = event["v"]

                    parent_closures = close_measurement(parent)
                    next_partials = set()

                    for partial in partial_closures:
                        for parent_closure in parent_closures:
                            if consistent:
                                merged = merge_compatible(partial, parent_closure)
                                if merged is None:
                                    continue

                                # add the direct enabling event itself
                                expanded = dict(merged)
                                if parent in expanded and expanded[parent] != value:
                                    continue
                                expanded[parent] = value
                                next_partials.add(frozenset(expanded.items()))
                            else:
                                merged = set(partial)
                                merged.update(parent_closure)
                                merged.add((parent, value))
                                next_partials.add(frozenset(merged))

                    partial_closures = next_partials
                    if not partial_closures and consistent:
                        break

                result.update(partial_closures)

            seen[name] = result
            return result
        finally:
            visiting.remove(name)

    return {name: close_measurement(name) for name in measurements}


def compute_mappings_for_extensive_game(extensive_game):
    """Parse an extensive form game tree and compute iset to nodes and player to iset mappings.

    Args:
        extensive_game: Python dict valid against the extensive game form schema.Collection

    Returns:
        A tuple (iset_to_nodes, player_to_isets)
            iset_to_nodes: A dictionary mapping information set IDs to a list of node references in
                that information set.
            player_to_isets A dictionary mapping player IDs to a list of unique information set IDs
                they play in.
    """
    iset_to_nodes = defaultdict(list)
    player_to_isets = defaultdict(set)

    def traverse(node):
        if not isinstance(node, dict):
            return

        if node.get("kind") == "choice":
            player = node.get("player")
            iset = node.get("information-set")

            if iset is not None:
                iset_to_nodes[iset].append(node)

                if player is not None:
                    player_to_isets[player].add(iset)

        for child in node.get("Children", node.get("children", [])):
            traverse(child)

    traverse(extensive_game)

    iset_to_nodes_dict = dict(iset_to_nodes)
    player_to_isets_dict = {
        player: sorted(list(isets)) for player, isets in player_to_isets.items()
    }

    return iset_to_nodes_dict, player_to_isets_dict


# https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time
# https://stackoverflow.com/questions/644073/signal-alarm-replacement-in-windows-python
def cancel_call(seconds=1.0):
    # custom signal handler
    def timeout_handler(signum, frame):
        raise TimeoutError()

    def function(function):
        def wrapper(*args, **kwargs):
            # NOTE: these signal methods only work on Unix systems
            old_handler = signal.getsignal(signal.SIGALRM)
            old_timer = signal.getitimer(signal.ITIMER_REAL)

            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                return function(*args, **kwargs)
            except TimeoutError as e:
                raise TimeoutError(f"Timeout: {seconds} sec reached. {function.__name__}") from e
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

                # restore the previous handler and timer state
                signal.signal(signal.SIGALRM, old_handler)
                if old_timer[0] > 0 or old_timer[1] > 0:
                    signal.setitimer(signal.ITIMER_REAL, *old_timer)  # pragma: no cover

        return wrapper

    return function


def json_file_size(path):
    """Return JSON file size in bytes."""
    return Path(path).stat().st_size


def json_size_bytes(obj):
    """Return serialized JSON size in bytes."""
    # NOTE: this specification of separtors eliminates whitespace (default is ", " and ": "),
    # so the value given by this function will be a lower bound.
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def get_json_obj_size(obj):
    """Return approximate in-memory size of a JSON-like object in bytes.

    The following types are supported.
      - dict
      - list / tuple
      - str
      - int / float / bool / None
    """
    seen = set()

    def sizeof(x):
        if not isinstance(x, (bool, int, float)) or x is None:
            obj_id = id(x)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

        size = sys.getsizeof(x)

        if isinstance(x, Mapping):
            size += sum(sizeof(k) + sizeof(v) for k, v in x.items())
        elif isinstance(x, (list, tuple)):
            size += sum(sizeof(item) for item in x)

        return size

    return sizeof(obj)


def count_enabling_relations_no_duplicates(n):
    """Count all enabling relations for n ordered measurements.

    For the i-th measurement (0-indexed), there are i earlier measurements
    available as possible causes. The number of enabling relations for that
    measurement is computed by '_count_enabling_relations_no_duplicates(i)'.

    The total number of enabling-relation choices for the full ordered list of
    n measurements is the product over i = 0, 1, ..., n - 1.

    Args:
        n: Number of measurements.

    Returns:
        The total number of enabling-relation configurations, with duplicate
        outcomes inside a context forbidden.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")

    count = 1
    for prior_measurements in range(n):
        count *= _count_enabling_relations_no_duplicates(prior_measurements)
    return count


def _count_enabling_relations_no_duplicates(prior_measurements):
    """Count enabling relations for one measurement.

    A candidate context is a non-empty consistent partial assignment on the
    prior_measurements earlier measurements. For each earlier measurement,
    the context may contain:
      - None: the measurement is absent
      - 0: outcome 0
      - 1: outcome 1

    Duplicate outcomes of the same measurement inside one context are
    forbidden, so contexts are exactly the non-empty elements of
    {None, 0, 1}^{prior_measurements} with no coordinate conflict.

    Contexts are ordered by extension:
        a <= b iff every specified coordinate of a agrees with b.

    An enabling relation is an antichain in this poset.

    The empty antichain is allowed and represents the always-enabled case.
    There is no separate never-enabled case in this count.

    Args:
        prior_measurements: Number of earlier measurements available as
            potential causes.

    Returns:
        The number of antichains in the poset of non-empty consistent contexts.

    Notes:
        Computationally infeasible for n > 5 most likely.
    """
    if prior_measurements < 0:
        raise ValueError("prior_measurements must be non-negative.")

    contexts = [
        ctx
        for ctx in itertools.product((None, 0, 1), repeat=prior_measurements)
        if any(v is not None for v in ctx)
    ]
    k = len(contexts)

    def leq(a, b):
        """Return True iff a <= b in the order."""
        return all(x is None or x == y for x, y in zip(a, b))

    comparable = [0] * k
    for i, x in enumerate(contexts):
        mask = 0
        for j, y in enumerate(contexts):
            if leq(x, y) or leq(y, x):
                mask |= 1 << j
        comparable[i] = mask

    @lru_cache(maxsize=None)
    def count(active_mask):
        """Count antichains in the induced subposet given by active_mask."""
        if active_mask == 0:
            return 1

        pivot = max(
            (i for i in range(k) if active_mask & (1 << i)),
            key=lambda i: (comparable[i] & active_mask).bit_count(),
        )

        without_pivot = active_mask & ~(1 << pivot)
        without_neighborhood = active_mask & ~comparable[pivot]

        return count(without_pivot) + count(without_neighborhood)

    return count((1 << k) - 1)


def count_enabling_relations(n):
    """Return the number of enabling relations possible.

    This assumes a linearization of the variables and binary outcomes.
    """
    if 2 * n - 2 >= len(DEDEKIND_NUMBERS):
        raise ValueError(
            "Not enough Dedekind numbers are known to calculate the number of enabling relations "
            f"for {n=}. Can only support queries for n < {len(DEDEKIND_NUMBERS) / 2 + 1}."
        )
    count = 1
    for i in range(1, n + 1):
        count *= DEDEKIND_NUMBERS[2 * i - 2] - 1
    return count


def count_covers(n):
    """Return the number of covers possible for n variables.

    A cover needs to be an anti-chain, and it needs to include all the variables. The formula this
    function uses can be derived using the principle of inclusion-exclusion.
    """
    if n >= len(DEDEKIND_NUMBERS):
        raise ValueError(
            f"Not enough Dedekind numbers are known to calculate the number of covers for {n=} "
            f"Can only support queries for n < {len(DEDEKIND_NUMBERS)}."
        )
    count = 0
    for k in range(0, n + 1):
        count += (-1) ** k * math.comb(n, k) * DEDEKIND_NUMBERS[n - k]
    return count


def count_causal_contextuality_scenarios(n, allow_duplicates=False):
    """Return the number of causal contextuality scenarios for n variables.

    This assumes that the outcomes are binary (0 or 1) and that the variables are linearized,
    meaning that the enabling relations cannot exhibit cycles.
    """
    enabling_relations_counter = (
        count_enabling_relations if allow_duplicates else count_enabling_relations_no_duplicates
    )
    return enabling_relations_counter(n) * count_covers(n)


def get_all_subsets(measurements):
    """Generate all non-empty subsets of the measurement set."""
    s = list(measurements)
    return [
        frozenset(combo) for i in range(1, len(s) + 1) for combo in itertools.combinations(s, i)
    ]


def is_antichain(collection):
    """Check if a collection of sets is an anti-chain (no set is a subset of another)."""
    for c1, c2 in itertools.combinations(collection, 2):
        if c1 < c2 or c2 < c1:
            return False
    return True


def create_local_covers(measurements):
    """Find all valid local covers for a given set of measurements.

    A local cover must be an anti-chain and its union must equal the ground set.
    The algorithm searches through all possible subsets of all non-empty subsets of the given
    measurements, so it quickly becomes very expensive and will only be efficient for fewer than
    five measurements.

    Raises:
        ValueError if len(measurements) > 4, because the algorithm would be too slow.

    Notes:
        Closely related to Sperner families: https://en.wikipedia.org/wiki/Sperner_family
        It is possible to express the number of local covers for n variables using the Dedekind
        numbers. See the function 'count_covers' above.
    """
    if len(measurements) > 4:
        raise ValueError(
            "The brute force approach scales as 2^(n^2 - 1), "
            "that is the power set of all non-empty subsets. "
            "For n >= 5, 2^(n^2 - 1) >= 2^31, which is infeasible."
        )
    measurement_set = set(measurements)
    subsets = get_all_subsets(measurements)
    valid_covers = []

    # iterate through the power set of the subsets.
    for r in range(1, len(subsets) + 1):
        for collection in itertools.combinations(subsets, r):
            # union must cover all measurements
            current_union = set().union(*collection)
            if current_union != measurement_set:
                continue

            # must be an anti-chain (maximality)
            if is_antichain(collection):
                readable = sorted([sorted(list(c)) for c in collection])
                valid_covers.append(readable)

    return valid_covers


def _parse_range(s):
    """Parse a range expression given as a string and return a pair of integers.

    This helper extracts one or two integer tokens from an input string and returns an integer
    2-tuple (min_val, max_val). The function accepts strings of the form "k" or "min:max" but is
    lenient about surrounding text — it will simply find the first one or two integer substrings
    and interpret them as the bounds. If a single integer is found it is returned duplicated
    (i.e. the closed interval [k, k]).

    The leniency means that you can give any string of the form '*\\d*\\d*' or '*\\d*'.

    Args:
        s (str): Input string containing one or two integers (examples: "3:6", "5", "range=2..4").

    Returns:
        tuple[int, int]: A pair (a, b) where a and b are integers. If the input contained a single
        integer k, the function returns (k, k). If two integers were found the first is returned
        as the lower bound and the second as the upper bound (no sorting is performed here).

    Raises:
        ValueError: If the input string does not contain exactly one or two integer tokens or if
        the parsing logic fails to extract the expected counts.
    """
    int_range = list(int(match) for match in re.findall(r"\d+", s))
    n_ints = len(int_range)
    if n_ints == 1:
        return int_range * 2
    if n_ints != 2:
        raise ValueError(f"Expected pattern with one or two integers but got {int_range}.")
    return int_range


def create_anti_chain(contexts):
    """Prune contexts that are subsets of other contexts to create an anti-chain.

    The definition of a cover, as given by Abramsky and Brandenburger (2011), states that it
    should be an anti-chain, which means that if c, c' ∈ C and c' ⊆ c then c = c'. The easiest
    way to ensure this is by only keeping the maximal elements in the cover, which also
    guarantees that the cover is still covering all measurements.


    Args:
        contexts: a list of sets containing the measurement names.

    Returns:
        the pruned cover, which will be an anti-chain
    """
    n = len(contexts)
    # interpret subset_matrix[i][j] as the answer to the question "Is c_i a subset of c_j?"
    subset_matrix = [[False] * n for _ in range(n)]
    for i, c1 in enumerate(contexts):
        for j, c2 in enumerate(contexts[i + 1 :], start=i + 1):
            intersection = c1 & c2
            if intersection == c1:
                # c1 is a subset of c2 (=> c2 is not a subset of c1)
                subset_matrix[i][j] = True
            elif intersection == c2:
                # c2 is a subset of c1
                subset_matrix[j][i] = True
            # neither is a subset of the other, so do not modify the matrix

    # if row i in subset_matrix has any True, then c_i is a subset of some other context
    # (the diagonal entries are set to False to avoid counting being a subset of itself)
    cover = [list(contexts[i]) for i in range(n) if not any(subset_matrix[i])]
    return cover


def extend_with_default(validator_class):
    """Create a validator that populates defaults.

    Finds defaults either directly on the property subschema or inside combination keywords
    ('allOf', 'anyOf', 'oneOf') so that schemas like
    '{"allOf": [{"$ref": "#/$defs/range"}, {"default": [3,6]}]}' work.
    """

    validate_properties = validator_class.VALIDATORS["properties"]

    def _find_default(schema_fragment):
        """Recursively search schema_fragment for a 'default' value.

        Looks at the fragment itself and then at combination keywords ('allOf', 'anyOf', 'oneOf').

        Returns:
            if there exists a default value, (default, True) is returned; if not, (False,) is
            returned. This is to allow falsy default values in the schema.
        """
        if not isinstance(schema_fragment, dict):
            return (False,)
        if "default" in schema_fragment:
            return (schema_fragment["default"], True)
        for comb in ("allOf", "anyOf", "oneOf"):
            members = schema_fragment.get(comb)
            if isinstance(members, list):
                for member in members:
                    d = _find_default(member)
                    if any(d):
                        return d
        return (False,)

    def set_defaults(validator, properties, instance, schema):
        if not isinstance(instance, dict):
            return

        for prop, subschema in properties.items():
            default_value = _find_default(subschema)
            if any(default_value):
                instance.setdefault(prop, default_value[0])

        yield from validate_properties(validator, properties, instance, schema)

    return jsonschema.validators.extend(  # type: ignore
        validator_class, {"properties": set_defaults}
    )


DefaultValuesValidator = extend_with_default(jsonschema.Draft202012Validator)


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


class ArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Amalgamation of argparse formatting classes."""
