"""Collection of helpful functions and classes."""

import argparse
import json
import re

import jsonschema
import numpy as np


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

    return jsonschema.validators.extend(validator_class, {"properties": set_defaults})


DefaultValuesValidator = extend_with_default(jsonschema.Draft202012Validator)


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Amalgamation of argparse formatting classes."""
