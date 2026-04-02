"""Collection of helpful functions and classes."""

import argparse
from collections import defaultdict
import itertools
import json
import re

import graphviz
import jsonschema
import numpy as np


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


def plot_spacetime_game(game, filename="spacetime_game", format="png", view=True, same_size=False):
    """Produce a visual representation of a spacetime game DAG.

    Args:
        game: the qes.SpacetimeGame instance.
        filename: name of file to which the graph will be written.
        format: file format.
        view: boolean indicating whether to immediately open the created file.
        same_size: boolean indicating whether all the nodes in the created DAG should have the same
            size. If True, the minimum necessary size will be heuristically estimated based on the
            longest labels present in the spacetime game. If False, dynamic sizing will be used.

    Returns:
        the graphviz.Digraph object that is created.
    """
    if same_size:
        # pre-calculate all potential labels to find the maximum dimensions needed
        all_potential_labels = []
        for node_name, node_info in game.nodes.items():
            iset_id = node_info["info_set_id"]
            player = game.info_sets[iset_id].get("p", "N/A")
            all_potential_labels.append(f"{node_name}\n({iset_id})\n[{player}]")

        for history in game.data.get("z", []):
            u_list = [f"{u['p']}:{u['v']}" for u in history["u"]]
            all_potential_labels.append("Payoffs:\n" + "\n".join(u_list))

        max_chars = 0
        max_lines = 0
        for label in all_potential_labels:
            lines = label.split("\n")
            max_lines = max(max_lines, len(lines))
            for line in lines:
                max_chars = max(max_chars, len(line))

        # heuristic for 32pt font: ~0.25in per char width, ~0.5in per line height
        # We take the max of width and height to keep the nodes circular/square
        width = max_chars * 0.25
        height = max_lines * 0.5
        node_size_val = max(width, height)
        node_size = str(node_size_val)
        leaf_size = (str(width), str(height))
    else:
        node_size = "2.5"
        leaf_size = ("1.5", "1.0")
    fixed_size = str(same_size).lower()

    dot = graphviz.Digraph(
        name=filename,
        comment="Spacetime Game DAG",
        graph_attr={
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "nodesep": "0.8",
            "ranksep": "1.0",
        },
        node_attr={
            "fontsize": "32",
            "shape": "circle",
            "fixedsize": fixed_size,
            "width": node_size,
            "height": node_size,
        },
        edge_attr={
            "fontsize": "48",
            "fontcolor": "darkgreen",
        },
    )

    # '{', '}', '|', etc. are special characters with dot, so prefix 'n' to the names
    node_to_id = {name: f"n{i}" for i, name in enumerate(game.nodes.keys())}

    # 1) create decision nodes
    for node_name, node_info in game.nodes.items():
        safe_id = node_to_id[node_name]
        iset_id = node_info["info_set_id"]
        player = game.info_sets[iset_id].get("p", "N/A")

        display_label = f"{node_name}\n({iset_id})\n[{player}]"
        dot.node(safe_id, display_label)

    # 2) create edges and terminal payoff nodes
    for node_name, node_info in game.nodes.items():
        safe_id = node_to_id[node_name]
        iset_id = node_info["info_set_id"]
        actions = game.info_sets[iset_id].get("a", [])

        # track which actions from this node actually lead to children
        out_edges = game.adj.get(node_name, [])
        actions_with_children = set(edge["a"] for edge in out_edges)

        for action in actions:
            if action in actions_with_children:
                # standard causal transition
                for edge in out_edges:
                    if edge["a"] == action:
                        dot.edge(safe_id, node_to_id[edge["c"]], label=str(action))
            else:
                # terminal branch: lead to a payoff node
                leaf_id = f"leaf_{safe_id}_{action}"

                # search for a history 'z' that contains this (iset, action) pair to find payoff
                payoff_label = "End"
                for history in game.data.get("z", []):
                    if any(h["i"] == iset_id and h["a"] == action for h in history["h"]):
                        u_list = [f"{u['p']}:{u['v']}" for u in history["u"]]
                        payoff_label = "Payoffs:\n" + "\n".join(u_list)
                        break

                # render the terminal node as a smaller box or empty shape
                dot.node(
                    leaf_id,
                    payoff_label,
                    shape="box",
                    style="dotted",
                    width=leaf_size[0],
                    height=leaf_size[1],
                    fixedsize=fixed_size,
                    fontsize="24",
                )
                dot.edge(safe_id, leaf_id, label=str(action))

    # 3) handle information sets (add dashed lines)
    iset_members = defaultdict(list)
    for node_name, node_info in game.nodes.items():
        iset_id = node_info["info_set_id"]
        iset_members[iset_id].append(node_name)

    for iset_id, nodes in iset_members.items():
        n = len(nodes)
        if n <= 1:
            continue
        for i in range(n - 1):
            dot.edge(
                node_to_id[nodes[i]],
                node_to_id[nodes[i + 1]],
                style="dashed",
                color="blue",
                arrowhead="none",
                constraint="false",
                penwidth="2.0",
            )

    dot.render(filename, format=format, cleanup=True, view=view)
    return dot


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
