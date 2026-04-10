"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

from collections import defaultdict, deque
from dataclasses import dataclass
import inspect
import json
from pathlib import Path

from quantum_experiment_structures.data.schemas import CCS_SCHEMA
import jsonschema


@dataclass(frozen=True)
class _BobNode:
    """Internal representation of a Bob node.

    Attributes:
        n: Unique node id.
        bridge: The enabling bridge represented by this Bob node.
        ps: Parent witness pairs as (parent_node_id, action_label).
        a: The actions available at this Bob node.
    """

    n: str
    bridge: frozenset[tuple[str, str]]
    ps: tuple[tuple[str, str], ...]
    a: tuple[str, ...]


@dataclass(frozen=True)
class _AlfredNode:
    """Internal representation of an Alfred node.

    Attributes:
        n: Unique node id.
        bob: Parent Bob node id.
        m: Measurement name.
        c: Local context.
        a: Alfred action label, i.e. the context label.
    """

    n: str
    bob: str
    m: str
    c: frozenset[str]
    a: str


class CausalContextualityScenario:
    """Represents a causal contextuality scenario and converts it to a game.

    A causal contextuality scenario consists of a family of measurements, each
    with a finite set of outcomes and optional enabling relations, together
    with a global cover of compatible measurement contexts.

    In addition to just representing the data, a number of methods are supplied which can be
    leveraged to ensure validity and/or calculate properties of the CCS.

    A method converting the CCS to a spacetime game also exists. Note that this requires the
    scenario to have unique causal bridges and a causally secured cover.
    """

    def __init__(self, json_data):
        """Initialize a causal contextuality scenario from JSON-like data.

        Args:
            json_data: A dictionary containing the scenario description. The
                dictionary is expected to contain (there are some optional fields too):
                - "ms": a list of measurements, where each measurement has:
                    - "m": the measurement name
                    - "e": a list of enabling sets
                    - "o": a list of outcomes
                - "c": a list of contexts forming the global cover

        Attributes:
            data: The raw input dictionary.
            measurements: A mapping from measurement name to its measurement record.
            cover: A set of frozensets, each representing a context in the global cover.
        """
        self.data = json_data
        self.measurements = {measurement["m"]: measurement for measurement in self.data["ms"]}
        self.cover = set(frozenset(context) for context in self.data["c"])

    def __repr__(self):
        if "h" in self.data:
            return "\n".join(f"{k:6s}: {v}" for k, v in self.data["h"].items())
        return str(self.data)

    def validate(self):
        """Validate the data using the CCS schema."""
        try:
            jsonschema.validate(self.data, schema=CCS_SCHEMA)
            return True
        except jsonschema.ValidationError as e:
            print(f"Validation error: {e}")
            return False

    def check_consistency(self):
        """Ensure that the enabling relations are consistent and free of duplicates.

        Assume X is a measurement setting with outcomes 0 or 1, then an enabling relation is
        consistent if it does not contain conflicting events for any measurement, meaning the set of
        enabling relations does not contain {(X,0),(X,1)} or similar.

        Raises:
            ValueError if there is any inconsistency or duplicate evens in an enabling relation.
        """
        for measurement in self.data["ms"]:
            for enabling_relation in measurement["e"]:
                seen = set()
                for event in enabling_relation:
                    measurement = event["m"]
                    if measurement in seen:
                        raise ValueError(
                            f"Duplicate measurement {measurement} "
                            f"in enabling relation {enabling_relation}"
                        )
                    seen.add(measurement)
        return True

    def add_leaves(self):
        """Add the leaf field to all measurement outcomes in the CCS.

        Note that this will not overwrite any existing leaf-fields.
        """
        self._handle_leaves(False)

    def check_leaves(self):
        """Check all measurement outcomes' leaves for correctness; otherwise raise error.

        Raises:
            ValueError: if any measurement outcome does not have the leaf field, or if the outcome
                is incorrectly labeled as leaf/not a leaf.
        """
        if not self._handle_leaves(True):
            raise ValueError(
                "Leaves are not valid. Either a leaf is missing, or a leaf has the incorrect value."
            )
        return True

    def _handle_leaves(self, check):
        """Iterate through the scenario and determine which measurement outcomes are leaves.

        Args:
            check: if False, then all measurement outcomes will be populated with the leaf
                field, but an existing leaf will be left as is; if True, then all measurement
                outcomes' leaves are checked that they both exist and are correctly annotated as a
                leaf or not.

        Returns:
            boolean indicating if the leaves are correct or that the missing leaves were written.
        """
        # put all events that are part of an enabling relation in a hash set
        enabling_events = set(
            tuple(event.values())
            for measurement in self.data["ms"]
            for enabling_relation in measurement["e"]
            for event in enabling_relation
        )
        # iterate through all measurement outcomes check/write leaves accordingly
        for measurement in self.data["ms"]:
            measurement_variable = measurement["m"]
            for outcome in measurement["o"]:
                leaf = outcome.get("l")
                if leaf is None:
                    if check:
                        return False
                    outcome["l"] = (measurement_variable, outcome["v"]) not in enabling_events
                elif check:
                    correct = (measurement_variable, outcome["v"]) not in enabling_events
                    if leaf != correct:
                        return False
        return True

    def calculate_memberships(self):
        """Calculate which contexts each measurement setting is part of.

        Returns:
            a dict mapping each measurements to all contexts it appears in.
        """
        measurements_to_contexts = defaultdict(list)
        for context in self.data["c"]:
            for measurement in context:
                # NOTE: because the schema forbids duplicate contexts and duplicate measurements
                # within a context, this is fine
                measurements_to_contexts[measurement].append(context)
        return measurements_to_contexts

    def add_memberships(self):
        """Add .ms.[$k].c field if it is not already there.

        Which contexts each measurement is part of can be calculated based on the top-level cover,
        so it is optional in the schema, and, if the field is missing, this method will add it.
        Whereas if it is already there, we will simply ignore it for now, and let the later check
        determine whether it is correct.
        """
        measurements_to_contexts = self.calculate_memberships()
        for measurement in self.data["ms"]:
            if "c" not in measurement:
                measurement["c"] = measurements_to_contexts[measurement["m"]]

    def check_contexts(self):
        """Certify that the cover and the memberships are consistent and valid.

        The contexts listed for each measurement should all contain the measurement that they are
        nested with. Furthermore, all contexts that appear as these 'membership'-contexts should
        also be present in the top level cover of the scenario.

        Furthermore, the schema can check that there are no duplicate arrays, however, this only
        checks uniqueness up to (not including) order within the array. For example, the following
        cover is valid against the schema [ ["A", "B"], ["B", "A"] ], but contains duplicate
        contexts. The schema only flags covers like [ ["A", "B"], ["A", "B"] ] as invalid.
        """
        if any(measurement.get("c") is None for measurement in self.data["ms"]):
            # one or more measurements are missing their membership contexts, so we can not properly
            # check that the union of this agree with the top-level cover
            return True
        contexts = set()
        for measurement in self.data["ms"]:
            for context in measurement["c"]:
                context = frozenset(context)
                if measurement["m"] not in context:
                    return False
                contexts.add(context)
        return self.cover == contexts and len(contexts) == len(self.data["c"])

    def check_cover(self):
        """Ensure union of all contexts in the cover covers all measurements and nothing more.

        Because the schema validation forces the cover to have unique contexts and that the contexts
        themselves have unique measurements, we only need to check that the union of all contexts
        equals the set of measurements.
        """
        # TODO: if needed, investigate if representing measurements as bit vectors can optimize
        measurements_in_contexts = set(
            measurement for context in self.data["c"] for measurement in context
        )
        # this also returns False if the contexts contain measurements that are not in the scenario
        return measurements_in_contexts == set(self.measurements)

    def check_unique_values(self):
        """Ensure that there are no duplicates in the values of any measurement's outcomes.

        The schema can ensure that the outcome-objects are unique, but because these objects contain
        both a value ('v') field and a leaf ('l') field, you may still have duplicate values. E.g.
        the following array of outcomes is valid against the schema:
            [
                {"v": 0},
                {"v": 0, "l": true},
                {"v": 0, "l": false}
            ]
        even though all outcome objects have the same value.
        """
        for measurement in self.data["ms"]:
            outcomes = measurement["o"]
            values = set(outcome["v"] for outcome in outcomes)
            if len(values) != len(outcomes):
                return False
        return True

    def add_human_readable(self):
        """Add a human readable representation of the CCS to the data.

        There are no checks to ensure that the human readable format correctly described the CCS.
        However, the schema enforces some contstraints on the format of the human readable part.
        """
        # TODO: cache results that are calculated in methods, or store them as attributes
        # document that human readable is not checked.
        measurements = "{" + ", ".join(measurement["m"] for measurement in self.data["ms"]) + "}"
        outcomes = {
            measurement["m"]: set(outcome["v"] for outcome in measurement["o"])
            for measurement in self.data["ms"]
        }
        outcomes_representation = ", ".join(
            f"O_{measurement} = {values}" for measurement, values in outcomes.items()
        )
        enabling_events_per_measurement = {
            measurement["m"]: [
                [tuple(event.values()) for event in enabling_relation]
                for enabling_relation in measurement["e"]
            ]
            for measurement in self.data["ms"]
        }
        enabling_relations = ", ".join(
            (
                f"∅ ⊢ {measurement}"
                if not events
                else ", ".join(
                    f"{{{','.join(f'({x},{v})' for x, v in event)}}} ⊢ {measurement}"
                    for event in events
                )
            )
            for measurement, events in enabling_events_per_measurement.items()
        )
        cover = "{" + ", ".join("{" + ", ".join(context) + "}" for context in self.data["c"]) + "}"
        self.data["h"] = {
            "ms": measurements,
            "o": outcomes_representation,
            "e": enabling_relations,
            "c": cover,
        }

    def check_anti_chain(self):
        """Check that no context is a superset of another context.

        In the contextuality setting, it is not meaningful to have contexts that are supersets of
        other contexts, since you could simply always choose the larger context. This is referred to
        as the cover being an anti-chain, hence the name.
        """
        if len(self.data["c"]) < 2:
            return True
        contexts = [frozenset(context) for context in self.data["c"]]
        for i, c1 in enumerate(contexts):
            for c2 in contexts[i + 1 :]:
                intersection = c1 & c2
                if intersection == c1 or intersection == c2:
                    return False
        return True

    def sort_data(self):
        """Sort the cover, contexts and measurement lexicographically w.r.t. measurement names."""
        for measurement in self.data["ms"]:
            if "c" in measurement:
                measurement["c"] = sorted(sorted(context) for context in measurement["c"])
        self.data["ms"] = sorted(self.data["ms"], key=lambda measurement: measurement["m"])
        self.data["c"] = sorted(sorted(context) for context in self.data["c"])

    def all_checks(self):
        """Perform all checks."""
        for name, member in inspect.getmembers(self):
            if inspect.ismethod(member) and name.startswith("check"):
                ok = member()
                if not ok:
                    raise ValueError(f"Inconsistency detected: {name} failed.")
        return True

    def all_adds(self):
        """Add everything that can be added based on the base CCS data."""
        for name, member in inspect.getmembers(self):
            if inspect.ismethod(member) and name.startswith("add"):
                member()

    # TODO: this will probably flag most scenarios created by CCSGenerator -- how to handle?
    def check_unique_causal_bridges(self):
        """Check that the scenario only has unique causal bridges.

        Specifically, check that each measurement only has one enabling relation that can be
        activated at one time -- meaning that you could have ∅ ⊢ A, {(A,0)} ⊢ B and {(A,1)} ⊢ B,
        because A cannot give both outcomes 0 and 1 in one experiment. However, ∅ ⊢ A, ∅ ⊢ B,
        {(A,0)} ⊢ C and {(B,0)} ⊢ C would be an example of a non-unique causal bridge.

        This is true if every pair of enabling relations for a given measurement contains at least
        one contradictory outcome for a common parent measurement.
        """
        for measurement in self.data["ms"]:
            enabling_relations = measurement["e"]

            # trivially unique
            if len(enabling_relations) <= 1:
                continue

            # compare every pair of enabling relations for this measurement
            for i, rel1 in enumerate(enabling_relations):
                for j, rel2 in enumerate(enabling_relations[i + 1 :], start=i + 1):
                    # empty set enablings conflict with everything
                    if not rel1 or not rel2:
                        return False

                    found_contradiction = False
                    rel2_map = {event["m"]: event["v"] for event in rel2}

                    for event in rel1:
                        m_parent, v_parent = event["m"], event["v"]
                        if m_parent in rel2_map and v_parent != rel2_map[m_parent]:
                            found_contradiction = True
                            break

                    # if the enabling relations do not conflict, we have a non-unique causal bridge
                    if not found_contradiction:
                        return False

        return True

    def check_no_cycles(self):
        """Make sure that there are no cycles in the enabling relations of the scenario.

        That is, we can linearize the measurements in some order. If X is enabled by Y,
        then Y cannot enable X. Cycles are detected using DFS.
        """
        # adjacency relations
        adj = {measurement: set() for measurement in self.measurements}
        for measurement in self.data["ms"]:
            child = measurement["m"]
            for relation in measurement["e"]:
                for event in relation:
                    parent = event["m"]
                    adj[parent].add(child)

        # all nodes we have been to
        visited = set()
        # all nodes in the current dfs path
        recursion_stack = set()

        def has_cycle(u):
            visited.add(u)
            recursion_stack.add(u)

            for v in adj[u]:
                if v not in visited:
                    if has_cycle(v):
                        return True
                elif v in recursion_stack:
                    return True

            recursion_stack.remove(u)
            return False

        for node in adj:
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    @staticmethod
    def _dedupe_preserve_order(items):
        """Remove duplicates while preserving the original order.

        Args:
            items: An iterable of hashable items.

        Returns:
            A list containing the unique items from 'items', in their first observed order.
        """
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @staticmethod
    def _value_label(value):
        """Convert an outcome value into a schema-safe string label.

        Args:
            value: An outcome value. This may be a dictionary such as
                {'v': 0}, a string, or another JSON-serializable value.

        Returns:
            A string label suitable for use as an action label in the game schema.
        """
        if isinstance(value, dict) and set(value.keys()) == {"v"}:
            value = value["v"]

        if isinstance(value, str):
            return value

        # Stable, JSON-safe fallback for non-string outcomes.
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _context_label(context):
        """Format a context as a readable string label.

        Args:
            context: A frozenset of measurement names.

        Returns:
            A string of the form '{}', or '{X,Y,Z}' with sorted names.
        """
        if not context:
            return "{}"
        return "{" + ",".join(sorted(context)) + "}"

    def to_spacetime_game(self):
        """Convert the scenario into a spacetime game dictionary.

        The conversion constructs a game with two players:
        Bob, who has perfect information, and Alfred, who may have imperfect
        information.

        The conversion is done according to the following plan:

        1. Compute the local cover restriction at each enabling set by
           intersecting the global cover with the enabled measurements.
        2. Build Bob nodes for enabling histories and Alfred nodes for
           measurement/context pairs reachable from those Bob nodes.
        3. Group nodes into information sets:
           - Bob information sets are singletons.
           - Alfred information sets contain all nodes for the same
             measurement.

        Returns:
            A dictionary representing the spacetime game. The returned value is
            intended to be valid against the spacetime game JSON schema.

        Raises:
            ValueError: If the scenario violates the unique causal bridge
                assumption, if a local cover restriction is empty, or if the
                conversion encounters an inconsistent causal structure.

        Naming convention of nodes:
            Bob node ids
                The base form is:
                    B:{} for the root Bob node
                    B:{Y=0} for a Bob node whose enabling bridge is Y=0

            Alfred node ids
                The form is:
                    A:<bob-node-id>:<measurement>:<context>
                Example:
                    A:B:{}:X:{X,Y}
                This means:
                    Alfred node reached from Bob root B:{}, measurement X, local context {X,Y}.

        Information set ids
            Bob:<node-id> for Bob singleton sets
            Alfred:<measurement> for Alfred measurement sets
        """
        # parse enabling relations and outcomes
        enabling_of = dict()
        enabled_by = defaultdict(list)
        outcomes_by_measurement = dict()

        for name, measurement in self.measurements.items():
            outcomes_by_measurement[name] = [
                self._value_label(outcome["v"]) for outcome in measurement["o"]
            ]

            enabling_relations = measurement["e"]
            if len(enabling_relations) > 1:
                raise ValueError(
                    f"Multiple enabling relations for {name}; unique causal bridges are required."
                )

            if enabling_relations:
                bridge = frozenset(
                    (event["m"], self._value_label(event["v"])) for event in enabling_relations[0]
                )
            else:
                bridge = frozenset()

            # bridge is the unique causal bridge for this measurement
            enabling_of[name] = bridge
            enabled_by[bridge].append(name)

        # compute local cover restrictions C_t = {C' ∩ enabled(t) | C' in C} \ {∅}
        local_cover = dict()
        for bridge, enabled_measurements in enabled_by.items():
            enabled_set = set(enabled_measurements)
            restriction = {
                frozenset(context & enabled_set) for context in self.cover if context & enabled_set
            }
            if not restriction:
                raise ValueError(
                    f"Empty local cover at enabling set {sorted(bridge)}; incompatible scenario."
                )
            local_cover[bridge] = restriction

        root = frozenset()
        if root not in local_cover:
            raise ValueError("No root local cover found for the empty enabling set.")

        # created nodes
        bob_nodes = []
        alfred_nodes = []

        # store nodes for fast lookup
        bob_by_id = dict()
        bob_by_bridge = dict()
        alfred_by_id = dict()
        alfred_by_signature = set()

        # Alfred nodes grouped by measurement
        alfred_by_measurement = defaultdict(list)

        # queue of Bob node ids to expand into Alfred nodes
        bob_queue = deque()

        def add_bob_node(bridge, parents):
            """Create a Bob node instance keyed only by its enabling bridge."""
            if bridge in bob_by_bridge:
                return None

            # NOTE: this needs to be sorted for the key lookups to work properly
            parent_entries = tuple(sorted(tuple(p.values()) for p in parents))

            if not bridge:
                bridge_str = "{}"
            else:
                inside = ",".join(f"({m},{v})" for m, v in bridge)
                bridge_str = "{" + inside + "}"

            node_id = f"B:{bridge_str}"

            node = _BobNode(
                n=node_id,
                bridge=bridge,
                ps=parent_entries,
                a=tuple(self._context_label(context) for context in local_cover[bridge]),
            )

            bob_by_bridge[bridge] = node
            bob_by_id[node_id] = node
            bob_nodes.append(node)
            bob_queue.append(node_id)
            return node

        def add_alfred_node(bob_node_id, measurement, context):
            """Create an Alfred node for a particular Bob node, measurement, and context."""
            context_label = self._context_label(context)
            sig = (bob_node_id, measurement, context_label)
            if sig in alfred_by_signature:
                return None

            alfred_by_signature.add(sig)
            node_id = f"A:{measurement}_{context_label}"
            node = _AlfredNode(
                n=node_id,
                bob=bob_node_id,
                m=measurement,
                c=context,
                a=context_label,
            )

            alfred_nodes.append(node)
            alfred_by_id[node_id] = node
            alfred_by_measurement[measurement].append(node)
            return node

        # root node
        add_bob_node(root, [])

        processed_bob = set()

        def expand_bob_nodes():
            """Expand all queued Bob nodes into Alfred nodes."""
            while bob_queue:
                current_id = bob_queue.popleft()
                if current_id in processed_bob:
                    continue
                processed_bob.add(current_id)

                current_bridge = bob_by_id[current_id].bridge

                for context in local_cover[current_bridge]:
                    for measurement in sorted(context):
                        add_alfred_node(current_id, measurement, context)

        def try_create_bob_nodes():
            """Create all Bob nodes whose enabling bridges are currently witnessable."""
            created_any = False

            for bridge in enabled_by:
                if not bridge:
                    continue  # root already handles empty enabling set

                if bridge in bob_by_bridge:
                    continue

                antecedents = bridge  # [(m, v), ...]
                candidate_lists = []
                possible = True

                for ant_measurement, _ in antecedents:
                    candidates = alfred_by_measurement.get(ant_measurement, [])
                    if not candidates:
                        possible = False
                        break
                    candidate_lists.append(candidates)

                if not possible:
                    continue

                # Alfred nodes for the same measuremnt lead to the same Bob node
                # because they correspond to the same measurement and the same causal bridge
                parents = []
                for alfred_measurement_nodes, (_, ant_value) in zip(candidate_lists, antecedents):
                    for alfred_node in alfred_measurement_nodes:
                        parents.append({"p": alfred_node.n, "a": ant_value})

                add_bob_node(bridge, parents)
                created_any = True

            return created_any

        # build the game until we reach a fixpoint
        while bob_queue:
            expand_bob_nodes()
            # new Alfred nodes may enable more nodes for Bob
            while try_create_bob_nodes():
                expand_bob_nodes()

        info_sets = []

        # Bob: singleton information sets
        for bob_node in bob_nodes:
            info_sets.append(
                {
                    "i": f"Bob:{bob_node.n}",
                    "ns": [
                        {
                            "n": bob_node.n,
                            "ps": [{"p": pid, "a": action} for pid, action in bob_node.ps],
                        }
                    ],
                    "p": "Bob",
                    "a": self._dedupe_preserve_order(list(bob_node.a)),
                }
            )

        # Alfred: one information set per measurement
        for measurement, nodes in alfred_by_measurement.items():
            info_sets.append(
                {
                    "i": f"Alfred:{measurement}",
                    "ns": [
                        {
                            "n": node.n,
                            "ps": [{"p": node.bob, "a": node.a}],
                        }
                        for node in nodes
                    ],
                    "p": "Alfred",
                    "a": self._dedupe_preserve_order(outcomes_by_measurement[measurement]),
                }
            )

        # global action array
        actions = set()

        # Bob actions
        actions.update(action for bob_node in bob_nodes for action in bob_node.a)

        # Alfred actions
        actions.update(out for outcomes in outcomes_by_measurement.values() for out in outcomes)

        return {
            "ps": ["Bob", "Alfred"],
            "as": list(actions),
            "is": info_sets,
        }

    def to_json(self, filename, indent=None):
        """Flush data to a JSON file.

        Args:
            filename: path-like name of the output file.
        """
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".json")

        with path.open("w") as f:
            json.dump(self.data, f, indent=indent)

    def append_to_json_lines(self, filename):
        """Append the CCS data to a JSON Lines file.

        Creates the file if it does not already exist. Each call writes
        one JSON object on a single line.

        Args:
            filename: path-like name of the .jsonl file.
        """
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".jsonl")

        with path.open("a") as f:
            json.dump(self.data, f)
            f.write("\n")

    def everything(self):
        # first validate against schema
        if not self.validate():
            raise jsonschema.ValidationError("The data is not valid against the schema.")
        # sort data for readability/quality of life
        self.sort_data()
        # then add missing fields
        self.all_adds()
        # then check that everything is correct
        self.all_checks()
        # TODO: validate against schema again?
        return True
