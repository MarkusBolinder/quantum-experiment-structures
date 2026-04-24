"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

from collections import defaultdict, deque
import copy
from dataclasses import dataclass
import inspect
from itertools import product
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


class StableCausalContextualityScenario(CausalContextualityScenario):
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

    def _topological_order(self):
        """Create a topological order of the measurements.

        The scenario is assumed to be acyclic. The order is used so that parent
        measurements are always processed before their children.

        Returns:
            A list of measurement names in topological order.

        Raises:
            ValueError: If the enabling relations contain a cycle.
        """
        indegree = {measurement: 0 for measurement in self.measurements}
        adjacency = defaultdict(set)

        for measurement in self.data["ms"]:
            child = measurement["m"]
            for enabling_relation in measurement["e"]:
                for event in enabling_relation:
                    parent = event["m"]
                    if child not in adjacency[parent]:
                        adjacency[parent].add(child)
                        indegree[child] += 1

        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in sorted(adjacency[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.measurements):
            raise ValueError("The enabling relations contain a cycle.")

        return order

    def _measurement_is_stable(self, measurement_name):
        """Check whether a measurement can be safely duplicated.

        A measurement is considered stable if no facet of the cover can realize two
        different enabling relations for that measurement at the same time.

        This is used by the duplication routine -- if two bridges can coexist inside one facet,
        then the split would be ambiguous and should not be done.

        Args:
            measurement_name: name of the measurement to test.

        Returns:
            True if the measurement is stable; False if not.
        """
        enabling_relations = self.measurements[measurement_name]["e"]

        # trivially stable if there is no ambiguity
        if len(enabling_relations) <= 1:
            return True

        supports = [
            frozenset(event["m"] for event in enabling_relation)
            for enabling_relation in enabling_relations
        ]

        # NOTE: because the local covers consist of subsets of the global contexts, it should be
        # sufficient to check if the support is a subset of the global contexts
        for facet in self.cover:
            compatible = [i for i, support in enumerate(supports) if support <= facet]

            # if a single facet can enable more than one bridge, then the measurement is unstable
            if len(compatible) > 1:
                return False

        return True

    def check_stability(self):
        """Check that the scenario is stable.

        A measurement is trivially stable if it has a unique causal bridge. If the measurement has
        multiple causal bridges, i.e. multiple enabling relations, it is stable if only one causal
        bridge can be realized in a history. This will depend on the local context and the events in
        the causal bridges. For example, if ∅ ⊢ ∅ ⊢ X,Y and {(X,0)} ⊢ Z, {(Y,1)} ⊢ Z and the cover
        is {{X,Y,Z}} it is not stable, because both the enabling relations can be active at the same
        time, but if we have the cover  {{X,Z},{Y,Z}} it is stable because only one causal bridge
        can be active. Stability implies a notion of measurements that correspond to different
        physical entities, i.e. in the example (with the cover that gives stability) if we have the
        event (X,0), the Z we enable is physically different from the Z we enable by the event
        (Y,1), but if we do not have stability, i.e. the cover {{X,Y,Z}}, the causal bridges lead to
        same physical measurement Z.
        """
        for measurement in self.measurements:
            if not self._measurement_is_stable(measurement):
                raise ValueError(
                    f"Measurement '{measurement}' is unstable and cannot be duplicated "
                    "because more than one enabling relation can occur in the same facet."
                )
        return True

    def deduplicate_causal_bridges(self):
        """Duplicate measurements until all causal bridges are unique.

        If a measurement has multiple causal bridges, which hinders it from being converted to a
        spacetime game immediately, it may still be possible to duplicate the measurements with
        multiple enabling relations, such that every copy only has one enabling relation (i.e. we
        have a new scenario with unique causal bridges). This is only possible if the enabling
        relations are stable, meaning that only one causal bridge can be activated given the
        contexts available in the scenario. If the bridge is unstable, then we could follow two or
        more enabling relations at the same time, which pollutes the causal history, since we cannot
        know which causal bridge enabled the measurement.

        The method unfolds the scenario by creating one copy for every stable enabling relation.
        The duplication is propagated recursively through the causal structure, and the cover is
        modified to include the duplicates.

        If a measurement has several enabling relations, the method first checks that those
        relations are stable with respect to the cover. More formally, stability means that no
        facet can realize two different enabling relations for the same measurement at the same
        time. If such ambiguity exists, the measurement cannot be split and the method raises an
        error.

        This method assumes that all measurements are stable.

        Returns:
            A new scenario object of the same class, with unique causal bridges.

        Raises:
            ValueError: if the scenario contains a cycle
        """
        topo_order = self._topological_order()
        # create the duplicated measurements
        #
        # copy_records_by_original[m] is a list of duplicated measurements for m
        # each record stores:
        #   - name: the new measurement name
        #   - original: the original measurement name
        #   - parents: the concrete copied parent names used to enable it
        #   - relation: the original enabling relation index
        copy_records_by_original = defaultdict(list)
        copy_lookup = dict()

        def make_copy_name(original_name, index):
            """Create a short, deterministic copy name.

            The original name is kept for the first copy. Subsequent copies use a
            numbered suffix.
            """
            if index == 0:
                return original_name
            return f"{original_name}_{index}"

        for measurement_name in topo_order:
            measurement = self.measurements[measurement_name]
            outcomes = copy.deepcopy(measurement["o"])
            enabling_relations = measurement["e"]

            # root measurements, or measurements with a single bridge, are copied
            # once per compatible parent-copy combination
            if not enabling_relations:
                copy_name = make_copy_name(measurement_name, index=0)
                copy_record = {
                    "name": copy_name,
                    "original": measurement_name,
                    "parents": tuple(),
                    "relation": None,
                    "measurement": {
                        "m": copy_name,
                        "e": [],
                        "o": outcomes,
                    },
                }
                copy_records_by_original[measurement_name].append(copy_record)
                copy_lookup[(measurement_name, None, tuple())] = copy_record
                continue

            for relation_index, enabling_relation in enumerate(enabling_relations):
                parent_measurements = [event["m"] for event in enabling_relation]

                # gather the created copies of each parent measurement
                parent_copy_lists = [
                    copy_records_by_original[parent_name] for parent_name in parent_measurements
                ]

                if any(not parent_copy_list for parent_copy_list in parent_copy_lists):
                    raise ValueError(
                        f"Measurement '{measurement_name}' cannot be duplicated because "
                        f"one of its parents has not been expanded yet."
                    )

                # every combination of parent copies corresponds to a valid copy of the measurement
                for parent_combo in product(*parent_copy_lists):
                    parent_names = tuple(parent_copy["name"] for parent_copy in parent_combo)
                    copy_name = make_copy_name(
                        measurement_name,
                        len(copy_records_by_original[measurement_name]),
                    )

                    copied_relation = [
                        {
                            "m": parent_copy["name"],
                            "v": event["v"],
                        }
                        for parent_copy, event in zip(parent_combo, enabling_relation)
                    ]

                    copy_record = {
                        "name": copy_name,
                        "original": measurement_name,
                        "parents": parent_names,
                        "relation": relation_index,
                        "measurement": {
                            "m": copy_name,
                            "e": [copied_relation],
                            "o": outcomes,
                        },
                    }

                    copy_records_by_original[measurement_name].append(copy_record)
                    copy_lookup[(measurement_name, relation_index, parent_names)] = copy_record

        # lift the cover: for each original facet, we select the unique compatible copy of every
        # measurement that appears in that facet. Stability guarantees uniqueness
        new_cover = []

        for facet in self.cover:
            chosen = dict()

            for measurement_name in topo_order:
                if measurement_name not in facet:
                    continue

                candidates = []

                for copy_record in copy_records_by_original[measurement_name]:
                    # admissible if all copied parents have already been chosen for this facet
                    if set(copy_record["parents"]) <= set(chosen.values()):
                        candidates.append(copy_record)

                if len(candidates) != 1:
                    raise ValueError(
                        f"Could not lift facet '{sorted(facet)}' uniquely for measurement "
                        f"'{measurement_name}'. Got the following candidate contexts: {candidates}."
                    )

                chosen[measurement_name] = candidates[0]["name"]

            new_cover.append(sorted(chosen.values()))

        new_cover = self._dedupe_preserve_order(tuple(context) for context in new_cover)
        new_cover = [list(context) for context in new_cover]

        # flatten the duplicated measurements
        new_measurements = [
            copy_record["measurement"]
            for measurement_name in topo_order
            for copy_record in copy_records_by_original[measurement_name]
        ]

        new_data = dict(self.data)
        new_data["ms"] = new_measurements
        new_data["c"] = new_cover

        return self.__class__(new_data)


class CausallySecuredScenario(StableCausalContextualityScenario):
    """Subclass for causal contextuality scenarios convertible to spacetime games.

    Requires unique causal bridges, no cycles, a causally secured cover,
    and clean local covers.
    """

    def _get_transitive_enabling(self, measurement_name):
        """Compute the transitive closure of enabling events for a measurement.

        Returns:
            a dictionary mapping measurement names to their required outcome values.
        """
        closure = dict()
        queue = deque([measurement_name])
        seen = set()

        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)

            # NOTE: this assumes unique causal bridges
            enabling_relations = self.measurements[current]["e"]
            if not enabling_relations:
                continue

            for event in enabling_relations[0]:
                parent_m, parent_v = event.values()

                if parent_m in closure and closure[parent_m] != parent_v:
                    # internally inconsistent scenario (unused setting)
                    return None

                closure[parent_m] = parent_v
                queue.append(parent_m)

        return closure

    def check_causally_secured_cover(self):
        """Verify that the cover is causally secured.

        This means facet membership propagates to all enabling measurements,
        two measurements with inconsistent enabling conditions never belong to the same facet,
        and C is the maximum among all covers C′ that have the same local cover restrictions.
        """
        # precompute transitive closures for all measurements
        tau_bars = dict()
        for m in self.measurements:
            t_bar = self._get_transitive_enabling(m)
            if t_bar is None:
                return False
            tau_bars[m] = t_bar

        for facet in self.cover:
            # 1) propagation: ∀C' ∈ C, ∀x ∈ C, support(τ(x)) ⊆ C'
            for x in facet:
                enabling_relations = self.measurements[x]["e"]
                if not enabling_relations:
                    continue
                support = set(event["m"] for event in enabling_relations[0])
                if not support <= facet:
                    return False

            # 2) consistency: (τ_bar(x) ∪ τ_bar(y)) must be consistent for all x, y in C
            facet_list = list(facet)
            for i, x in enumerate(facet_list):
                for y in facet_list[i + 1 :]:
                    t_x, t_y = tau_bars[x], tau_bars[y]
                    # check for conflicting assignments in the transitive history
                    common_parents = set(t_x.keys()) & set(t_y.keys())
                    if any(t_x[p] != t_y[p] for p in common_parents):
                        return False

        # 3) maximality: this is verified by the anti-chain check
        return True

    def check_local_covers_clean(self):
        """Verify that all local covers are clean.

        No local cover restriction should separate measurements that collectively serve as an
        enabling condition for a downstream measurement. For example, if ∅ ⊢ X, Y and
        {(X,0),(Y,0)} ⊢ Z, then the local cover C_∅ = {{X}, {Y}} would be deemed 'unclean'
        (or 'dirty'), and the only possible clean local cover would be C_∅ = {X, Y}.
        """
        # map each bridge to the measurements it enables
        enabled_by_bridge = defaultdict(list)
        for name, measurement in self.measurements.items():
            enabling_relations = measurement["e"]
            if enabling_relations:
                bridge = frozenset(tuple(event.values()) for event in enabling_relations[0])
            else:
                bridge = frozenset()
            enabled_by_bridge[bridge].append(name)

        # for every measurement z, check if its enabling bridge t_z
        # is realizable in the local cover of its own parents
        for z, measurement in self.measurements.items():
            enabling_relations = measurement["e"]
            if not enabling_relations:
                continue

            t_z = enabling_relations[0]
            support_z = set(event["m"] for event in t_z)

            # group support_z by the bridges that enable them
            support_by_parent_bridge = defaultdict(set)
            for name in support_z:
                enabling_relations = self.measurements[name]["e"]
                if enabling_relations:
                    m_bridge = frozenset(tuple(event.values()) for event in enabling_relations[0])
                else:
                    m_bridge = frozenset()
                support_by_parent_bridge[m_bridge].add(name)

            for parent_bridge, sub_support in support_by_parent_bridge.items():
                # compute local cover C_t at the parent level
                # C_t = {C' ∩ enabled(t) | C' ∈ C} \ {∅}
                enabled_at_parent = set(enabled_by_bridge[parent_bridge])
                local_contexts = [
                    frozenset(context & enabled_at_parent)
                    for context in self.cover
                    if (context & enabled_at_parent)
                ]

                # clean check: sub_support must be a subset of at least one local context,
                # (sub_support split across local contexts => z can never be enabled => not clean)
                if not any(sub_support <= context for context in local_contexts):
                    return False

        return True

    # TODO: make sure that this one is run first in all_checks to fail fast instead of doing the
    # expensive checks above (causally secured cover and clean local covers)
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
                for rel2 in enabling_relations[i + 1 :]:
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
                    {} for the root Bob node
                    {Y=0} for a Bob node whose enabling bridge is Y=0

            Alfred node ids
                The form is:
                    <measurement>_<context>
                Example:
                    X_{X,Y}
                This means:
                    Alfred node for measurement X, reached from Bob choosing context {X,Y}.

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

            node_id = bridge_str

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
            node_id = f"{measurement}_{context_label}"
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

                # Alfred nodes for the same measurement lead to the same Bob node
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
