"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

from collections import defaultdict
import inspect
import itertools
import json
from pathlib import Path

from quantum_experiment_structures.data.schemas import CCS_SCHEMA
import jsonschema


class CausalContextualityScenario:
    """Python representation of a causal contextuality scenario.

    In addition to just representing the data, a number of methods are supplied which can be
    leveraged to ensure validity and/or calculate properties of the CCS.
    """

    def __init__(self, json_data):
        """Read, validate and initialize an instance of a causal causal contextuality scenario.

        Args:
            json_data: a dict-object representing well-formed JSON.
        """
        self.data = json_data
        self.measurements = set(measurement["m"] for measurement in self.data["ms"])
        self.cover = set(frozenset(context) for context in self.data["c"])

    def __repr__(self):
        if "h" in self.data:
            return "\n".join(f"{k:6s}: {v}" for k, v in self.data["h"].items())
        return self.data

    def validate(self):
        """Validate the data using the CCS schema."""
        try:
            jsonschema.validate(self.data, schema=CCS_SCHEMA)
            return True
        except Exception as e:
            print(e)
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

        Note that this will overwrite any existing leaf-fields.
        """
        self._handle_leaves(True)

    def check_leaves(self):
        """Check all measurement outcomes' leaves for correctness; otherwise raise error.

        Raises:
            ValueError: if any measurement outcome does not have the leaf field, or if the outcome
                is incorrectly labeled as leaf/not a leaf.
        """
        if not self._handle_leaves(False):
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
        return measurements_in_contexts == self.measurements

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

    def get_causally_secured_cover(self, sort=True):
        # map measurements to their data
        measurements_dict = {m["m"]: m for m in self.data["ms"]}
        m_names = list(measurements_dict.keys())
        all_measurements_fs = frozenset(m_names)

        # determine which outcomes occur in enabling relations
        # significantly reduces the branching factor
        enabling_outcomes = {name: set() for name in m_names}
        for measurement in self.data["ms"]:
            for relation in measurement["e"]:
                for event in relation:
                    if event["m"] in enabling_outcomes:
                        enabling_outcomes[event["m"]].add(event["v"])

        valid_contexts = set()
        visited_assignments = set()
        found_full_set = False

        def backtrack(current_assignment):
            nonlocal found_full_set
            if found_full_set:
                return

            # dp caching
            state = frozenset(current_assignment.items())
            if state in visited_assignments:
                return
            visited_assignments.add(state)

            enabled_in_this_world = set()
            changed = True
            while changed:
                changed = False
                for m_name in m_names:
                    if m_name in enabled_in_this_world:
                        continue

                    enabling_conditions = measurements_dict[m_name]["e"]

                    if not enabling_conditions:
                        enabled_in_this_world.add(m_name)
                        changed = True
                        continue

                    for relation in enabling_conditions:
                        satisfied = True
                        for event in relation:
                            m, v = event["m"], event["v"]
                            # a relation is satisfied if all (m, v) in it are in our assignment
                            # AND those measurements are themselves enabled
                            if m not in enabled_in_this_world or current_assignment.get(m) != v:
                                satisfied = False
                                break

                        if satisfied:
                            enabled_in_this_world.add(m_name)
                            changed = True
                            break

            if enabled_in_this_world:
                fs = frozenset(enabled_in_this_world)
                valid_contexts.add(fs)
                # no need to keep recursing
                if fs == all_measurements_fs:
                    found_full_set = True
                    return

            # identify enabled measurements without assigned outcomes
            to_assign = [m for m in enabling_outcomes if m not in current_assignment]
            if not to_assign:
                return

            # take the first unassigned measurement as the next one
            next_m = to_assign[0]
            all_values = [o["v"] for o in measurements_dict[next_m]["o"]]

            # determine which outcomes are worth testing
            outcomes_to_try = set()
            for v in enabling_outcomes[next_m]:
                if v in all_values:
                    outcomes_to_try.add(v)

            # add an outcome that is not represented to cover the case when the measurement is not
            # present in the context, if such a case exists
            for v in all_values:
                if v not in enabling_outcomes[next_m]:
                    outcomes_to_try.add(v)
                    break

            for val in outcomes_to_try:
                current_assignment[next_m] = val
                backtrack(current_assignment)
                if found_full_set:
                    return
                del current_assignment[next_m]

        # start recursion
        backtrack(dict())

        # ensure the cover is an anti-chain (keep maximal elements)
        cover = []
        if found_full_set:
            cover = [all_measurements_fs]
        else:
            sorted_contexts = sorted(list(valid_contexts), key=len, reverse=True)
            for context in sorted_contexts:
                if not any(context < other for other in cover):
                    cover.append(context)

        if sort:
            return sorted(sorted(list(context)) for context in cover)

        return [list(context) for context in cover]

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
        adj = defaultdict(set)
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
