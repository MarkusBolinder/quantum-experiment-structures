"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

from collections import defaultdict
import inspect
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

    # TODO: investigate how this should be handled
    def wip_check_no_superset_contexts(self):
        """Check that no context is a superset of another context.

        In the contextuality setting, it is not meaningful to have contexts that are supersets of
        other contexts, since you could simply always choose the larger context.
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
            raise jsonschema.ValidationError()
        # sort data for readability/quality of life
        self.sort_data()
        # then add missing fields
        self.all_adds()
        # then check that everything is correct
        self.all_checks()
        # TODO: validate against schema again?
        return True
