"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

from collections import defaultdict
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

    def validate_data(self):
        """Validate the data using the CCS schema."""
        try:
            jsonschema.validate(self.data, schema=CCS_SCHEMA)
            return True
        except Exception as e:
            print(e)
            return False

    def check_consistency(self, allow_duplicates=False):
        """Ensure that the enabling relations are consistent.

        Assume X is a measurement setting with outcomes 0 or 1, then an enabling relation is
        consistent if it does not contain conflicting events for any measurement, meaning the set of
        enabling relations does not contain {(X,0),(X,1)} or similar.

        Args:
            allow_duplicates: if True, an enabling relation of the form {(X,0),(X,0)} is not treated
                as a consistency error; if False, then it is.
        Returns:
            a boolean indicating if the scenario is consistent (True if yes).
        """
        for measurement in self.data["ms"]:
            for enabling_relation in measurement["e"]:
                required = set()
                for event in enabling_relation:
                    if not allow_duplicates:
                        event_id = [event["m"]]
                    else:
                        event_id = [event["m"], tuple(event.values())]
                    # necessary condition: the measurement (event_id[0]) is in the set
                    # sufficient condition:
                    # 1) if no duplicates, then the above is sufficient OR
                    # 2) if duplicates are allowed, then there is a consistency error if the event
                    #    tuple (event_id[1]) differs, i.e. if the outcome value is different.
                    if event_id[0] in required and (
                        not allow_duplicates or event_id[1] not in required
                    ):
                        return False
                    required.update(event_id)
        return True

    def calculate_leafs(self):
        """Iterate through the scenario and determine which measurement outcomes are leaves."""
        # put all events that are part of an enabling relation in a hash set
        enabling_events = set(
            tuple(event.values())
            for measurement in self.data["ms"]
            for enabling_relation in measurement["e"]
            for event in enabling_relation
        )
        # iterate through all measurement outcomes and mark leaves accordingly
        for measurement in self.data["ms"]:
            measurement_variable = measurement["m"]
            for outcome in measurement["o"]:
                if (measurement_variable, outcome["v"]) not in enabling_events:
                    outcome["l"] = True
                else:
                    # NOTE: might be redundant, but cost is probably negligible
                    outcome["l"] = False

    def calculate_memberships(self):
        """Calculate which contexts each measurement setting is part of.

        Returns:
            a dict mapping each measurements to all contexts it appears in.
        """
        measurement_to_contexts = defaultdict(list)
        for context in self.data["c"]:
            for measurement in context:
                # NOTE: this assumes that there are no duplicate contexts,
                # nor measurements within the context
                measurement_to_contexts[measurement].append(context)
        return measurement_to_contexts

    def check_contexts(self):
        """Certify that the cover and the memberships are consistent and valid.

        The contexts listed for each measurement should all contain the measurement that they are
        nested with. Furthermore, all contexts that appear as these 'membership'-contexts should
        also be present in the top level cover of the scenario.
        """
        # TODO: combine this method with totality of cover method, or break into two parts
        cover = set(frozenset(context) for context in self.data["c"])
        contexts = set()
        for measurement in self.data["ms"]:
            for context in measurement["c"]:
                context = frozenset(context)
                if measurement["m"] not in context:
                    return False
                contexts.add(context)
        return cover == contexts

    def check_totality_of_union(self):
        """Ensure that the union of all contexts covers all measurements and nothing more."""
        # TODO: if needed, investigate if representing measurements as bit vectors can optimize
        measurements_in_contexts = set(
            measurement for context in self.data["c"] for measurement in context
        )
        measurements = set(measurement["m"] for measurement in self.data["ms"])
        # this also returns False if the contexts contain measurements that are not in the scenario
        return measurements_in_contexts == measurements

    def check_unique_contexts(self, check_all=True):
        """Ensure that the cover does not contain duplicate contexts.

        Args:
            check_all: indicates whether all the contexts for each measurement should be checked for
            duplicates too. If False, only the top level field 'c' is checked.
        """
        # TODO: can frozenset be used everywhere?
        if check_all:
            for measurement in self.data["ms"]:
                contexts = set(frozenset(context) for context in measurement["c"])
                if len(contexts) != len(measurement["c"]):
                    return False
        contexts = set(frozenset(context) for context in self.data["c"])
        if len(contexts) != len(self.data["c"]):
            return False
        return True

    def add_human_readable(self):
        """Add a human readable representation of the CCS to the data."""
        # TODO: cache results that are calculated in methods, or store them as attributes
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

    def to_json(self, filename):
        """Flush data to a JSON file.

        Args:
            filename: path-like name of the output file.
        """
        path = Path(filename)
        with path.open() as f:
            json.dump(f)

    def append_to_json_lines(self, filename):
        """Append the CCS to a JSON lines file specified by filename.

        Args:
            filename: name of the file to which the data shall be appended.
        """
        pass
