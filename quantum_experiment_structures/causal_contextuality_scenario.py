"""Validate a JSON representation of a CCS using both a schema and additional consistency checks."""

import json
from pathlib import Path

import quantum_experiment_structures as qes
from quantum_experiment_structures.data.schemas import CCS_SCHEMA
import jsonschema


class CausalContextualityScenario:

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


    def check_consistency(self):
        """Ensure that the enabling relations are consistent.

        Assume X is a measurement setting with outcomes 0 or 1, then an enabling relation is
        consistent if it does not contain conflicting events for any measurement, meaning the set of
        enabling relations does not contain {(X,0),(X,1)} or similar.
        """
        pass

    def calculate_leafs(self):
        """Iterate through the scenario and determine which measurement outcomes are leaves."""
        pass

    def calculate_memberships(self):
        """Calculate which contexts each measurement setting is part of."""
        pass

    def check_unique_contexts(self):
        """Ensure that the cover does not contains duplicate contexts."""
        pass

    def add_human_readable(self):
        """Add a human readable representation of the CCS to the data."""
        pass

    def to_json(self, filename):
        """Flush data to a JSON file.

        Args:
            filename: name of the output file.
        """
        pass

    def append_to_json_lines(self, filename):
        """Append the CCS to a JSON lines file specified by filename.

        Args:
            filename: name of the file to which the data shall be appended.
        """
        pass
