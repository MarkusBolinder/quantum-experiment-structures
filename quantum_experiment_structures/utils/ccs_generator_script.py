#!/usr/bin/env python3
"""Script for generating CCS using the CCSGenerator class.

Used as a command line interface (CLI) for batch generation of datasets.

When executed from the CLI, parameters such as the number of measurements,
contexts, and enabling relations are sampled from user-provided ranges.
Generated scenarios can be written either as individual JSON files or as
JSON Lines files suitable for large datasets.

Ranges accepted by the CLI follow the format:
    - 'k' for a fixed value
    - 'min:max' for an inclusive range

However, any string containing one or two integers will be parsed accordingly.

Example usage:

    Generate 10 scenarios and store them as JSON Lines batches of size 100:

        python -m quantum_experiment_structures.generator \\
            --n-measurements-range 4:7 \\
            --n-values-range 2:3 \\
            --n-contexts-range 3:6 \\
            --context-size-range 2:3 \\
            --n-scenarios 10 \\
            --batch-size 100 \\
            --output-dir out


    Generate scenarios with a fixed number of measurements and contexts:

        python quantum_experiment_structures/utils/ccs_generator_script.py \\
            --n-measurements-range 5 \\
            --n-contexts-range 4 \\
            --n-values-range 2 \\
            --n-scenarios 20 \\
            --output-dir out
"""

import argparse

import quantum_experiment_structures as qes
from quantum_experiment_structures.utils import utils
from quantum_experiment_structures.data.schemas import CCS_GENERATOR_SETTINGS_SCHEMA


def _default_value(key):
    """Return the schema specified default value for 'key'."""
    key_obj = CCS_GENERATOR_SETTINGS_SCHEMA["properties"][key]
    if "allOf" in key_obj:
        value = None
        for obj in key_obj["allOf"]:
            if "default" in obj:
                value = obj["default"]
                break
    else:
        value = key_obj["default"]
    if key.endswith("range"):
        return ":".join(str(x) for x in value)
    return value


def main():
    """Parse CLI arguments, convert range strings, instantiate a CCSGenerator, and run generation.

    - Builds an argparse.ArgumentParser with options to control the random generation (number of
      measurements, value ranges, context ranges, enabling-relation parameters, RNG seed, output
      directory, batch size, number of scenarios, etc.).
    - Parses the CLI arguments into a kwargs dictionary.
    - Converts any '*_range' argument strings into integer tuples via _parse_range.
    - Constructs a CCSGenerator with the parsed kwargs and calls its .generate() method to
    perform the generation and optional output writing.
    """
    parser = argparse.ArgumentParser(
        description="Generate random causal contextuality scenarios.",
        formatter_class=utils.ArgparseFormatter,
    )
    parser.add_argument(
        "--n-measurements-range",
        default=_default_value("n_measurements_range"),
        metavar="MEASUREMENTS_RANGE",
        help="\nRange for number of measurements to generate."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-values-range",
        default=_default_value("n_values_range"),
        metavar="VALUES_RANGE",
        help="Range for number of outcomes per measurement. "
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-contexts-range",
        default=_default_value("n_contexts_range"),
        metavar="CONTEXTS_RANGE",
        help="Range for number of contexts to sample."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--context-size-range",
        default=_default_value("context_size_range"),
        metavar="CONTEXT_SIZE_RANGE",
        help="Range for size of a context."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-alternatives-range",
        default=_default_value("n_alternatives_range"),
        metavar="ENABLING_RELATIONS_RANGE",
        help="Maximum number of alternative enabling relations per measurement. "
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--enabling-relation-size-range",
        default=_default_value("enabling_relation_size_range"),
        metavar="ENABLING_RELATION_SIZE_RANGE",
        help="Range of sizes for a single enabling relation (default: number of measurements - 1)."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-samples-per-causal-structure",
        type=int,
        default=_default_value("n_samples_per_causal_structure"),
        help="Number of covers for a given a causal scenario (a set of enabling relations).",
    )
    parser.add_argument(
        "--p-has-enabled",
        type=float,
        default=_default_value("p_has_enabled"),
        help="Probability that a measurement has enabling relations. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--n-alternatives-mean",
        type=float,
        default=_default_value("n_alternatives_mean"),
        help="Mean number of alternative enabling relations (outer array).",
    )
    parser.add_argument(
        "--enabling-relation-size-mean",
        type=float,
        default=_default_value("enabling_relation_size_mean"),
        help="Mean number of events per enabling relation (inner array).",
    )
    parser.add_argument(
        "--no-lexicographic-order",
        action="store_true",
        help="Do not use lexicographic order when imposing an order to ensure acyclicity.\n"
        "This means that a random order is chosen.",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_value("output_dir"),
        help="Directory to write generated scenarios.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_default_value("batch_size"),
        help="Number of lines to write per JSON Lines in the output.\nIf this is 1, then write the "
        "data to regular JSON files instead.",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=_default_value("n_scenarios"),
        help="Number of scenarios to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_default_value("seed"),
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    # replace all the range strings with the integer tuple representation of the range
    for k, v in kwargs.items():
        if k.endswith("range"):
            kwargs[k] = utils._parse_range(v)

    # initialize the generator and generate causal contextuality scenarios
    generator = qes.CCSGenerator(**kwargs)
    generator.generate()

    # print the results if they are not saved
    if kwargs.get("output_dir") is None:
        # TODO: print so it can be piped to jq?
        for i, ccs in enumerate(generator.scenarios):
            print(f"\nCCS {i}")
            print(ccs)


if __name__ == "__main__":
    main()
