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

# TODO: add as script to pyproject.toml so it can be run when pip installing


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
        default="3:6",
        metavar="MEASUREMENTS_RANGE",
        help="\nRange for number of measurements to generate."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-values-range",
        default="2:2",
        metavar="VALUES_RANGE",
        help="Range for number of outcomes per measurement. "
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-contexts-range",
        default="2:5",
        metavar="CONTEXTS_RANGE",
        help="Range for number of contexts to sample."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--context-size-range",
        default="2:3",
        metavar="CONTEXT_SIZE_RANGE",
        help="Range for size of a context."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n-alternatives-range",
        default="0:3",
        metavar="ENABLING_RELATIONS_RANGE",
        help="Maximum number of alternative enabling relations per measurement. "
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--enabling-relation-size-range",
        default="1:4",
        metavar="ENABLING_RELATION_SIZE_RANGE",
        help="Range of sizes for a single enabling relation (default: number of measurements - 1)."
        "\nUse 'k' or 'min:max' (integers)."
        "\nYou can submit any string that has exactly one or two integers in it.",
    )
    parser.add_argument(
        "--n_samples_per_causal_structure",
        type=int,
        default=1,
        help="Number of covers to generate given a causal scenario (a set of enabling relations).",
    )
    parser.add_argument(
        "--p-has-enabled",
        type=float,
        default=0.6,
        help="Probability that a measurement has enabling relations. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--n-alternatives-mean",
        type=float,
        default=1.2,
        help="Mean number of alternative enabling relations (outer array).",
    )
    parser.add_argument(
        "--enabling-relation-size-mean",
        type=float,
        default=1.3,
        help="Mean number of events per enabling relation (inner array).",
    )
    parser.add_argument(
        "--use-lexicographic-order",
        action="store_true",
        help="Use lexicographic order when imposing an order to ensure acyclicity.\n"
        "Otherwise, a random order is chosen.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write generated scenarios.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of lines to write per JSON Lines in the output.\nIf this is 1, then write the "
        "data to regular JSON files instead.",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=1,
        help="Number of scenarios to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
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


if __name__ == "__main__":
    main()
