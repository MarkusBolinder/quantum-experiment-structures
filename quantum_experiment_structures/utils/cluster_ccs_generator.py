#!/usr/bin/env python3
"""Module providing a function that generates CCSs using Spark."""

import copy
import json
import uuid

from pyspark.sql import SparkSession

import quantum_experiment_structures as qes
from quantum_experiment_structures.data.schemas import CCS_GENERATOR_SETTINGS_SCHEMA
from quantum_experiment_structures.utils import utils


def _generate_partition(partition, settings_broadcast):
    """Run generation inside a Spark partition."""
    settings = settings_broadcast.value

    for i in partition:
        obj = copy.deepcopy(settings)

        seed = obj.get("seed")
        if seed is not None:
            obj["seed"] = seed + i

        generator = qes.CCSGenerator(**obj)

        for ccs in generator.generate():
            yield json.dumps(ccs.data)


def run_generation(settings, n_scenarios, output_dir=None, partitions=None):
    """Run distributed generation using Spark.

    Args:
        settings: Input object adhering to CCS_GENERATOR_SETTINGS_SCHEMA.
        n_scenarios: Number of independent generation runs.
        output_dir: Output directory (Spark will create partition files).
        partitions: Number of Spark partitions.

    Returns:
        Output directory path.
    """

    spark = SparkSession.builder.appName("ccs-generator").getOrCreate()

    sc = spark.sparkContext

    validator = utils.DefaultValuesValidator(CCS_GENERATOR_SETTINGS_SCHEMA)

    settings = copy.deepcopy(settings)
    validator.validate(settings)
    # Spark will handle output, so, make sure that output_dir is None
    settings["output_dir"] = None

    if output_dir is None:
        output_dir = f"ccs_output_{uuid.uuid4().hex}"

    if partitions is None:
        partitions = min(sc.defaultParallelism, n_scenarios)

    # Broadcast validated settings to workers
    settings_bc = sc.broadcast(settings)

    # Parallelize scenario indices
    seeds_rdd = sc.parallelize(range(n_scenarios), partitions)

    # Run distributed generation
    results_rdd = seeds_rdd.mapPartitions(lambda part: _generate_partition(part, settings_bc))

    # Write distributed JSONL output
    results_rdd.saveAsTextFile(output_dir)

    return output_dir


def main():
    """Initialize SparkSession and run generation."""
    _ = SparkSession.builder.master("local[*]").getOrCreate()

    settings = {"seed": 0}

    output_dir = run_generation(
        settings=settings, output_dir="output", n_scenarios=100, partitions=8
    )

    print("Output written to:", output_dir)


if __name__ == "__main__":
    main()
