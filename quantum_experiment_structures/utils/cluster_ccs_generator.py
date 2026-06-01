#!/usr/bin/env python3
"""Module providing a function that generates CCSs using Spark."""

import copy
import json
from pathlib import Path
import uuid

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

import quantum_experiment_structures as qes
from quantum_experiment_structures.data.schemas import CCS_GENERATOR_SETTINGS_SCHEMA
from quantum_experiment_structures.utils import utils


def run_generation(n_scenarios=None, settings=dict(), output_dir=None, partitions=None, spark=None):
    """Run distributed generation using Spark.

    Args:
        n_scenarios: Number of independent generation runs.
        settings: Input object adhering to CCS_GENERATOR_SETTINGS_SCHEMA.
        output_dir: Output directory (Spark will create partition files).
        partitions: Number of Spark partitions.
        spark: a SparkSession. If not given, one will be created.

    Returns:
        Output directory path.
    """

    if not spark:
        spark = SparkSession.builder.appName("ccs-generator").getOrCreate()

    sc = spark.sparkContext

    validator = utils.DefaultValuesValidator(CCS_GENERATOR_SETTINGS_SCHEMA)

    settings = copy.deepcopy(settings)
    validator.validate(settings)
    # Spark will handle output, so make sure that output_dir is None
    settings["output_dir"] = None

    if output_dir is None:
        output_dir = f"ccs_output_{uuid.uuid4().hex}"

    seed = settings.pop("seed", None)
    val = settings.pop("n_scenarios")
    # use settings value if no specific was given
    # but override a settings specified n_scenarios if an explicit one is given
    if n_scenarios is None:
        n_scenarios = val

    if partitions is None:
        partitions = min(sc.defaultParallelism, n_scenarios)

    # broadcast validated settings and seed to workers
    settings_bc = sc.broadcast(settings)
    seed_bc = sc.broadcast(seed)

    # calculate the workload for each input object
    n_scenarios_per_shard = n_scenarios // partitions
    n_pads = n_scenarios - n_scenarios_per_shard * partitions
    n_scenarios_col = [n_scenarios_per_shard] * partitions
    # distribute remainder to first n_pads shards
    for i in range(n_pads):
        n_scenarios_col[i] += 1
    rows = list(zip(range(partitions), n_scenarios_col))

    # create df with input describing number of scenarios for each partition
    df = spark.createDataFrame(sc.parallelize(rows, partitions), ["shard_id", "n_scenarios"])
    # FIXME: handle the Spark schema in a better way
    with Path("quantum_experiment_structures/data/spark_ccs_schema.json").open("r") as f:
        spark_schema = StructType.fromJson(json.load(f))

    def generator_df_wrapper(df_iterator):
        """Generate partition-level data, executed on Spark workers."""
        settings = settings_bc.value
        orig_seed = seed_bc.value
        column_names = [field.name for field in spark_schema.fields]

        flush_threshold = 10_000
        buffer = []

        for df in df_iterator:
            for shard_id, n_scenarios in df[["shard_id", "n_scenarios"]].itertuples(index=False):
                if orig_seed is not None:
                    local_seed = orig_seed + int(shard_id)
                else:
                    local_seed = None
                generator = qes.CCSGenerator(seed=local_seed, n_scenarios=n_scenarios, **settings)

                for ccs in generator.generate():
                    buffer.append(ccs.data)

                    if len(buffer) >= flush_threshold:
                        yield pd.DataFrame.from_records(buffer, columns=column_names)
                        buffer.clear()

        if buffer:
            yield pd.DataFrame.from_records(buffer, columns=column_names)

    result_df = df.mapInPandas(generator_df_wrapper, schema=spark_schema)

    result_df.write.mode("overwrite").parquet(output_dir)
    # NOTE: from the parquet data, you can get back to the JSON form by:
    # 1) df = pd.read_parquet(path)
    # 2) row = df.iloc[i, :]  # some row (could also do df.iterrows())
    # 3) json_data = json.loads(row.to_json())
    # 4) ccs = qes.CausalContextualityScenario(json_data)

    return output_dir


def main():
    """Initialize SparkSession and run generation."""
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    settings = {"seed": 0}
    settings = {
        "n_measurements_range": [10, 15],
        "n_values_range": [1, 5],
        "n_contexts_range": [2, 30],
        "context_size_range": [2, 6],
        "n_alternatives_range": [1, 5],
        "enabling_relation_size_range": [1, 3],
        "n_samples_per_causal_structure": 100,
        "p_has_enabled": 0.7,
        "n_alternatives_mean": 3.2,
        "enabling_relation_size_mean": 2.1,
        "n_scenarios": 2**12,  # 4096
        "seed": 0,
    }

    output_dir = run_generation(
        n_scenarios=None,
        settings=settings,
        output_dir="output",
        partitions=8,
        spark=spark,
    )

    print("Output written to:", output_dir)


if __name__ == "__main__":
    main()
