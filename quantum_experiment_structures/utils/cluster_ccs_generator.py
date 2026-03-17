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


def run_generation(n_scenarios, settings, output_dir=None, partitions=None):
    """Run distributed generation using Spark.

    Args:
        n_scenarios: Number of independent generation runs.
        settings: Input object adhering to CCS_GENERATOR_SETTINGS_SCHEMA.
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
    # Spark will handle output, so make sure that output_dir is None
    settings["output_dir"] = None

    if output_dir is None:
        output_dir = f"ccs_output_{uuid.uuid4().hex}"

    if partitions is None:
        partitions = min(sc.defaultParallelism, n_scenarios)

    seed = settings.pop("seed")
    del settings["n_scenarios"]
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
    # FIXME: handle the Spark schema in a better way -- should it also support 'n' (notes)?
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
                local_seed = orig_seed + int(shard_id)
                generator = qes.CCSGenerator(seed=local_seed, n_scenarios=n_scenarios, **settings)

                for ccs in generator.generate():
                    buffer.append(ccs.data)

                    if len(buffer) >= flush_threshold:
                        yield pd.DataFrame.from_records(buffer, columns=column_names)
                        buffer.clear()

        if buffer:
            yield pd.DataFrame.from_records(buffer, columns=column_names)

    # distributed generation
    result_df = df.mapInPandas(generator_df_wrapper, schema=spark_schema)

    # write output
    result_df.write.mode("overwrite").parquet(output_dir)
    # NOTE: from the parquet data, you can get back to the JSON form by:
    # 1) df = pd.read_parquet(path)
    # 2) row = df.iloc[i, :]  # some row (could also do df.iterrows())
    # 3) json_data = json.loads(row.to_json())
    # 4) ccs = qes.CausalContextualityScenario(json_data)

    return output_dir


def main():
    """Initialize SparkSession and run generation."""
    _ = SparkSession.builder.master("local[*]").getOrCreate()

    settings = {"seed": 0}

    output_dir = run_generation(
        n_scenarios=1000, settings=settings, output_dir="output", partitions=8
    )

    print("Output written to:", output_dir)


if __name__ == "__main__":
    main()
