#!/usr/bin/env python3
"""Transitively close CCS rows and keep only distinct scenarios."""

import json
import os
from pathlib import Path

import pyarrow as pa
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType

import quantum_experiment_structures as qes


def to_pure_python(data):
    """Convert Spark/Arrow nested rows into pure Python objects recursively."""
    if hasattr(data, "asDict"):
        return {k: to_pure_python(v) for k, v in data.asDict().items()}
    if isinstance(data, list):
        return [to_pure_python(item) for item in data]
    return data


def close_and_refresh(iterator):
    """Close enabling relations and rebuild the human-readable field."""
    for batch in iterator:
        output_rows = []

        for row_struct in batch.to_pylist():
            pure_dict = to_pure_python(row_struct)
            scenario = qes.CausalContextualityScenario(pure_dict)

            try:
                scenario.transitively_close_enabling_relations()
                scenario.add_human_readable()
                output_rows.append(scenario.data)
            except Exception:
                pass

        if output_rows:
            out_table = pa.Table.from_pylist(output_rows, schema=batch.schema)
            yield from out_table.to_batches()


def main():
    """Run the CCS closure and deduplication job."""
    spark = SparkSession.builder.appName("close-and-deduplicate-ccs").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")

    username = os.environ.get("USER")

    schema_path = Path(
        f"/cluster/home/{username}/quantum-experiment-structures/"
        "quantum_experiment_structures/data/spark_ccs_schema.json"
    )
    with schema_path.open("r") as f:
        spark_schema = StructType.fromJson(json.load(f))

    base_path = Path(f"/cluster/scratch/{username}/quantum-experiments-tests/datasets")
    input_path = str(base_path / "all_scenarios_n4")
    output_path = str(base_path / "closed_and_distinct_scenarios_n4")

    df = spark.read.parquet(input_path)

    # 1) transitively close enabling relations
    # 2) recompute h because e changed
    closed_df = df.mapInArrow(close_and_refresh, schema=spark_schema)

    # pure Spark key derived from the human readable enabling relation + cover
    keyed_df = closed_df.withColumn(
        "scenario_key",
        F.sha2(
            F.to_json(
                F.struct(
                    F.col("h.e").alias("e"),
                    F.col("h.c").alias("c"),
                )
            ),
            256,
        ),
    )

    # distinct scenarios only
    distinct_df = (
        keyed_df.repartition(4096, "scenario_key")
        .dropDuplicates(["scenario_key"])
        .drop("scenario_key")
    )
    distinct_df.cache()
    print(f"{distinct_df.count()=}")

    distinct_df.write.mode("overwrite").parquet(output_path)
    distinct_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
