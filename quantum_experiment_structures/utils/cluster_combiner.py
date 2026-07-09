#!/usr/bin/env python3
"""Combine the enabling relations and local covers into complete causal contextuality scenarios."""

import json
import os
import uuid
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import LongType, StructField, StructType

import quantum_experiment_structures as qes
from quantum_experiment_structures.utils import spark_utils


def _schema_path():
    """Return the CCS Spark schema path."""
    return Path(qes.__file__).resolve().parent / "data" / "spark_ccs_schema.json"


def _relations_path(n):
    """Return the CCS enabling relations path."""
    directory = Path(qes.__file__).resolve().parent.parent / f"{n}_complete_enabling_relations"
    file = ""
    for candidate in os.listdir(directory):
        if candidate.endswith(".parquet"):
            file = candidate
            break
    return str(directory / file)


def _load_ccs_schema():
    """Load the final CCS Spark schema from JSON."""
    with _schema_path().open("r", encoding="utf-8") as f:
        return StructType.fromJson(json.load(f))


def _make_enumerator(n, names=None, covers=None, allow_duplicates=False):
    """Create a CCSEnumerator instance."""
    return qes.CCSEnumerator(n, names=names, covers=covers, allow_duplicates=allow_duplicates)


def run_enumeration(n, tasks_multiplier=8, output_dir=None, spark=None, names=None, covers=None):
    """Run distributed CCS enumeration with Spark.

    Args:
        n: Number of measurements.
        tasks_multiplier: Number of tasks per Spark core. Higher means finer-grained sharding.
        output_dir: Final output directory for CCS parquet output.
        spark: Optional existing SparkSession.
        names: Optional measurement names.
        covers: Optional explicit covers.

    Returns:
        The final output directory.
    """
    if spark is None:
        spark = SparkSession.builder.appName("ccs-enumerator").getOrCreate()

    if output_dir is None:
        output_dir = f"ccs_output_{uuid.uuid4().hex}"

    final_schema = _load_ccs_schema()
    ms_dtype = final_schema["ms"].dataType
    # get rid of the per measurement context memberships and leaves from the schema,
    # otherwise they will be populated with null, which violates the real CCS schema
    ms_dtype = spark_utils.drop_nested_fields(ms_dtype, set(["c", "l"]))
    cover_dtype = final_schema["c"].dataType

    # construct the enumerator on the driver only for the base enumeration and cover lookup
    driver_enum = _make_enumerator(n, names=names, covers=covers)

    # use the generated base rows to determine the number of partitions
    target_partitions = len(driver_enum.covers)
    print(f"{target_partitions=}")

    # broadcast immutable data to workers
    final_colnames = [field.name for field in final_schema.fields]

    relations_df = spark.read.parquet(_relations_path(n))

    # create a dataframe of the covers and broadcast them into the join
    cover_rows = [
        {"cover_id": i, "c": driver_enum.rename_cover(cover)}
        for i, cover in enumerate(driver_enum.covers)
    ]
    cover_schema = StructType(
        [
            StructField("cover_id", LongType(), False),
            StructField("c", cover_dtype, False),
        ]
    )
    covers_df = spark.createDataFrame(cover_rows, schema=cover_schema)

    paired_df = relations_df.crossJoin(F.broadcast(covers_df)).select(
        "base_id",
        F.to_json("ms").alias("ms_json"),
        "cover_id",
        F.to_json("c").alias("c_json"),
    )

    def materialize_ccs_partition(pdf_iter):
        """Turn joined rows into fully populated CCS rows."""
        flush_threshold = 500
        buffer = []

        for pdf in pdf_iter:
            for base_id, ms_json, cover_id, cover_json in pdf[
                ["base_id", "ms_json", "cover_id", "c_json"]
            ].itertuples(index=False):
                scenario = {
                    "ms": json.loads(ms_json),
                    "c": json.loads(cover_json),
                }
                try:
                    ccs = qes.CausalContextualityScenario(scenario)
                    ccs.everything()
                    buffer.append(ccs.data)
                except Exception:
                    continue

                if len(buffer) >= flush_threshold:
                    yield pd.DataFrame.from_records(buffer, columns=final_colnames)
                    buffer.clear()

        if buffer:
            yield pd.DataFrame.from_records(buffer, columns=final_colnames)

    final_df = paired_df.mapInPandas(materialize_ccs_partition, schema=final_schema)

    final_df.write.mode("overwrite").parquet(output_dir)
    return output_dir


def main():
    """Cluster execution configuration."""
    spark = SparkSession.builder.appName("ccs-enumerator-n4").getOrCreate()

    username = os.environ.get("USER")

    # TODO: document all the hardcoded stuff somewhere, e.g. README.md
    out = f"/cluster/scratch/{username}/ccs_enumeration_output_n4"

    print("Starting enumeration for n=4...")
    print(f"Output will be saved to: {out}")

    output_dir = run_enumeration(n=4, tasks_multiplier=8, output_dir=out, spark=spark)

    print("Execution completed successfully. Data written to:", output_dir)


if __name__ == "__main__":
    main()
