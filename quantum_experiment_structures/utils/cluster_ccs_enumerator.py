#!/usr/bin/env python3
"""Distributed CCS enumeration with Spark.

Pipeline:
1. Enumerate partial enabling-structure bases up to 'stop_depth'.
2. Expand each base to full enabling relations.
3. Pair each full relation set with every static cover.
4. Materialize/populate the CCS object with 'everything()'.
"""

import json
import math
import uuid
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

import quantum_experiment_structures as qes
from quantum_experiment_structures.utils import spark_utils


def _schema_path():
    """Return the CCS Spark schema path."""
    return Path(qes.__file__).resolve().parent / "data" / "spark_ccs_schema.json"


def _load_ccs_schema():
    """Load the final CCS Spark schema from JSON."""
    with _schema_path().open("r", encoding="utf-8") as f:
        return StructType.fromJson(json.load(f))


def _make_enumerator(n, names=None, covers=None, allow_duplicates=False):
    """Create a CCSEnumerator instance."""
    return qes.CCSEnumerator(n, names=names, covers=covers, allow_duplicates=allow_duplicates)


def _iter_complete_measurements(enumerator, partial_measurements, start_i):
    """Complete a partial enabling structure from measurement index 'start_i' onward."""
    if start_i == enumerator.n_measurements:
        yield partial_measurements
        return

    prior_names = enumerator.names[:start_i]
    for relation in enumerator.iter_enabling_relations(prior_names):
        ms_obj = {
            "m": enumerator.names[start_i],
            "e": relation,
            "o": [{"v": 0}, {"v": 1}],
        }
        yield from _iter_complete_measurements(
            enumerator,
            partial_measurements + [ms_obj],
            start_i + 1,
        )


def _enumerate_bases(enumerator, stop_depth):
    """Enumerate partial bases up to 'stop_depth' using the enumerator's existing recursion.

    This assumes the enumerator has the 'enumerate_causal_structures(i, measurements)' method
    and that it respects 'self.stop_depth', as in your rewritten version.
    """
    enumerator.stop_depth = min(stop_depth, enumerator.n_measurements)

    bases = []
    for base_id, item in enumerate(enumerator.enumerate_causal_structures(0, [])):
        # NOTE: the 'ms' here are not complete, and should not be part of the final output
        bases.append({"base_id": base_id, "ms_json": json.dumps(item)})
    return bases


def run_enumeration(
    n,
    stop_depth=2,
    tasks_multiplier=8,
    output_dir=None,
    relations_output_dir=None,
    spark=None,
    names=None,
    covers=None,
    allow_duplicates=False,
    only_relations=False,
):
    """Run distributed CCS enumeration with Spark.

    Args:
        n: Number of measurements.
        stop_depth: Depth at which to stop the base-enumeration step.
            This is the partition seed for the job.
        tasks_multiplier: Number of tasks per Spark core. Higher means finer-grained sharding.
        output_dir: Final output directory for CCS parquet output.
        relations_output_dir: Optional parquet directory for the intermediate full-relation table.
        spark: Optional existing SparkSession.
        names: Optional measurement names.
        covers: Optional explicit covers.
        allow_duplicates: Whether enabling relations may contain duplicate measurements.
        only_relations: Whether to only compute and write the enabling relations.

    Returns:
        The final output directory.
    """
    if spark is None:
        spark = SparkSession.builder.appName("ccs-enumerator").getOrCreate()

    sc = spark.sparkContext

    if output_dir is None:
        output_dir = f"ccs_output_{uuid.uuid4().hex}"

    if only_relations and relations_output_dir is None:
        relations_output_dir = f"ccs_output_{uuid.uuid4().hex}"

    final_schema = _load_ccs_schema()
    ms_dtype = final_schema["ms"].dataType
    # get rid of the per measurement context memberships and leaves from the schema,
    # otherwise they will be populated with null, which violates the real CCS schema
    ms_dtype = spark_utils.drop_nested_fields(ms_dtype, set(["c", "l"]))
    cover_dtype = final_schema["c"].dataType

    # construct the enumerator on the driver only for the base enumeration and cover lookup
    driver_enum = _make_enumerator(n, names=names, covers=covers, allow_duplicates=allow_duplicates)

    base_rows = _enumerate_bases(driver_enum, stop_depth=stop_depth)
    if not base_rows:
        return output_dir

    # use the generated base rows to determine the number of partitions
    target_partitions = min(len(base_rows), max(1, sc.defaultParallelism * tasks_multiplier))
    print(f"{target_partitions=}, {sc.defaultParallelism=}, {len(base_rows)=}")
    bases_per_partition = max(1, math.ceil(len(base_rows) / target_partitions))
    for row in base_rows:
        row["shard_id"] = row["base_id"] // bases_per_partition

    base_schema = StructType(
        [
            StructField("base_id", LongType(), False),
            StructField("shard_id", LongType(), False),
            StructField("ms_json", StringType(), False),
        ]
    )

    base_df = spark.createDataFrame(base_rows, schema=base_schema).repartition("shard_id")

    # broadcast immutable data to workers
    names_bc = sc.broadcast(driver_enum.names)
    covers_bc = sc.broadcast(driver_enum.covers)
    allow_duplicates_bc = sc.broadcast(allow_duplicates)
    n_bc = sc.broadcast(n)
    stop_depth_bc = sc.broadcast(min(stop_depth, n))
    final_colnames = [field.name for field in final_schema.fields]
    relation_colnames = ["base_id", "ms"]

    def _build_worker_enumerator():
        """Rebuild a lightweight enumerator inside each worker."""
        return _make_enumerator(
            n_bc.value,
            names=names_bc.value,
            covers=covers_bc.value,
            allow_duplicates=allow_duplicates_bc.value,
        )

    def expand_base_partition(pdf_iter):
        """Expand partial bases into full enabling-structure rows."""
        enum = _build_worker_enumerator()
        flush_threshold = 1000
        buffer = []

        i = 0
        for pdf in pdf_iter:
            for base_id, ms_json in pdf[["base_id", "ms_json"]].itertuples(index=False):
                ms = json.loads(ms_json)
                for full_ms in _iter_complete_measurements(enum, list(ms), stop_depth_bc.value):
                    i += 1
                    buffer.append({"base_id": int(base_id), "ms": full_ms})
                    if len(buffer) >= flush_threshold:
                        yield pd.DataFrame.from_records(buffer, columns=relation_colnames)
                        buffer.clear()
                    if i % 100000 == 0:
                        print(i)

        if buffer:
            print(i)
            yield pd.DataFrame.from_records(buffer, columns=relation_colnames)

    relations_schema = StructType(
        [
            StructField("base_id", LongType(), False),
            StructField("ms", ms_dtype, False),
        ]
    )

    relations_df = base_df.mapInPandas(expand_base_partition, schema=relations_schema)

    if relations_output_dir is not None:
        relations_df.write.mode("overwrite").parquet(relations_output_dir)

    if only_relations:
        return relations_output_dir

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
    """Example local execution."""
    spark = SparkSession.builder.master("local[*]").appName("ccs-enumerator").getOrCreate()

    only_relations = True
    output_dir = run_enumeration(
        n=4,
        stop_depth=3,
        tasks_multiplier=8,
        output_dir="ccs_enumeration_output",
        relations_output_dir="ccs_relations_output",
        spark=spark,
        allow_duplicates=False,
        only_relations=only_relations,
    )
    info_text = (
        "Causal structures (enabling relations) written to:"
        if only_relations
        else ("Full CCSs (covers + enabling relations) written to:")
    )
    print(info_text, output_dir)


if __name__ == "__main__":
    main()
