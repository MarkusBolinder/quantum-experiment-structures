#!/usr/bin/env python3

import gc
import json
import os
from pathlib import Path
import resource

import pyarrow as pa
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import quantum_experiment_structures as qes


def to_pure_python(data):
    """Convert internal Spark Row structures into pure Python dicts/lists recursively."""
    if hasattr(data, "asDict"):
        return {k: to_pure_python(v) for k, v in data.asDict().items()}
    elif isinstance(data, list):
        return [to_pure_python(item) for item in data]
    else:
        return data


def log_memory_and_gc(label, limit, warning_threshold=0.8):
    """Monitor worker RSS memory and force garbage collection if approaching limit."""
    # value is given in kB
    current_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"[{label}] Python Worker Core RAM: {current_rss_gb:.3f} GB", flush=True)

    if current_rss_gb > warning_threshold * limit:
        print(
            f"[{label}] WARNING: Memory threshold exceeded ({current_rss_gb:.3f} GB)."
            "Forcing explicit garbage collection...",
            flush=True,
        )
        gc.collect()


def main():
    spark = SparkSession.builder.appName("analyze-datasets").getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    username = os.environ.get("USER")

    spark_schema_path = Path(
        f"/cluster/home/{username}/quantum-experiment-structures/"
        "quantum_experiment_structures/data/spark_ccs_schema.json"
    )
    with Path(spark_schema_path).open("r") as f:
        spark_schema = StructType.fromJson(json.load(f))

    base_path = Path(f"/cluster/scratch/{username}/quantum-experiments-tests/datasets")
    assert os.path.exists(base_path)

    input_path = str(base_path / "all_scenarios_n4")

    # output paths for all derived datasets
    clean_path = str(base_path / "clean_scenarios_n4")
    stable_path = str(base_path / "stable_scenarios_n4")
    clean_and_stable_path = str(base_path / "clean_and_stable_scenarios_n4")
    deduplicated_path = str(base_path / "deduplicated_scenarios_n4")
    deduplicated_from_clean_and_stable_path = str(
        base_path / "deduplicated_from_clean_and_stable_path"
    )
    secured_path = str(base_path / "secured_scenarios_n4")
    secured_after_deduplication_path = str(base_path / "secured_after_deduplication_scenarios_n4")

    stable_but_not_deduplicated_path = str(base_path / "stable_but_not_deduplicated_scenarios_n4")
    clean_but_not_deduplicated_path = str(base_path / "clean_but_not_deduplicated_scenarios_n4")
    clean_and_stable_but_not_deduplicated_path = str(
        base_path / "clean_and_stable_but_not_deduplicated_scenarios_n4"
    )
    deduplicated_but_not_clean_path = str(base_path / "deduplicated_but_not_clean_scenarios_n4")
    not_secured_after_deduplicated_path = str(
        base_path / "not_secured_after_deduplication_scenarios_n4"
    )

    print("Reading complete raw dataset...", flush=True)
    all_df = spark.read.parquet(input_path).repartition(1024 * 16)  # 341 518 920
    print("Initial DataFrame created.", flush=True)

    def process_valid(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.CausalContextualityScenario(pure_dict)
                try:
                    scenario.everything()
                    output_rows.append(scenario.data)
                except Exception:
                    pass
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_clean(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.CausalContextualityScenario(pure_dict)
                try:
                    if scenario.is_scenario_clean():
                        output_rows.append(scenario.data)
                except Exception:
                    pass
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_stable(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.StableCausalContextualityScenario(pure_dict)
                try:
                    scenario.everything()
                    output_rows.append(scenario.data)
                except Exception:
                    pass
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_deduplicate(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                stable_scenario = qes.StableCausalContextualityScenario(pure_dict)
                try:
                    dedup = stable_scenario.deduplicate_causal_bridges()
                    dedup.everything()
                    output_rows.append(dedup.data)
                except Exception:
                    pass
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_secured(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.CausallySecuredScenario(pure_dict)
                try:
                    scenario.everything()
                    output_rows.append(scenario.data)
                except Exception:
                    pass
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_not_deduplicatable(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                stable_scenario = qes.StableCausalContextualityScenario(pure_dict)
                try:
                    dedup = stable_scenario.deduplicate_causal_bridges()
                    dedup.everything()
                except Exception:
                    output_rows.append(pure_dict)
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_deduplicated_but_not_clean(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.CausalContextualityScenario(pure_dict)
                try:
                    if not scenario.is_scenario_clean():
                        output_rows.append(pure_dict)
                except Exception:
                    output_rows.append(pure_dict)
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    def process_not_secured(iterator):
        for batch in iterator:
            output_rows = []
            for row_struct in batch.to_pylist():
                pure_dict = to_pure_python(row_struct)
                scenario = qes.CausallySecuredScenario(pure_dict)
                try:
                    scenario.everything()
                except Exception:
                    output_rows.append(pure_dict)
            if output_rows:
                for out_batch in pa.Table.from_pylist(
                    output_rows, schema=batch.schema
                ).to_batches():
                    yield out_batch

    print("Computing derived datasets and set transformations...", flush=True)

    # valid_df = all_df.mapInArrow(process_valid, schema=spark_schema)
    valid_df = all_df
    valid_df.cache()

    clean_df = valid_df.mapInArrow(process_clean, schema=spark_schema).repartition(2048)
    clean_df.cache()

    stable_df = valid_df.mapInArrow(process_stable, schema=spark_schema).repartition(2048)
    stable_df.cache()

    clean_and_stable_df = clean_df.mapInArrow(process_stable, schema=spark_schema).repartition(128)
    clean_and_stable_df.cache()

    deduplicated_df = stable_df.mapInArrow(process_deduplicate, schema=spark_schema).repartition(
        128
    )
    deduplicated_df.cache()

    deduplicated_from_clean_and_stable_df = clean_and_stable_df.mapInArrow(
        process_deduplicate, schema=spark_schema
    ).repartition(128)
    deduplicated_from_clean_and_stable_df.cache()

    secured_df = stable_df.mapInArrow(process_secured, schema=spark_schema).repartition(2)

    secured_after_deduplication_df = deduplicated_df.mapInArrow(
        process_secured, schema=spark_schema
    ).repartition(128)

    stable_but_not_deduplicated_df = stable_df.mapInArrow(
        process_not_deduplicatable, schema=spark_schema
    )
    clean_but_not_deduplicated_df = clean_df.mapInArrow(
        process_not_deduplicatable, schema=spark_schema
    )
    clean_and_stable_but_not_deduplicated_df = clean_and_stable_df.mapInArrow(
        process_not_deduplicatable, schema=spark_schema
    )
    deduplicated_but_not_clean_df = deduplicated_df.mapInArrow(
        process_deduplicated_but_not_clean, schema=spark_schema
    )
    not_secured_after_deduplicated_df = deduplicated_df.mapInArrow(
        process_not_secured, schema=spark_schema
    )

    print("Writing derived datasets to disk...")

    print("Calculating final record counts and writing to disk...")
    # these datasets require valid_df / all_df
    total_count = valid_df.count()
    clean_count = clean_df.count()
    clean_df.write.mode("overwrite").parquet(clean_path)
    stable_count = stable_df.count()
    stable_df.write.mode("overwrite").parquet(stable_path)
    valid_df.unpersist()
    print("Partial results written [1/5]...")
    print(f"{total_count=}")
    print(f"{clean_count=}")
    print(f"{stable_count=}")

    # these datasets require clean_df
    clean_and_stable_count = clean_and_stable_df.count()
    clean_and_stable_df.write.mode("overwrite").parquet(clean_and_stable_path)
    clean_but_not_deduplicated_count = clean_but_not_deduplicated_df.count()
    clean_but_not_deduplicated_df.write.mode("overwrite").parquet(clean_but_not_deduplicated_path)
    clean_df.unpersist()
    print("Partial results written [2/5]...")
    print(f"{clean_and_stable_count=}")
    print(f"{clean_but_not_deduplicated_count=}")

    # these datasets require stable_df
    deduplicated_count = deduplicated_df.count()
    deduplicated_df.write.mode("overwrite").parquet(deduplicated_path)
    secured_count = secured_df.count()
    secured_df.write.mode("overwrite").parquet(secured_path)
    stable_but_not_deduplicated_count = stable_but_not_deduplicated_df.count()
    stable_but_not_deduplicated_df.write.mode("overwrite").parquet(stable_but_not_deduplicated_path)
    stable_df.unpersist()
    print("Partial results written [3/5]...")
    print(f"{deduplicated_count=}")
    print(f"{secured_count=}")
    print(f"{stable_but_not_deduplicated_count=}")

    # these datasets require clean_and_stable_df
    deduplicated_from_clean_and_stable_count = deduplicated_from_clean_and_stable_df.count()
    deduplicated_from_clean_and_stable_df.write.mode("overwrite").parquet(
        deduplicated_from_clean_and_stable_path
    )
    clean_and_stable_but_not_deduplicated_count = clean_and_stable_but_not_deduplicated_df.count()
    clean_and_stable_but_not_deduplicated_df.write.mode("overwrite").parquet(
        clean_and_stable_but_not_deduplicated_path
    )
    clean_and_stable_df.unpersist()
    print("Partial results written [4/5]...")
    print(f"{deduplicated_from_clean_and_stable_count=}")
    print(f"{clean_and_stable_but_not_deduplicated_count=}")

    # these datasets require deduplicated_df
    deduplicated_but_not_clean_count = deduplicated_but_not_clean_df.count()
    deduplicated_but_not_clean_df.write.mode("overwrite").parquet(deduplicated_but_not_clean_path)
    secured_after_deduplication_count = secured_after_deduplication_df.count()
    secured_after_deduplication_df.write.mode("overwrite").parquet(secured_after_deduplication_path)
    not_secured_after_deduplicated_count = not_secured_after_deduplicated_df.count()
    not_secured_after_deduplicated_df.write.mode("overwrite").parquet(
        not_secured_after_deduplicated_path
    )
    deduplicated_df.unpersist()
    print("Partial results written [5/5]...")
    print(f"{deduplicated_but_not_clean_count=}")
    print(f"{secured_after_deduplication_count=}")
    print(f"{not_secured_after_deduplicated_count=}")

    # print all results again
    print("\n================ COUNTS ================")
    print(f"{total_count=}")
    print(f"{clean_count=}")
    print(f"{stable_count=}")
    print(f"{clean_and_stable_count=}")
    print(f"{deduplicated_count=}")
    print(f"{deduplicated_from_clean_and_stable_count=}")
    print(f"{secured_count=}")
    print(f"{secured_after_deduplication_count=}")
    print("\n============== NEW RESULTS =============")
    print(f"{stable_but_not_deduplicated_count=}")
    print(f"{clean_but_not_deduplicated_count=}")
    print(f"{clean_and_stable_but_not_deduplicated_count=}")
    print(f"{deduplicated_but_not_clean_count=}")
    print(f"{not_secured_after_deduplicated_count=}")


if __name__ == "__main__":
    main()
