#!/usr/bin/env python3

import os
from pathlib import Path
from pyspark.sql import SparkSession


def calculate_partitions(count):
    """Calculate a suitable size for the partitions."""
    if count <= 100000:
        return 1

    # ideal number of partitions targeting the upper limit of 100,000 records
    p_ideal = count / 100000

    # find the smallest power of 2 that is >= p_ideal => # records in {50 000, 100 000}
    p = 1
    while p < p_ideal:
        p *= 2
    return p


def main():
    spark = SparkSession.builder.appName("repartition-all-datasets").getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    username = os.environ.get("USER")
    base_dir = Path(f"/cluster/scratch/{username}/quantum-experiments-tests")
    input_datasets_dir = base_dir / "datasets"
    output_datasets_dir = base_dir / "datasets_repartitioned"

    output_datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning input datasets in: {input_datasets_dir}", flush=True)
    dataset_paths = sorted([d for d in input_datasets_dir.iterdir() if not d.name.startswith(".")])

    print(f"Found {len(dataset_paths)} datasets to process.\n", flush=True)
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        print(f"{'=' * 70}", flush=True)
        print(f"STARTING: {dataset_name}", flush=True)

        try:
            df = spark.read.parquet(str(dataset_path))

            print(f"[{dataset_name}] Counting records...", flush=True)
            record_count = df.count()
            print(f"[{dataset_name}] Total record count: {record_count:,}", flush=True)

            num_partitions = calculate_partitions(record_count)
            expected_r_per_p = record_count / num_partitions

            print(f"[{dataset_name}] Target partitions: {num_partitions}", flush=True)
            print(
                f"[{dataset_name}] Expected records per file: {expected_r_per_p:,.2f}", flush=True
            )

            print(f"[{dataset_name}] Repartitioning dataframe...", flush=True)
            repartitioned_df = df.repartition(num_partitions)

            output_path = output_datasets_dir / dataset_name
            print(f"[{dataset_name}] Writing Parquet files to: {output_path}", flush=True)

            repartitioned_df.write.mode("overwrite").parquet(str(output_path))
            print(f"[{dataset_name}] SUCCESS!", flush=True)

        except Exception as e:
            print(f"[{dataset_name}] ERROR failed to process: {e}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print("All dataset repartitioning tasks completed successfully!", flush=True)


if __name__ == "__main__":
    main()
