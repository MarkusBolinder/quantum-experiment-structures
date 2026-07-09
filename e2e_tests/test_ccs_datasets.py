import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import ArrayType, StructType

from quantum_experiment_structures.utils import spark_utils
from quantum_experiment_structures.causal_contextuality_scenario import (
    CausallySecuredScenario,
    StableCausalContextualityScenario,
)
from quantum_experiment_structures.spacetime_game import AlternatingSpacetimeGame

DATA_ROOT = Path(os.environ.get("QES_DATA_ROOT", "e2e_tests/data"))
ARTIFACT_ROOT = Path(os.environ.get("QES_E2E_ARTIFACT_ROOT", "e2e_tests/generated/e2e"))

SPARK_CCS_SCHEMA = StructType.fromJson(
    json.load(Path("quantum_experiment_structures/data/spark_ccs_schema.json").open())
)

EXPECTED_COUNTS = {
    ("complete", "base"): {1: 1, 2: 8, 3: 1692, 4: 341518920},
    ("complete", "causally_secured"): {1: 1, 2: 4, 3: 45, 4: 1276},
}


@dataclass(frozen=True)
class DatasetMetadata:
    """Aggregate metrics collected for one dataset leaf directory."""

    dataset: str
    total_rows: int
    distinct_rows: int
    valid_rows: int
    stable_rows: int
    causally_secured_rows: int
    stable_rows_becoming_causally_secured_after_deduplication: int


@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for ensemble tests."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("E2E QES Tests")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def _dataset_path(category, kind, n_variables=None):
    """Build the path for one dataset directory."""
    if n_variables is None:
        return DATA_ROOT / category / kind
    return DATA_ROOT / category / kind / str(n_variables)


def _leaf_dataset_dirs(root):
    """Yield every directory under `root` that contains JSONL files."""
    if not root.exists():
        return []

    for dirpath, _, filenames in os.walk(root):
        if any(name.endswith(".jsonl") for name in filenames):
            yield Path(dirpath)


def _load_jsonl_dir(spark, path):
    """Load all JSONL files in a directory as a Spark DataFrame."""
    files = sorted(path.rglob("*.jsonl"))
    assert files, f"No JSONL files found under {path}"
    return spark.read.json([str(file_path) for file_path in files], schema=SPARK_CCS_SCHEMA)


def _canonical_count(df):
    """Count distinct rows using a canonical JSON representation."""
    canonical = df.select(F.to_json(F.struct(*df.columns)).alias("canonical"))
    return canonical.distinct().count()


def _valid_rows(df, secured):
    """Validate rows with the existing Spark-based validator."""
    validated = df.rdd.mapPartitions(lambda rows: spark_utils._validate_partition(rows, secured))
    return validated.filter(lambda x: x["valid"]).map(lambda x: x["record"]).toDF(df.schema)


def _measurement_df(df):
    """Flatten the `ms` array so that measurement-level metrics can be aggregated."""
    return df.select(F.explode_outer(F.col("ms")).alias("m"))


def _relation_df(df):
    """Flatten enabling relations so relation sizes can be aggregated."""
    return (
        df.select(F.explode_outer(F.col("ms")).alias("m"))
        .where(F.col("m.e").isNotNull())
        .select(F.explode(F.col("m.e")).alias("e"))
    )
    return df.select(F.explode_outer(F.col("ms")).alias("m")).select(
        F.explode_outer(F.col("m.e")).alias("e")
    )


def _cover_df(df):
    """Flatten the top-level cover so context sizes can be aggregated."""
    return df.select(F.explode_outer(F.col("c")).alias("context"))


def _maybe_size_expr(array_col):
    """Return a size expression that tolerates missing optional arrays."""
    return F.when(F.col(array_col).isNotNull(), F.size(F.col(array_col)))


def _grouped_distribution(df, value_col):
    """Return a tiny grouped distribution as a pandas frame."""
    grouped = df.groupBy(value_col).count().orderBy(value_col).toPandas()
    grouped[value_col] = grouped[value_col].astype(int)
    grouped["count"] = grouped["count"].astype(int)
    return grouped


def _write_distribution_plot(values, value_col, out_path, title):
    """Write a simple histogram/bar plot from a grouped Spark result."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(values[value_col], values["count"])
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_histogram(series, out_path, title, xlabel):
    """Write a histogram for already-collected scalar values."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(series.to_numpy(), bins="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _metadata_for_leaf_dataset(spark, path):
    """Compute the aggregate metadata for one leaf dataset directory."""
    df = _load_jsonl_dir(spark, path)
    total_rows = df.count()
    distinct_rows = _canonical_count(df)
    valid_rows = _valid_rows(df, secured=("causally_secured" in path.as_posix())).count()

    # Heavy checks stay distributed in Spark; we only count the records that satisfy the criteria.
    stable_rdd = df.rdd.map(lambda row: row.asDict(recursive=True)).filter(
        lambda record: spark_utils._record_is_stable(record)
    )
    stable_rows = stable_rdd.count()

    causally_secured_rdd = df.rdd.map(lambda row: row.asDict(recursive=True)).filter(
        lambda record: spark_utils._record_is_causally_secured(record)
    )
    causally_secured_rows = causally_secured_rdd.count()

    # this hangs if it is not possible to deduplicate I guess, but even when done on only stable
    stable_becomes_secured = stable_rdd.filter(
        lambda record: spark_utils._record_becomes_causally_secured_after_deduplication(record)
    ).count()
    stable_becomes_secured_rows = (
        df.rdd.map(lambda row: row.asDict(recursive=True))
        .map(
            lambda record: (
                1
                if spark_utils._record_is_stable(record)
                and spark_utils._record_becomes_causally_secured_after_deduplication(record)
                else 0
            )
        )
        .sum()
    )
    print(f"{stable_becomes_secured=}, {stable_becomes_secured_rows=}")

    return DatasetMetadata(
        dataset=str(path.relative_to(DATA_ROOT)),
        total_rows=total_rows,
        distinct_rows=distinct_rows,
        valid_rows=valid_rows,
        stable_rows=stable_rows,
        causally_secured_rows=causally_secured_rows,
        stable_rows_becoming_causally_secured_after_deduplication=stable_becomes_secured_rows,
    )


@pytest.mark.parametrize(
    "category,kind,secured,complete",
    [
        ("complete", "base", False, True),
        ("complete", "causally_secured", True, True),
        ("sampled", "base", False, False),
        ("sampled", "causally_secured", True, False),
    ],
)
def test_dataset_directories_are_valid_and_unique(spark, category, kind, secured, complete):
    """Validate each dataset directory end to end."""
    base_path = _dataset_path(category, kind)
    assert base_path.exists(), f"Missing dataset directory: {base_path}"

    expected = EXPECTED_COUNTS.get((category, kind))

    # complete datasets are split by number of variables in subdirectories like complete/base/1/
    if complete:
        assert isinstance(expected, dict), f"Set EXPECTED_COUNTS[{(category, kind)!r}] to a dict"
        variable_dirs = sorted(path for path in base_path.iterdir() if path.is_dir())
        assert variable_dirs, f"No variable subdirectories found under {base_path}"

        for var_dir in variable_dirs:
            n_variables = int(var_dir.name)
            df = _load_jsonl_dir(spark, var_dir)

            total = df.count()
            distinct_total = _canonical_count(df)
            assert total == distinct_total, f"Duplicate entries found in {var_dir}"

            assert n_variables in expected, f"Missing expected count for {var_dir}"
            assert total == expected[n_variables], (
                f"Unexpected number of complete scenarios in {var_dir}"
            )

            valid_df = _valid_rows(df, secured)
            assert valid_df.count() == total

    # sampled datasets are just read from the category/kind directory directly
    else:
        df = _load_jsonl_dir(spark, base_path)

        total = df.count()

        # TODO: should we force some level of not duplicates, e.g. at least 90% has to be distinct?
        # distinct_total = _canonical_count(df)
        # assert total == distinct_total, f"Duplicate entries found in {base_path}"

        valid_df = _valid_rows(df, secured)
        assert valid_df.count() == total


@pytest.mark.parametrize(
    "category,kind",
    [
        ("sampled", "base"),
        ("sampled", "causally_secured"),
    ],
)
def test_sampled_histograms_are_written(spark, category, kind, tmp_path=ARTIFACT_ROOT):
    """Write distribution plots for sampled datasets only."""
    base_path = _dataset_path(category, kind)
    assert base_path.exists(), f"Missing dataset directory: {base_path}"

    out_root = Path(os.environ.get("QES_E2E_ARTIFACT_ROOT", tmp_path / "e2e_artifacts"))
    histogram_root = out_root / "histograms"

    leaf_dirs = list(_leaf_dataset_dirs(base_path))
    assert leaf_dirs, f"No JSONL leaf directories found under {base_path}"

    for leaf_dir in leaf_dirs:
        df = _load_jsonl_dir(spark, leaf_dir)
        rel_leaf = leaf_dir.relative_to(DATA_ROOT).as_posix()

        # dataset scalars
        summary = pd.DataFrame(
            {
                "n_measurements": df.select(F.size(F.col("ms")).alias("n_measurements"))
                .toPandas()["n_measurements"]
                .astype(int),
                "n_contexts_in_cover": df.select(F.size(F.col("c")).alias("n_contexts_in_cover"))
                .toPandas()["n_contexts_in_cover"]
                .astype(int),
            }
        )
        _write_histogram(
            summary["n_measurements"],
            histogram_root / f"{rel_leaf}__n_measurements.png",
            title=f"{rel_leaf} — number of measurements",
            xlabel="number of measurements",
        )
        _write_histogram(
            summary["n_contexts_in_cover"],
            histogram_root / f"{rel_leaf}__n_contexts_in_cover.png",
            title=f"{rel_leaf} — number of contexts in cover",
            xlabel="number of contexts in cover",
        )

        # measurement distributions
        measurement_df = _measurement_df(df)
        measurement_sizes = measurement_df.select(
            F.size(F.col("m.o")).alias("n_outcomes"),
            F.coalesce(F.size(F.col("m.e")), F.lit(0)).alias("n_enabling_relations"),
            F.when(F.col("m.c").isNotNull(), F.size(F.col("m.c")))
            .otherwise(F.lit(0))
            .alias("n_local_contexts"),
        )

        _write_distribution_plot(
            _grouped_distribution(measurement_sizes.select("n_outcomes"), "n_outcomes"),
            "n_outcomes",
            histogram_root / f"{rel_leaf}__n_outcomes.png",
            title=f"{rel_leaf} — outcomes per measurement",
        )
        _write_distribution_plot(
            _grouped_distribution(
                measurement_sizes.select("n_enabling_relations"), "n_enabling_relations"
            ),
            "n_enabling_relations",
            histogram_root / f"{rel_leaf}__n_enabling_relations.png",
            title=f"{rel_leaf} — enabling relations per measurement",
        )

        local_contexts = measurement_sizes.select("n_local_contexts").where(
            F.col("n_local_contexts").isNotNull()
        )
        if local_contexts.count() > 0:
            _write_distribution_plot(
                _grouped_distribution(local_contexts, "n_local_contexts"),
                "n_local_contexts",
                histogram_root / f"{rel_leaf}__n_local_contexts.png",
                title=f"{rel_leaf} — local contexts per measurement",
            )

        # enabling relation sizes
        relation_sizes = _relation_df(df).select(F.size(F.col("e")).alias("n_enabled_by"))
        _write_distribution_plot(
            _grouped_distribution(relation_sizes, "n_enabled_by"),
            "n_enabled_by",
            histogram_root / f"{rel_leaf}__n_enabled_by.png",
            title=f"{rel_leaf} — size of enabling relations",
        )

        # cover context sizes
        context_sizes = _cover_df(df).select(
            F.size(F.col("context")).alias("n_measurements_in_context")
        )
        _write_distribution_plot(
            _grouped_distribution(context_sizes, "n_measurements_in_context"),
            "n_measurements_in_context",
            histogram_root / f"{rel_leaf}__n_measurements_in_context.png",
            title=f"{rel_leaf} — measurements per context",
        )


@pytest.mark.parametrize(
    "category,kind",
    [
        ("complete", "base"),
        ("complete", "causally_secured"),
        ("sampled", "base"),  # this one hangs -- why? -- deduplication takes very long for large
        ("sampled", "causally_secured"),
    ],
)
def test_metadata_counts_are_written_for_all_datasets(
    spark, category, kind, tmp_path=ARTIFACT_ROOT
):
    """Write the aggregate metadata for every dataset leaf."""
    base_path = _dataset_path(category, kind)
    assert base_path.exists(), f"Missing dataset directory: {base_path}"

    out_root = Path(os.environ.get("QES_E2E_ARTIFACT_ROOT", tmp_path / "e2e_artifacts"))
    metadata_root = out_root / "metadata"

    leaf_dirs = list(_leaf_dataset_dirs(base_path))
    assert leaf_dirs, f"No JSONL leaf directories found under {base_path}"

    summaries = []
    for leaf_dir in leaf_dirs:
        summary = _metadata_for_leaf_dataset(spark, leaf_dir)
        summaries.append(summary)

        rel_leaf = leaf_dir.relative_to(DATA_ROOT).as_posix()
        out_path = metadata_root / f"{rel_leaf}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True))

    index_path = out_root / "dataset_index.json"
    index_path.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True)
    )

    assert index_path.exists()
    assert any(metadata_root.rglob("*.json")), "No metadata JSON files were created"


@pytest.mark.parametrize(
    "category,kind",
    [
        ("complete", "causally_secured"),
        ("sampled", "causally_secured"),
    ],
)
def test_causally_secured_datasets_convert_to_spacetime_and_extensive_games(spark, category, kind):
    """Check the conversion chain CCS -> spacetime game -> extensive game."""
    base_path = _dataset_path(category, kind)
    assert base_path.exists(), f"Missing dataset directory: {base_path}"

    leaf_dirs = list(_leaf_dataset_dirs(base_path))
    assert leaf_dirs, f"No JSONL leaf directories found under {base_path}"

    for leaf_dir in leaf_dirs:
        df = _load_jsonl_dir(spark, leaf_dir)
        for row in df.rdd.map(lambda row: row.asDict(recursive=True)).collect():
            scenario = CausallySecuredScenario(copy.deepcopy(row))
            assert scenario.all_checks()

            spacetime_game = scenario.to_spacetime_game()
            alternating = AlternatingSpacetimeGame(copy.deepcopy(spacetime_game))
            assert alternating.all_checks()

            extensive_game = alternating.to_extensive_game()
            assert isinstance(extensive_game, dict)


@pytest.mark.parametrize(
    "category,kind",
    [
        ("complete", "base"),
        ("complete", "causally_secured"),
        ("sampled", "base"),
        ("sampled", "causally_secured"),
    ],
)
def test_sampled_and_complete_datasets_can_be_summarized_by_spark(spark, category, kind):
    """Sanity-check the aggregation pipeline used for metadata and histograms."""
    base_path = _dataset_path(category, kind)
    assert base_path.exists(), f"Missing dataset directory: {base_path}"

    leaf_dirs = list(_leaf_dataset_dirs(base_path))
    assert leaf_dirs, f"No JSONL leaf directories found under {base_path}"

    for leaf_dir in leaf_dirs:
        df = _load_jsonl_dir(spark, leaf_dir)
        measurement_df = _measurement_df(df)
        relation_df = _relation_df(df)
        cover_df = _cover_df(df)

        assert measurement_df.count() >= 0
        assert relation_df.count() >= 0
        assert cover_df.count() >= 0
