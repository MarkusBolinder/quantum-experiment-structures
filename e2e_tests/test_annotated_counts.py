"""Counting and integrity verification for annotated CCS datasets."""

from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType
import quantum_experiment_structures as qes

PACKAGE_ROOT = Path(qes.__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("QES_DATA_ROOT", PACKAGE_ROOT / "e2e_tests/data"))
ARTIFACT_ROOT = Path(
    os.environ.get("QES_E2E_ARTIFACT_ROOT", PACKAGE_ROOT / "e2e_tests/generated/e2e")
)
ANNOTATED_SCHEMA_PATH = (
    PACKAGE_ROOT / "quantum_experiment_structures" / "data" / "dataset_entry_schema.json"
)

with ANNOTATED_SCHEMA_PATH.open("r", encoding="utf-8") as f:
    ANNOTATED_SCHEMA = StructType.fromJson(json.load(f))

DATA_EXTENSIONS = {".jsonl", ".parquet"}


@dataclass(frozen=True)
class AnnotatedMetrics:
    """Container for calculated metrics on an annotated dataset."""

    dataset: str
    total_rows: int
    distinct_rows: int
    flat_true: int
    clean_true: int
    stable_true: int
    unique_bridges_true: int
    secured_true: int
    stg_not_null: int
    ext_not_null: int
    dedup_from_count: int
    dedup_to_count: int


@pytest.fixture(scope="session")
def spark():
    master_url = os.environ.get("SPARK_MASTER_URL", os.environ.get("MASTER", None))
    builder = SparkSession.builder.appName("E2E Annotated Counts")

    if master_url:
        builder = builder.master(master_url)
    else:
        num_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "*")
        builder = builder.master(f"local[{num_cpus}]")

    # TODO: make these not hard-coded
    shuffle_partitions = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "2048")
    session = (
        builder.config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.python.worker.memory", "4g")
        .getOrCreate()
    )
    yield session
    session.stop()


def _is_annotated_dataset(spark, path):
    """Check whether a dataset directory contains annotated schema."""
    files = [
        p
        for p in Path(path).rglob("*")
        if p.is_file() and p.suffix in DATA_EXTENSIONS and not p.name.startswith("_")
    ]
    if not files:
        return False
    sample_file = str(files[0])
    df_sample = (
        spark.read.json(sample_file)
        if sample_file.endswith(".jsonl")
        else spark.read.parquet(sample_file)
    )
    return "flat" in df_sample.columns or "deduplicated_from" in df_sample.columns


def _dataset_leaf_dirs(root):
    root = Path(root)
    if not root.exists():
        return []

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        path = Path(dirpath)
        parts = path.parts
        if any(part.startswith(".") for part in parts) or "covers" in parts:
            continue
        if any(
            Path(name).suffix in DATA_EXTENSIONS and not name.startswith("_") for name in filenames
        ):
            candidates.append(path)

    return [
        path
        for path in sorted(candidates)
        if not any(path in other.parents for other in candidates if other != path)
    ]


def _load_annotated_dataset(spark, path):
    files = sorted(
        str(p)
        for p in Path(path).rglob("*")
        if p.is_file() and p.suffix in DATA_EXTENSIONS and not p.name.startswith("_")
    )
    assert files, f"No files found under {path}"

    if files[0].endswith(".jsonl"):
        df = spark.read.schema(ANNOTATED_SCHEMA).json(files)
    else:
        df = spark.read.parquet(*files)

    default_parallelism = spark.sparkContext.defaultParallelism
    return df.repartition(max(default_parallelism * 4, 128))


def _write_annotated_dataset_html(metrics: AnnotatedMetrics, out_dir: Path) -> Path:
    """Write an HTML file containing a table of calculated metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "<!doctype html>",
        "<html>",
        "<head>",
        f"<title>{html.escape(metrics.dataset)} - Annotated Dataset Summary</title>",
        "<meta charset='utf-8' />",
        "<style>",
        "body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #333; }",
        "h1 { font-size: 1.5rem; margin-bottom: 1rem; }",
        "table { border-collapse: collapse; width: 100%; max-width: 600px; margin-top: 1rem; }",
        "th, td { border: 1px solid #e0e0e0; padding: 0.6rem 0.9rem; text-align: left; }",
        "th { background-color: #f5f5f5; font-weight: 600; }",
        "tr:nth-child(even) { background-color: #fafafa; }",
        "td.number { text-align: right; font-family: monospace; font-size: 0.95rem; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Metrics for {html.escape(metrics.dataset)}</h1>",
        "<table>",
        "<thead><tr><th>Metric Name</th><th style='text-align: right;'>Count / Value</th></tr></thead>",
        "<tbody>",
    ]

    metrics_dict = asdict(metrics)
    for key, value in metrics_dict.items():
        if key == "dataset":
            continue
        display_name = key.replace("_", " ").title()
        formatted_val = f"{value: }" if isinstance(value, int) else str(value)
        rows.append(
            f"<tr><td><strong>{html.escape(display_name)}</strong></td>"
            f"<td class='number'>{formatted_val}</td></tr>"
        )

    rows += ["</tbody>", "</table>", "</body>", "</html>"]

    page_path = out_dir / "summary.html"
    page_path.write_text("\n".join(rows), encoding="utf-8")
    return page_path


def _write_annotated_index_html(metrics: AnnotatedMetrics, out_root: Path) -> None:
    """Update or create the top-level index HTML file for annotated datasets."""
    annotated_root = out_root / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    index_file = annotated_root / "index.html"

    rows = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<title>Annotated Datasets Overview</title>",
        "<meta charset='utf-8' />",
        "<style>",
        "body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #333; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #e0e0e0; padding: 0.5rem 0.8rem; text-align: right; }",
        "th:first-child, td:first-child { text-align: left; }",
        "th { background-color: #f5f5f5; }",
        "tr:nth-child(even) { background-color: #fafafa; }",
        "a { color: #0066cc; text-decoration: none; } a:hover { text-decoration: underline; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Annotated Datasets Summary</h1>",
        "<table>",
        "<thead><tr>",
        "<th>Dataset</th><th>Total</th><th>Distinct</th><th>Flat</th><th>Clean</th><th>Stable</th>",
        "<th>Unique Bridges</th><th>Secured</th><th>STG Not Null</th><th>EXT Not Null</th>",
        "<th>Dedup From</th><th>Dedup To</th>",
        "</tr></thead>",
        "<tbody>",
        "<tr>",
        f"<td><a href='{html.escape(metrics.dataset)}/summary.html'>{html.escape(metrics.dataset)}</a></td>",
        f"<td>{metrics.total_rows:,}</td>",
        f"<td>{metrics.distinct_rows:,}</td>",
        f"<td>{metrics.flat_true:,}</td>",
        f"<td>{metrics.clean_true:,}</td>",
        f"<td>{metrics.stable_true:,}</td>",
        f"<td>{metrics.unique_bridges_true:,}</td>",
        f"<td>{metrics.secured_true:,}</td>",
        f"<td>{metrics.stg_not_null:,}</td>",
        f"<td>{metrics.ext_not_null:,}</td>",
        f"<td>{metrics.dedup_from_count:,}</td>",
        f"<td>{metrics.dedup_to_count:,}</td>",
        "</tr>",
        "</tbody>",
        "</table>",
        "</body>",
        "</html>",
    ]
    index_file.write_text("\n".join(rows), encoding="utf-8")


@pytest.mark.parametrize("leaf_dir", _dataset_leaf_dirs(DATA_ROOT))
def test_annotated_dataset_counts(spark, leaf_dir):
    """Count boolean flags and verify structural integrity of annotated datasets."""
    if not _is_annotated_dataset(spark, leaf_dir):
        pytest.skip(f"Skipping non-annotated dataset directory: {leaf_dir}")

    df = _load_annotated_dataset(spark, leaf_dir)
    label = Path(leaf_dir).relative_to(DATA_ROOT).as_posix()

    # exclude nested structs from shuffling when counting distinct
    lightweight_df = df.drop("ccs", "ext", "stg")
    total_rows = df.count()
    distinct_rows = lightweight_df.distinct().count()

    # aggregate boolean flags and check non-null states using Spark projection pushdown
    metrics_row = df.agg(
        F.count(F.when(F.col("flat"), 1)).alias("flat_true"),
        F.count(F.when(F.col("clean"), 1)).alias("clean_true"),
        F.count(F.when(F.col("stable"), 1)).alias("stable_true"),
        F.count(F.when(F.col("unique_bridges"), 1)).alias("unique_bridges_true"),
        F.count(F.when(F.col("secured"), 1)).alias("secured_true"),
        F.count(F.when(F.col("stg").isNotNull(), 1)).alias("stg_not_null"),
        F.count(F.when(F.col("ext").isNotNull(), 1)).alias("ext_not_null"),
        F.count(F.col("deduplicated_from")).alias("dedup_from_count"),
        F.count(F.col("deduplicated_to")).alias("dedup_to_count"),
    ).collect()[0]

    metrics = AnnotatedMetrics(
        dataset=label,
        total_rows=total_rows,
        distinct_rows=distinct_rows,
        flat_true=metrics_row["flat_true"],
        clean_true=metrics_row["clean_true"],
        stable_true=metrics_row["stable_true"],
        unique_bridges_true=metrics_row["unique_bridges_true"],
        secured_true=metrics_row["secured_true"],
        stg_not_null=metrics_row["stg_not_null"],
        ext_not_null=metrics_row["ext_not_null"],
        dedup_from_count=metrics_row["dedup_from_count"],
        dedup_to_count=metrics_row["dedup_to_count"],
    )

    # save HTML artifacts
    output_dir = ARTIFACT_ROOT / "annotated" / label
    _write_annotated_dataset_html(metrics, output_dir)
    _write_annotated_index_html(metrics, ARTIFACT_ROOT)

    # verification assertions
    assert metrics.distinct_rows <= metrics.total_rows
    assert metrics.flat_true <= metrics.total_rows
    assert metrics.clean_true <= metrics.total_rows
    assert metrics.stable_true <= metrics.total_rows
    assert metrics.unique_bridges_true <= metrics.total_rows
    assert metrics.secured_true <= metrics.total_rows

    # ensure deduplication index fields are balanced
    assert metrics.dedup_from_count == metrics.dedup_to_count, (
        f"Mismatch in deduplication fields for {leaf_dir}: "
        f"deduplicated_from count ({metrics.dedup_from_count}) != "
        f"deduplicated_to count ({metrics.dedup_to_count})"
    )
