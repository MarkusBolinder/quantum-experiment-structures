"""End-to-end validation and histogram reporting for raw CCS datasets."""

import html
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType
import quantum_experiment_structures as qes

PACKAGE_ROOT = Path(qes.__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("QES_DATA_ROOT", PACKAGE_ROOT / "e2e_tests/data"))
ARTIFACT_ROOT = Path(
    os.environ.get("QES_E2E_ARTIFACT_ROOT", PACKAGE_ROOT / "e2e_tests/generated/e2e")
)

with (PACKAGE_ROOT / "quantum_experiment_structures" / "data" / "spark_ccs_schema.json").open(
    "r"
) as f:
    SPARK_CCS_SCHEMA = StructType.fromJson(json.load(f))

DATA_EXTENSIONS = {".jsonl", ".parquet"}


@pytest.fixture(scope="session")
def spark():
    master_url = os.environ.get("SPARK_MASTER_URL", os.environ.get("MASTER", None))
    builder = SparkSession.builder.appName("E2E CCS Histograms")

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
    """Check whether a dataset directory contains annotated schema or raw CCS."""
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


def _load_raw_ccs_dataset(spark, path):
    files = sorted(
        str(p)
        for p in Path(path).rglob("*")
        if p.is_file() and p.suffix in DATA_EXTENSIONS and not p.name.startswith("_")
    )
    assert files, f"No data files found under {path}"

    if files[0].endswith(".jsonl"):
        df = spark.read.schema(SPARK_CCS_SCHEMA).json(files)
    else:
        df = spark.read.parquet(*files)

    default_parallelism = spark.sparkContext.defaultParallelism
    return df.repartition(max(default_parallelism * 4, 128))


def _scenario_histogram_frames(df):
    """Calculate metrics of per scenario properties."""
    scenario_df = df.withColumn("_sid", F.monotonically_increasing_id())

    scalars = scenario_df.select(
        "_sid",
        F.size("ms").alias("n_measurements"),
        F.size("c").alias("n_contexts_in_cover"),
    )
    measurements_in_context = scenario_df.select(
        "_sid", F.explode_outer("c").alias("context")
    ).select("_sid", F.size("context").alias("n_measurements_in_context"))

    enabling_sets_per_measurement = scenario_df.select(
        "_sid", F.explode_outer("ms").alias("m")
    ).select("_sid", F.size("m.e").alias("n_enabling_sets"))

    events_per_enabling_set = (
        scenario_df.select("_sid", F.explode_outer("ms").alias("m"))
        .select("_sid", F.explode_outer("m.e").alias("e"))
        .select("_sid", F.size("e").alias("n_events_in_enabling_set"))
    )

    return scalars, measurements_in_context, enabling_sets_per_measurement, events_per_enabling_set


def _write_histogram(series, out_path, title, xlabel):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if series.dropna().empty:
        return

    bins = int(series.max() - series.min())
    plt.figure(figsize=(10, 6))
    plt.hist(series.to_numpy(), bins=bins, edgecolor="black", linewidth=0.8)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.gca().set_axisbelow(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_histogram_2d(x_values, y_values, out_path, title, xlabel, ylabel):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({xlabel: x_values, ylabel: y_values}).dropna()
    if frame.empty:
        return

    bins = (frame.max(axis=0) - frame.min(axis=0)).to_numpy().astype(int)
    plt.figure(figsize=(10, 6))
    plt.hist2d(frame[xlabel].to_numpy(), frame[ylabel].to_numpy(), bins=bins)
    plt.colorbar(label="count")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.gca().set_axisbelow(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_dataset_html(histogram_dir):
    pngs = sorted(histogram_dir.glob("*.png"))

    rows = ["</table><h2>Histograms</h2><div class='grid'>"]
    for png in pngs:
        rows.append(
            f"<figure><img src='{html.escape(png.name)}'/><figcaption>{html.escape(png.stem)}</figcaption></figure>"
        )
    rows += ["</div></body></html>"]

    page_path = histogram_dir / "summary.html"
    page_path.write_text("\n".join(rows), encoding="utf-8")


@pytest.mark.parametrize("leaf_dir", _dataset_leaf_dirs(DATA_ROOT))
def test_ccs_histogram_generation(spark, leaf_dir):
    """Generate histograms for raw CCS datasets."""
    if _is_annotated_dataset(spark, leaf_dir):
        pytest.skip(f"Skipping histogram test for annotated dataset: {leaf_dir}")

    df = _load_raw_ccs_dataset(spark, leaf_dir)
    label = Path(leaf_dir).relative_to(DATA_ROOT).as_posix()

    # downsample large dataset to avoid OOM
    large_datasets = [
        "all_scenarios_n4",
        "stable_scenarios_n4",
        "stable_but_not_deduplicated_scenarios_n4",
    ]
    if any(dataset in leaf_dir.parts for dataset in large_datasets):
        # 340 000 000 -> 340 000
        # 50 000 000 -> 50 000
        sample_df = df.sample(False, 0.001, seed=42)
    else:
        sample_df = df

    histogram_frames = [metric.toPandas() for metric in _scenario_histogram_frames(sample_df)]
    flattened_metrics = [
        (metric_df[col], col)
        for metric_df in histogram_frames
        for col in metric_df.columns
        if col != "_sid"
    ]

    histogram_dir = ARTIFACT_ROOT / "histograms" / label
    for metric, name in flattened_metrics:
        _write_histogram(metric, histogram_dir / f"{name}.png", f"Distribution of {name}", name)
    for (left, l_name), (right, r_name) in itertools.combinations(flattened_metrics, 2):
        _write_histogram_2d(
            left,
            right,
            histogram_dir / f"{l_name}_vs_{r_name}.png",
            f"{l_name} vs {r_name}",
            l_name,
            r_name,
        )

    _write_dataset_html(histogram_dir)
