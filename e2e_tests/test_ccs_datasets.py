"""End-to-end validation and summarization of CCS datasets."""

from dataclasses import asdict, dataclass
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
from quantum_experiment_structures.utils import spark_utils

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


@dataclass(frozen=True)
class DatasetMetadata:
    """Aggregate metrics collected for one dataset leaf directory."""

    dataset: str
    total_rows: int
    distinct_rows: int
    valid_rows: int
    stable_rows: int
    clean_rows: int
    flat_rows: int
    unique_causal_bridges_rows: int
    causally_secured_cover_rows: int


@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for ensemble tests."""
    num_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "*")
    session = (
        SparkSession.builder.master(f"local[{num_cpus}]")
        .appName("E2E QES Tests")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    yield session
    session.stop()


def _html_link(path, text=None):
    """Return a relative HTML link tag."""
    text = path.name if text is None else text
    return f'<a href="{html.escape(path.as_posix())}">{html.escape(text)}</a>'


def _write_dataset_html(dataset_dir, metadata_row, histogram_dir, out_root):
    """Write a simple static HTML report for one dataset."""
    out_root = Path(out_root)
    html_root = out_root / "html"
    html_root.mkdir(parents=True, exist_ok=True)

    dataset_rel = Path(metadata_row.dataset)
    page_dir = html_root / dataset_rel
    page_dir.mkdir(parents=True, exist_ok=True)

    # collect PNGs for the page
    pngs = sorted(histogram_dir.glob("*.png"))

    rows = [
        "<!doctype html>",
        "<html>",
        "<head>",
        f"<title>{html.escape(metadata_row.dataset)}</title>",
        "<meta charset='utf-8' />",
        "<style>",
        "body { font-family: sans-serif; margin: 2rem; }",
        "table { border-collapse: collapse; }",
        "td, th { border: 1px solid #ccc; padding: 0.4rem 0.7rem; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 1rem; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(metadata_row.dataset)}</h1>",
        "<h2>Metadata</h2>",
        "<table>",
    ]

    for key, value in asdict(metadata_row).items():
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")

    rows += ["</table>", "<h2>Histograms</h2>", "<div class='grid'>"]
    for png in pngs:
        img_path = png.as_posix()
        rows.append(
            "<figure>"
            f"<img src='{html.escape(img_path)}' alt='{html.escape(png.stem)}' />"
            f"<figcaption>{html.escape(png.stem)}</figcaption>"
            "</figure>"
        )
    rows += ["</div>", "</body>", "</html>"]

    page_path = page_dir / "index.html"
    page_path.write_text("\n".join(rows), encoding="utf-8")
    return page_path


def _write_results_index_html(metadata_rows, out_root):
    """Write a top-level HTML index for all dataset reports."""
    out_root = Path(out_root)
    html_root = out_root / "html"
    html_root.mkdir(parents=True, exist_ok=True)

    rows = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<title>CCS dataset validation results</title>",
        "<meta charset='utf-8' />",
        "<style>",
        "body { font-family: sans-serif; margin: 2rem; }",
        "table { border-collapse: collapse; }",
        "td, th { border: 1px solid #ccc; padding: 0.4rem 0.7rem; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>CCS dataset validation results</h1>",
        "<table>",
        "<tr><th>Dataset</th><th>Total</th><th>Distinct</th><th>Valid</th><th>Stable</th><th>Clean</th><th>Flat</th><th>Unique bridges</th><th>Secured cover</th></tr>",
    ]

    for row in sorted(metadata_rows, key=lambda r: r.dataset):
        page = Path("html") / Path(row.dataset) / "index.html"
        rows.append(
            "<tr>"
            f"<td>{_html_link(page, row.dataset)}</td>"
            f"<td>{row.total_rows}</td>"
            f"<td>{row.distinct_rows}</td>"
            f"<td>{row.valid_rows}</td>"
            f"<td>{row.stable_rows}</td>"
            f"<td>{row.clean_rows}</td>"
            f"<td>{row.flat_rows}</td>"
            f"<td>{row.unique_causal_bridges_rows}</td>"
            f"<td>{row.causally_secured_cover_rows}</td>"
            "</tr>"
        )

    rows += ["</table>", "</body>", "</html>"]
    (html_root / "index.html").write_text("\n".join(rows), encoding="utf-8")


def _dataset_leaf_dirs(root):
    """Return the leaf dataset directories under 'root'."""
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


def _dataset_label(path):
    """Return a stable label for one dataset directory."""
    return path.relative_to(DATA_ROOT).as_posix()


def _load_dataset_dir(spark, path):
    """Load all data files in a dataset directory as a Spark DataFrame."""
    path = Path(path)
    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and file_path.suffix in DATA_EXTENSIONS
        and not file_path.name.startswith("_")
    )
    assert files, f"No JSONL or parquet files found under {path}"

    suffixes = {file_path.suffix for file_path in files}
    assert len(suffixes) == 1, f"Mixed file formats are not supported under {path}"

    if ".jsonl" in suffixes:
        return spark.read.schema(SPARK_CCS_SCHEMA).json([str(file_path) for file_path in files])

    return spark.read.parquet(*[str(file_path) for file_path in files])


def _scenario_histogram_frames(df):
    """Return per-scenario metric frames for histogramming."""
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


def _canonical_count(df):
    """Count distinct rows using a canonical JSON representation."""
    canonical = df.select(F.to_json(F.struct(*df.columns)).alias("canonical"))
    return canonical.distinct().count()


def _metadata_for_leaf_dataset(df, label):
    """Compute the aggregate metadata for one leaf dataset directory."""
    total_rows = df.count()
    distinct_rows = _canonical_count(df)

    if total_rows:
        per_record = (
            df.rdd.map(lambda row: row.asDict(recursive=True))
            .map(spark_utils.record_metadata)
            .reduce(lambda left, right: {key: left[key] + right[key] for key in left})
        )
    else:
        per_record = {
            "valid_rows": 0,
            "stable_rows": 0,
            "clean_rows": 0,
            "flat_rows": 0,
            "unique_causal_bridges_rows": 0,
            "causally_secured_cover_rows": 0,
        }

    return DatasetMetadata(
        dataset=label,
        total_rows=total_rows,
        distinct_rows=distinct_rows,
        valid_rows=per_record["valid_rows"],
        stable_rows=per_record["stable_rows"],
        clean_rows=per_record["clean_rows"],
        flat_rows=per_record["flat_rows"],
        unique_causal_bridges_rows=per_record["unique_causal_bridges_rows"],
        causally_secured_cover_rows=per_record["causally_secured_cover_rows"],
    )


def _write_histogram(series, out_path, title, xlabel):
    """Write a histogram for already-collected scalar values."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if series.dropna().empty:
        return

    bins = (series.max() - series.min() + 1).astype(int)
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
    """Write a 2D histogram for one metric pair."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({xlabel: x_values, ylabel: y_values}).dropna()
    if frame.empty:
        return

    bins = (frame.max(axis=0) - frame.min(axis=0) + 1).to_numpy().astype(int)
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


def _write_metadata_artifacts(metadata_rows, out_root):
    """Persist metadata rows."""
    out_root = Path(out_root)
    metadata_root = out_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    metadata_pdf = pd.DataFrame([asdict(row) for row in metadata_rows]).sort_values("dataset")
    metadata_pdf.to_csv(out_root / "dataset_summary.csv", index=False)
    metadata_pdf.to_json(out_root / "dataset_summary.jsonl", orient="records", lines=True)


def _write_histogram_artifacts(histogram_rows, out_root):
    """Persist histogram summaries."""
    histogram_root = out_root / "histograms"
    histogram_root.mkdir(parents=True, exist_ok=True)
    for dataset, metrics in histogram_rows:
        path = histogram_root / dataset
        for metric, name in metrics:
            out_path = path / f"{name}.png"
            title = f"Distribution of {name} for {dataset}"
            _write_histogram(metric, out_path, title, name)
        for (left, left_name), (right, right_name) in itertools.combinations(metrics, 2):
            out_path = path / f"{left_name}_vs_{right_name}.png"
            title = f"Distribution of {left_name} vs. {right_name} for {dataset}"
            _write_histogram_2d(left, right, out_path, title, left_name, right_name)


@pytest.mark.parametrize("leaf_dir", _dataset_leaf_dirs(DATA_ROOT))
def test_dataset_leaf_is_valid_and_summarized(spark, leaf_dir):
    """Validate one dataset leaf directory and save its artifacts."""
    metadata_row = None
    try:
        df = _load_dataset_dir(spark, leaf_dir)
        label = _dataset_label(leaf_dir)

        metadata_row = _metadata_for_leaf_dataset(df, label)

        histogram_frames = [metric.toPandas() for metric in _scenario_histogram_frames(df)]
        flattened_row = [
            (metric_df[col], col)
            for metric_df in histogram_frames
            for col in metric_df.columns
            if col != "_sid"
        ]

        _write_metadata_artifacts([metadata_row], ARTIFACT_ROOT)
        _write_histogram_artifacts([(label, flattened_row)], ARTIFACT_ROOT)
        histogram_dir = ARTIFACT_ROOT / "histograms" / label
        _write_dataset_html(leaf_dir, metadata_row, histogram_dir, ARTIFACT_ROOT)

        assert metadata_row.total_rows >= 0
        assert metadata_row.distinct_rows >= 0
        assert metadata_row.distinct_rows <= metadata_row.total_rows
        assert metadata_row.valid_rows <= metadata_row.total_rows
        assert metadata_row.stable_rows <= metadata_row.total_rows
        assert metadata_row.clean_rows <= metadata_row.total_rows
        assert metadata_row.flat_rows <= metadata_row.total_rows
        assert metadata_row.unique_causal_bridges_rows <= metadata_row.total_rows
        assert metadata_row.causally_secured_cover_rows <= metadata_row.total_rows

    finally:
        if metadata_row is not None:
            _write_results_index_html([metadata_row], ARTIFACT_ROOT)
