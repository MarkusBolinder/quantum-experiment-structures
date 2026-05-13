import json
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
import pytest

from quantum_experiment_structures.utils import spark_utils


DATA_ROOT = Path(os.environ.get("QES_DATA_ROOT", "e2e_tests/data"))

SPARK_CCS_SCHEMA = StructType.fromJson(
    json.load(Path("quantum_experiment_structures/data/spark_ccs_schema.json").open())
)

EXPECTED_COUNTS = {
    ("complete", "base"): {1: 1, 2: 8, 3: 1692, 4: 341518920},
    ("complete", "causally_secured"): {1: 1, 2: 4, 3: 57},
}


@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for ensemble tests."""
    spark = SparkSession.builder.master("local[*]").appName("E2E QES Tests").getOrCreate()
    yield spark
    spark.stop()


def _dataset_path(category, kind, n_variables=None):
    """Build the path for one dataset directory."""
    if n_variables is None:
        return DATA_ROOT / category / kind
    return DATA_ROOT / category / kind / str(n_variables)


def _load_jsonl_dir(spark, path):
    """Load all JSONL files in a directory as a Spark DataFrame."""
    files = sorted(path.rglob("*.jsonl"))
    assert files, f"No JSONL files found under {path}"
    return spark.read.json([str(file_path) for file_path in files], schema=SPARK_CCS_SCHEMA)


def _canonical_count(df):
    """Count distinct rows using a canonical JSON representation."""
    canonical = df.select(F.to_json(F.struct(*df.columns)).alias("canonical"))
    return canonical.distinct().count()


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

            validated = df.rdd.mapPartitions(
                lambda rows: spark_utils._validate_partition(rows, secured)
            )
            valid = validated.filter(lambda x: x["valid"]).map(lambda x: x["record"])
            assert valid.count() == total

    # sampled datasets are just read from the category/kind directory directly
    else:
        df = _load_jsonl_dir(spark, base_path)

        total = df.count()
        distinct_total = _canonical_count(df)

        validated = df.rdd.mapPartitions(
            lambda rows: spark_utils._validate_partition(rows, secured)
        )
        valid = validated.filter(lambda x: x["valid"]).map(lambda x: x["record"])
        assert valid.count() == total
