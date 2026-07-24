"""Collection of utils for PySpark code that are importable in the Spark runtime environment."""

import copy

from pyspark.sql.types import ArrayType, MapType, StructField, StructType
import quantum_experiment_structures as qes


def _validate_partition(rows, secured):
    """Validate scenarios within one Spark partition."""
    scenario_cls = qes.CausallySecuredScenario if secured else qes.CausalContextualityScenario

    for row in rows:
        record = row.asDict(recursive=True)

        try:
            scenario = scenario_cls(record)
            scenario.validate()
            scenario.all_checks()

            yield {
                "valid": True,
                "record": record,
            }

        except Exception as e:
            yield {
                "valid": False,
                "record": record,
                "error": str(e),
            }


def drop_nested_fields(dtype, drop_names):
    """Remove StructField names from a Spark schema recursively.

    Args:
        dtype: A pyspark.sql.types.DataType instance.
        drop_names: Iterable of field names to remove anywhere in the schema.

    Returns:
        A new DataType with matching StructFields removed.
    """
    drop_names = set(drop_names)

    if isinstance(dtype, StructType):
        new_fields = []
        for field in dtype.fields:
            if field.name in drop_names:
                continue
            new_field_type = drop_nested_fields(field.dataType, drop_names)
            new_fields.append(
                StructField(
                    name=field.name,
                    dataType=new_field_type,
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        return StructType(new_fields)

    if isinstance(dtype, ArrayType):
        return ArrayType(
            drop_nested_fields(dtype.elementType, drop_names),
            containsNull=dtype.containsNull,
        )

    if isinstance(dtype, MapType):
        return MapType(
            keyType=drop_nested_fields(dtype.keyType, drop_names),
            valueType=drop_nested_fields(dtype.valueType, drop_names),
            valueContainsNull=dtype.valueContainsNull,
        )

    return dtype


def _record_is_stable(record):
    """Check stability using the project implementation."""
    scenario = qes.StableCausalContextualityScenario(copy.deepcopy(record))
    try:
        stable = scenario.all_checks()
    except Exception:
        stable = False
    return stable


def _record_is_causally_secured(record):
    """Check whether a record is already causally secured and convertible."""
    scenario = qes.CausallySecuredScenario(copy.deepcopy(record))
    try:
        secured = scenario.all_checks()
    except Exception:
        secured = False
    return secured


def _record_becomes_causally_secured_after_deduplication(record):
    """Check whether a stable record becomes convertible after bridge deduplication."""
    try:
        stable = qes.StableCausalContextualityScenario(copy.deepcopy(record))
        okay = stable.all_checks()
        if not okay:
            return False
        # TODO: implement some safeguard so the test does not get stuck here
        deduped = stable.deduplicate_causal_bridges()
        secured = qes.CausallySecuredScenario(copy.deepcopy(deduped.data))
        spacetime_game = secured.to_spacetime_game()
        alternating = qes.AlternatingSpacetimeGame(copy.deepcopy(spacetime_game))
        alternating.to_extensive_game()
        is_secured = secured.all_checks()
    except Exception:
        is_secured = False
    return is_secured


def _safe_check(fn):
    """Return 'True' when 'fn' completes successfully and returns truthy."""
    try:
        return fn()
    except Exception:
        return False


def record_metadata(record):
    """Compute CCS metrics for one raw record."""
    flat = all(not measurement["e"] for measurement in record["ms"])

    valid = _safe_check(lambda: qes.CausalContextualityScenario(record).all_checks())
    stable = _safe_check(lambda: qes.StableCausalContextualityScenario(record).check_stability())
    clean = _safe_check(lambda: qes.CausalContextualityScenario(record).is_scenario_clean())
    unique_causal_bridges = _safe_check(
        lambda: qes.CausallySecuredScenario(record).check_unique_causal_bridges()
    )
    causally_secured_cover = _safe_check(
        lambda: qes.CausallySecuredScenario(record).check_causally_secured_cover()
    )

    return {
        "valid_rows": int(valid),
        "stable_rows": int(stable),
        "clean_rows": int(clean),
        "flat_rows": int(flat),
        "unique_causal_bridges_rows": int(unique_causal_bridges),
        "causally_secured_cover_rows": int(causally_secured_cover),
    }
