"""Collection of utils for PySpark code that are importable in the Spark runtime environment."""

import quantum_experiment_structures as qes

from pyspark.sql.types import ArrayType, MapType, StructField, StructType


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
