"""Collection of utils for PySpark code that are importable in the Spark runtime environment."""

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
