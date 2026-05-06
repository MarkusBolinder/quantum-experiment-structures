"""Unit tests for the utils module."""

import pytest
import json
import numpy as np

from quantum_experiment_structures.utils.utils import (
    get_all_subsets,
    is_antichain,
    create_local_covers,
    _parse_range,
    create_anti_chain,
    DefaultValuesValidator,
    NumpyEncoder,
)


def test_get_all_subsets():
    """Test generating all non-empty subsets."""
    subsets = get_all_subsets([1, 2])
    assert len(subsets) == 3
    assert frozenset({1}) in subsets
    assert frozenset({1, 2}) in subsets


def test_is_antichain():
    """Test antichain-checking function."""
    assert is_antichain([frozenset({1}), frozenset({2})]) is True
    assert is_antichain([frozenset({1}), frozenset({1, 2})]) is False


def test_create_local_covers():
    """Verify bruteforce local cover generation."""
    covers = create_local_covers([1, 2])
    assert [[1], [2]] in covers
    assert [[1, 2]] in covers

    with pytest.raises(ValueError, match="The brute force approach scales"):
        create_local_covers([1, 2, 3, 4, 5])


def test_parse_range():
    """Try parsing some different range formats."""
    assert _parse_range("5") == [5, 5]
    assert _parse_range("1:3") == [1, 3]
    assert _parse_range("range=2..4") == [2, 4]

    with pytest.raises(ValueError, match="Expected pattern with one or two"):
        _parse_range("1 2 3")


def test_create_anti_chain():
    """Ensure the anti-chain creating works as intended."""
    contexts = [set(["A"]), set(["A", "B"]), set(["B", "C"])]
    cover = create_anti_chain(contexts)
    cover = sorted(sorted(context) for context in cover)
    assert ["A", "B"] in cover
    assert ["B", "C"] in cover
    assert ["A"] not in cover


def test_create_anti_chain_reverse_order():
    """Ensure the anti-chain creating works as intended with different subset ordering."""
    contexts = list(reversed([set(["A"]), set(["A", "B"]), set(["B", "C"])]))
    cover = create_anti_chain(contexts)
    cover = sorted(sorted(context) for context in cover)
    assert ["A", "B"] in cover
    assert ["B", "C"] in cover
    assert ["A"] not in cover


def test_extend_with_default():
    """Test the handling of composite keywords (allOf/anyOf schemas)."""
    schema = {
        "properties": {
            "prop1": {"default": "val1"},
            "prop2": {"allOf": [{"default": "val2"}]},
            "prop3": {"anyOf": [{"default": "val3"}]},
            "prop4": {"type": "string"},
        }
    }
    instance = {}
    validator = DefaultValuesValidator(schema)
    validator.validate(instance)

    assert instance["prop1"] == "val1"
    assert instance["prop2"] == "val2"
    assert instance["prop3"] == "val3"
    assert "prop4" not in instance


def test_numpy_encoder():
    """Test numpy serialization method."""
    data = {
        "int_val": np.int64(42),
        "float_val": np.float32(3.14),
        "array_val": np.array([1, 2, 3]),
    }
    json_str = json.dumps(data, cls=NumpyEncoder)
    assert "42" in json_str
    assert "3.14" in json_str
    assert "[1, 2, 3]" in json_str


def test_extend_with_default_finds_default_through_non_dict_schema_fragment():
    """Find defaults even when one combination member is a boolean schema."""
    schema = {
        "properties": {
            "prop": {
                "allOf": [
                    True,  # Triggers the non-dict branch in _find_default.
                    {"default": "val"},
                ]
            }
        }
    }
    instance = {}
    validator = DefaultValuesValidator(schema)

    properties_validator = DefaultValuesValidator.VALIDATORS["properties"]
    list(properties_validator(validator, schema["properties"], instance, schema))

    assert instance["prop"] == "val"


def test_extend_with_default_returns_early_for_non_dict_instance():
    """Return immediately when the instance being validated is not a dict."""
    schema = {"properties": {"prop": {"default": "val"}}}
    validator = DefaultValuesValidator(schema)

    properties_validator = DefaultValuesValidator.VALIDATORS["properties"]
    assert list(properties_validator(validator, schema["properties"], [], schema)) == []


def test_numpy_encoder_falls_back_to_base_json_encoder():
    """Defer to the base JSON encoder for unsupported objects."""

    class Unsupported:
        pass

    with pytest.raises(TypeError):
        json.dumps({"x": Unsupported()}, cls=NumpyEncoder)
