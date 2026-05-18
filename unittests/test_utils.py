"""Unit tests for the utils.utils module."""

import sys

import pytest
import json
import numpy as np

from quantum_experiment_structures.data.integer_sequences import DEDEKIND_NUMBERS
from quantum_experiment_structures.utils.utils import (
    json_file_size,
    json_size_bytes,
    get_json_obj_size,
    count_enabling_relations_no_duplicates,
    _count_enabling_relations_no_duplicates,
    count_enabling_relations,
    count_covers,
    count_causal_contextuality_scenarios,
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


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 4),
        (3, 188),
    ],
)
def test_count_enabling_relations_no_duplicates(n, expected):
    """Count the number of enabling relations without duplicate measurements."""
    result = count_enabling_relations_no_duplicates(n)
    assert result == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 5),
        (3, 835),
    ],
)
def test_count_enabling_relations(n, expected):
    """Count the number of enabling relations that may contain duplicates."""
    result = count_enabling_relations(n)
    assert result == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 9),
        (4, 114),
    ],
)
def test_count_covers(n, expected):
    """Count the number of covers (anti-chain and union covers all measurements)."""
    result = count_covers(n)
    assert result == expected


@pytest.mark.parametrize(
    ("n", "expected", "expected_with_duplicates"),
    [
        (1, 1, 1),
        (2, 8, 10),
        (3, 1692, 7515),
    ],
)
def test_count_causal_contextuality_scenarios(n, expected, expected_with_duplicates):
    """Count the number of causal contextuality scenarios."""
    result = count_causal_contextuality_scenarios(n, allow_duplicates=False)
    result_dupe = count_causal_contextuality_scenarios(n, allow_duplicates=True)
    assert result == expected
    assert result_dupe == expected_with_duplicates
    assert result_dupe >= result
    enabling = count_enabling_relations_no_duplicates(n)
    enabling_dupe = count_enabling_relations(n)
    covers = count_covers(n)
    assert enabling * covers == expected
    assert enabling_dupe * covers == expected_with_duplicates


def test_count_enabling_relations_no_duplicates_negative_n():
    """Raise for negative n when counting enabling relations without duplicates."""
    with pytest.raises(ValueError, match="n must be non-negative"):
        count_enabling_relations_no_duplicates(-1)


def test__count_enabling_relations_no_duplicates_negative_prior_measurements():
    """Raise when receivig a negative amount of prior measuremnts."""
    with pytest.raises(
        ValueError,
        match="prior_measurements must be non-negative",
    ):
        _count_enabling_relations_no_duplicates(-1)


def test_count_enabling_relations_insufficient_dedekind_numbers():
    """Complain when there are too few Dedekind numbers to calculate the answer."""
    # smallest n that violates:
    # 2 * n - 2 >= len(DEDEKIND_NUMBERS)
    n = len(DEDEKIND_NUMBERS) // 2 + 1

    with pytest.raises(
        ValueError,
        match="Not enough Dedekind numbers are known",
    ):
        count_enabling_relations(n)


def test_count_covers_insufficient_dedekind_numbers():
    """Complain some more when there are still too few Dedekind numbers to calculate the answer."""
    n = len(DEDEKIND_NUMBERS)

    with pytest.raises(
        ValueError,
        match="Not enough Dedekind numbers are known",
    ):
        count_covers(n)


def test_count_causal_contextuality_scenarios_propagates_count_enabling_relations_exception():
    """Trigger exceptions from top level function."""
    n = len(DEDEKIND_NUMBERS) // 2 + 1

    with pytest.raises(
        ValueError,
        match="Not enough Dedekind numbers are known",
    ):
        count_causal_contextuality_scenarios(
            n,
            allow_duplicates=True,
        )


def test_json_file_size_reads_exact_file_size(tmp_path):
    """Read the correct size of JSON file written to disk."""
    path = tmp_path / "payload.json"
    payload = '{"snowman":"☃","x":1}'
    path.write_text(payload, encoding="utf-8")

    assert json_file_size(path) == len(payload.encode("utf-8"))


def test_json_file_size_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        json_file_size(missing)


@pytest.mark.parametrize(
    "obj, expected",
    [
        ({"a": 1, "b": [2, 3]}, len('{"a":1,"b":[2,3]}'.encode("utf-8"))),
        ({"snowman": "☃"}, len('{"snowman":"\\u2603"}'.encode("utf-8"))),
        ([True, None, 1.5], len("[true,null,1.5]".encode("utf-8"))),
    ],
)
def test_json_size_bytes_matches_compact_json_encoding(obj, expected):
    """Determine JSON file size of an object that has not been written to disk yet."""
    assert json_size_bytes(obj) == expected


def test_get_json_obj_size_handles_nested_structures_and_shared_references(monkeypatch):
    """Find the in-memory size of nested Python objects."""

    def fake_getsizeof(x):
        """Dummy method for returning deterministic sizes of ojects."""
        if isinstance(x, dict):
            return 10
        if isinstance(x, list):
            return 20
        if isinstance(x, tuple):
            return 30
        if isinstance(x, str):
            return 4 + len(x)
        if isinstance(x, (int, float, bool)) or x is None:
            return 1
        return 7

    monkeypatch.setattr(sys, "getsizeof", fake_getsizeof)

    shared = [1]
    obj = {
        "k": [1, (2, "ab"), True, None, 3.5],
        "shared_1": shared,
        "shared_2": shared,
    }

    # dict: 10
    # keys: "k"=5, "shared_1"=12, "shared_2"=12
    # list under "k": 20 + 1 + (tuple 30 + 1 + 6) + 1 + 1 + 1 = 61
    # shared list counted once: 20 + 1 = 21
    expected = 10 + 5 + 61 + 12 + 21 + 12
    assert get_json_obj_size(obj) == expected


def test_get_json_obj_size_avoids_double_counting_cycles(monkeypatch):
    """Avoid counting cyclically referenced objects more than once."""
    monkeypatch.setattr(sys, "getsizeof", lambda x: 20)

    cycle = []
    cycle.append(cycle)

    assert get_json_obj_size(cycle) == 20


def test_get_json_obj_size_supports_dict_tuple_and_scalars(monkeypatch):
    """Test the range of support for JSON object size finding function."""

    def fake_getsizeof(x):
        """Dummy method for returning deterministic sizes of ojects."""
        if isinstance(x, dict):
            return 10
        if isinstance(x, tuple):
            return 30
        if isinstance(x, str):
            return 4 + len(x)
        if isinstance(x, (int, float, bool)) or x is None:
            return 1
        return 7

    monkeypatch.setattr(sys, "getsizeof", fake_getsizeof)

    obj = {"x": (1, "ab", False, None, 3.5)}
    expected = 10 + 5 + (30 + 1 + 6 + 1 + 1 + 1)
    assert get_json_obj_size(obj) == expected
