"""Unit tests for the causal_contextuality_scenario module."""

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import pytest

from quantum_experiment_structures.causal_contextuality_scenario import (
    CausalContextualityScenario,
    CausallySecuredScenario,
    StableCausalContextualityScenario,
)
from quantum_experiment_structures.spacetime_game import AlternatingSpacetimeGame
from quantum_experiment_structures.utils import utils


def _base_valid_ccs_data():
    """Build a minimal CCS instance that passes the base checks."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "B"]],
            },
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["A", "B"]],
            },
        ],
        "c": [["A", "B"]],
    }


def _stable_valid_ccs_data():
    """Build a minimal stable CCS instance."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "C"]],
            },
            {
                "m": "B",
                "e": [],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": False}],
                "c": [["B", "C"]],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}], [{"m": "B", "v": 1}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["A", "C"], ["B", "C"]],
            },
        ],
        "c": [["A", "C"], ["B", "C"]],
    }


def _complex_stable_valid_ccs_data():
    """Build a more complex stable scenario base."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}], [{"m": "B", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "D",
                "e": [[{"m": "C", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "C", "D"], ["B", "C"]],
    }


def _secured_valid_ccs_data():
    """Build a minimal causally secured CCS instance."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
            {
                "m": "C",
                "e": [[{"m": "B", "v": 0}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
        ],
        "c": [["A", "B", "C"]],
    }


@pytest.fixture
def valid_ccs_data():
    """Return a minimal CCS instance that passes the base checks."""
    return _base_valid_ccs_data()


@pytest.fixture
def ccs_consistency_fail_data(valid_ccs_data):
    """Return a CCS instance with a duplicated measurement inside one bridge."""
    data = deepcopy(valid_ccs_data)
    data["ms"][1]["e"] = [[{"m": "A", "v": 0}, {"m": "A", "v": 1}]]
    return data


@pytest.fixture
def ccs_contexts_fail_data(valid_ccs_data):
    """Return a CCS instance whose memberships disagree with the cover."""
    data = deepcopy(valid_ccs_data)
    data["ms"][1]["c"] = [["B"]]
    return data


@pytest.fixture
def ccs_cover_fail_data(valid_ccs_data):
    """Return a CCS instance whose cover misses one measurement."""
    data = deepcopy(valid_ccs_data)
    del data["ms"][0]["c"]
    del data["ms"][1]["c"]
    data["c"] = [["A"]]
    return data


@pytest.fixture
def ccs_leaves_fail_data(valid_ccs_data):
    """Return a CCS instance with an incorrect leaf annotation."""
    data = deepcopy(valid_ccs_data)
    data["ms"][0]["o"][0]["l"] = True
    return data


@pytest.fixture
def ccs_unique_values_fail_data(valid_ccs_data):
    """Return a CCS instance with duplicated outcome values."""
    data = deepcopy(valid_ccs_data)
    data["ms"][0]["o"] = [{"v": 0, "l": False}, {"v": 0, "l": True}]
    return data


@pytest.fixture
def ccs_antichain_fail_data():
    """Return a CCS instance with a nested cover."""
    data = _base_valid_ccs_data()
    data["ms"][0]["c"] = [["A", "B"], ["A"]]
    data["ms"][1]["c"] = [["A", "B"], ["A"]]
    data["c"] = [["A", "B"], ["A"]]
    return data


@pytest.fixture
def ccs_all_checks_fail_data():
    """Return a CCS instance that fails all_checks via check_anti_chain."""
    data = _base_valid_ccs_data()
    del data["ms"][0]["c"]
    del data["ms"][1]["c"]
    data["c"] = [["A", "B"], ["A"]]
    return data


@pytest.fixture
def stable_ccs_data():
    """Return a stable CCS instance with two compatible causal bridges."""
    return _stable_valid_ccs_data()


@pytest.fixture
def stable_ccs_fail_data():
    """Return an unstable CCS instance with both bridges in one facet."""
    data = _stable_valid_ccs_data()
    data["ms"][0]["c"] = [["A", "B", "C"]]
    data["ms"][1]["c"] = [["A", "B", "C"]]
    data["ms"][2]["c"] = [["A", "B", "C"]]
    data["c"] = [["A", "B", "C"]]
    return data


@pytest.fixture
def two_duplications_ccs_data():
    """Return a stable CCS instance that will duplicate two measurements."""
    data = deepcopy(_complex_stable_valid_ccs_data())
    data["c"][1].append("D")
    return data


@pytest.fixture
def one_duplication_based_on_enabling_relations():
    """Return stable CCS where enabling relations indicate only one duplication is needed."""
    data = deepcopy(_complex_stable_valid_ccs_data())
    for measurement in data["ms"]:
        if measurement["m"] == "D":
            measurement["e"][0].append({"m": "A", "v": 0})
    return data


@pytest.fixture
def one_duplication_based_on_cover():
    """Return stable CCS where the cover indicates only one duplication is needed."""
    return deepcopy(_complex_stable_valid_ccs_data())


@pytest.fixture
def secured_ccs_data():
    """Return a causally secured CCS instance that passes all checks."""
    return _secured_valid_ccs_data()


@pytest.fixture
def secured_unique_bridges_fail_data():
    """Return a CCS instance with two non-contradictory bridges."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
            {
                "m": "B",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}], [{"m": "B", "v": 0}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["A", "B", "C"]],
            },
        ],
        "c": [["A", "B", "C"]],
    }


@pytest.fixture
def secured_causal_cover_fail_data():
    """Return a CCS instance with an invalid causally secured cover."""
    data = {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["B"]],
            },
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["B"]],
            },
        ],
        "c": [["B"]],
    }
    return data


@pytest.fixture
def secured_local_covers_fail_data():
    """Return a CCS instance with a dirty local cover."""
    data = {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "C"], ["B", "C"]],
            },
            {
                "m": "B",
                "e": [],
                "o": [{"v": 0, "l": False}, {"v": 1, "l": True}],
                "c": [["A", "C"], ["B", "C"]],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}, {"m": "B", "v": 0}]],
                "o": [{"v": 0, "l": True}, {"v": 1, "l": True}],
                "c": [["A", "C"], ["B", "C"]],
            },
        ],
        "c": [["A", "C"], ["B", "C"]],
    }
    return data


@pytest.fixture
def secured_cycle_fail_data():
    """Return a CCS instance with a causal cycle."""
    data = {
        "ms": [
            {
                "m": "A",
                "e": [[{"m": "B", "v": 0}]],
                "o": [{"v": 0, "l": True}],
                "c": [["A", "B"]],
            },
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0, "l": True}],
                "c": [["A", "B"]],
            },
        ],
        "c": [["A", "B"]],
    }
    return data


@pytest.fixture
def unstable_ccs_data(stable_ccs_data):
    """Return an unstable CCS fixture with overlapping bridges in one facet."""
    data = deepcopy(stable_ccs_data)
    data["c"] = [["A", "B", "C"]]
    return data


def test_ccs_init_repr_and_validate(valid_ccs_data):
    """Verify CCS initialization, representation, and schema validation."""
    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    assert ccs.measurements["A"]["m"] == "A"
    assert ccs.cover == {frozenset({"A", "B"})}

    with patch("jsonschema.validate") as mock_validate:
        mock_validate.return_value = None
        assert ccs.validate() is True

        mock_validate.side_effect = jsonschema.ValidationError("boom")
        assert ccs.validate() is False

    assert isinstance(repr(ccs), str)


def test_ccs_consistency_and_leaves(valid_ccs_data):
    """Verify enabling consistency and leaf handling."""
    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    assert ccs.check_consistency() is True

    ccs.add_leaves()
    assert ccs.check_leaves() is True
    assert ccs.data["ms"][0]["o"][0]["l"] is False
    assert ccs.data["ms"][0]["o"][1]["l"] is True


def test_ccs_consistency_duplicate_measurement_raises(valid_ccs_data):
    """Reject duplicate measurements inside one enabling relation."""
    data = deepcopy(valid_ccs_data)
    data["ms"][1]["e"] = [[{"m": "A", "v": 0}, {"m": "A", "v": 1}]]
    ccs = CausalContextualityScenario(data)

    with pytest.raises(ValueError, match="Duplicate measurement"):
        ccs.check_consistency()


def test_ccs_leaf_validation_failure_raises(valid_ccs_data):
    """Reject incorrect leaf annotations."""
    data = deepcopy(valid_ccs_data)
    data["ms"][0]["o"][0]["l"] = True
    ccs = CausalContextualityScenario(data)

    with pytest.raises(ValueError, match="Leaves are not valid"):
        ccs.check_leaves()


def test_ccs_missing_leaves_added(valid_ccs_data):
    """Make sure that leaves are added when missing."""
    data = deepcopy(valid_ccs_data)
    for measurement in data["ms"]:
        for outcome in measurement["o"]:
            del outcome["l"]
    ccs = CausalContextualityScenario(data)
    assert ccs._handle_leaves(check=True) is False
    assert ccs._handle_leaves(check=False) is True
    assert ccs.check_leaves() is True


def test_ccs_memberships_and_context_checks(valid_ccs_data):
    """Verify context membership calculations and checks."""
    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    memberships = ccs.calculate_memberships()
    assert ["A", "B"] in memberships["A"]
    assert ["A", "B"] in memberships["B"]

    del ccs.data["ms"][0]["c"]
    ccs.add_memberships()
    assert ccs.data["ms"][0]["c"] == [["A", "B"]]

    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    assert ccs.check_contexts() is True
    assert ccs.check_cover() is True

    del ccs.data["ms"][0]["c"]
    assert ccs.check_contexts() is True


def test_ccs_context_checks_detect_bad_data(valid_ccs_data):
    """Reject invalid contexts and duplicate outcome values."""
    bad_context = deepcopy(valid_ccs_data)
    bad_context["ms"][0]["c"] = [["B", "C"]]
    assert CausalContextualityScenario(bad_context).check_contexts() is False

    bad_cover = deepcopy(valid_ccs_data)
    bad_cover["c"] = [["A", "B"], ["A"]]
    assert CausalContextualityScenario(bad_cover).check_anti_chain() is False

    bad_values = deepcopy(valid_ccs_data)
    bad_values["ms"][0]["o"].append({"v": 0})
    assert CausalContextualityScenario(bad_values).check_unique_values() is False


def test_ccs_cover_is_anti_chain(stable_ccs_data):
    """Test that a non-trivial cover is an anti-chain."""
    data = deepcopy(stable_ccs_data)
    ccs = CausalContextualityScenario(data)
    assert ccs.check_anti_chain() is True


def test_ccs_sort_data_and_human_readable(valid_ccs_data):
    """Sort CCS data and add a human-readable summary."""
    data = deepcopy(valid_ccs_data)
    data["ms"] = list(reversed(data["ms"]))
    data["ms"][0]["c"] = [["B", "A"]]
    ccs = CausalContextualityScenario(data)
    ccs.sort_data()
    assert [measurement["m"] for measurement in ccs.data["ms"]] == ["A", "B"]
    assert ccs.data["ms"][0]["c"] == [["A", "B"]]

    ccs.add_human_readable()
    assert set(ccs.data["h"]) == {"ms", "o", "e", "c"}
    assert "ms" in repr(ccs)


def test_ccs_everything_and_file_outputs(tmp_path, valid_ccs_data):
    """Run the full CCS pipeline and verify file output helpers."""
    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    assert ccs.everything() is True

    json_path = tmp_path / "scenario"
    ccs.to_json(json_path)
    assert (tmp_path / "scenario.json").exists()

    jsonl_path = tmp_path / "scenario_lines"
    ccs.append_to_json_lines(jsonl_path)
    assert (tmp_path / "scenario_lines.jsonl").exists()


def test_ccs_everything_rejects_invalid_schema(valid_ccs_data):
    """Reject CCS data that violates the schema."""
    data = deepcopy(valid_ccs_data)
    data["extra_key"] = True
    ccs = CausalContextualityScenario(data)

    with pytest.raises(jsonschema.ValidationError):
        ccs.everything()


def test_ccs_to_spacetime_game(valid_ccs_data):
    """Convert a CCS to an alternating spacetime game."""
    ccs = CausallySecuredScenario(deepcopy(valid_ccs_data))
    ccs.add_leaves()
    ccs.add_memberships()
    game_dict = ccs.to_spacetime_game()
    assert game_dict["ps"] == ["Bob", "Alfred"]
    # create game instance
    game = AlternatingSpacetimeGame(game_dict)
    assert game.validate()
    assert game.all_checks()


def test_ccs_with_cycles():
    """Detect causal cycles in a scenario."""
    cyclic = {
        "ms": [
            {"m": "A", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["A", "B"]],
    }
    assert CausallySecuredScenario(cyclic).check_no_cycles() is False


def test_ccs_topological_order_with_cycle():
    """Detect causal cycles in a scenario using topological order."""
    cyclic = {
        "ms": [
            {"m": "A", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["A", "B"]],
    }
    with pytest.raises(ValueError, match="The enabling relations contain a cycle"):
        CausallySecuredScenario(cyclic)._topological_order()


def test_stable_ccs_stability_and_deduplication(stable_ccs_data):
    """Verify stability checks, bridge duplication, and convert the deduped scenario to a game."""
    ccs = StableCausalContextualityScenario(deepcopy(stable_ccs_data))
    assert ccs.check_stability() is True

    deduped = ccs.deduplicate_causal_bridges()
    assert isinstance(deduped, StableCausalContextualityScenario)
    assert len(deduped.data["ms"]) > len(ccs.data["ms"])
    new_ccs = CausallySecuredScenario(deduped.data)
    assert new_ccs.check_unique_causal_bridges() is True
    game = AlternatingSpacetimeGame(new_ccs.to_spacetime_game())
    assert game.all_checks() is True


def test_stable_ccs_detects_unstable_facet(unstable_ccs_data):
    """Reject a facet that can realize two bridges at once."""
    ccs = StableCausalContextualityScenario(deepcopy(unstable_ccs_data))
    with pytest.raises(ValueError, match="unstable"):
        ccs.check_stability()

    with pytest.raises(ValueError, match="Could not lift facet"):
        ccs.deduplicate_causal_bridges()


def test_causally_secured_ccs_checks_and_conversion(valid_ccs_data):
    """Verify causally secured checks and the game conversion path."""
    ccs = CausallySecuredScenario(deepcopy(valid_ccs_data))
    assert ccs.check_unique_causal_bridges() is True
    assert ccs.check_causally_secured_cover() is True
    assert ccs.check_local_covers_clean() is True
    assert ccs.check_no_cycles() is True

    game_dict = ccs.to_spacetime_game()
    game = AlternatingSpacetimeGame(game_dict)
    assert game.check_2_players() is True


def test_base_checks_pass(valid_ccs_data):
    """Verify the base CCS checks on a minimal valid instance."""
    ccs = CausalContextualityScenario(deepcopy(valid_ccs_data))
    assert ccs.check_consistency() is True
    assert ccs.check_contexts() is True
    assert ccs.check_cover() is True
    assert ccs.check_leaves() is True
    assert ccs.check_unique_values() is True
    assert ccs.check_anti_chain() is True


def test_check_consistency_rejects_duplicate_measurement(ccs_consistency_fail_data):
    """Reject an enabling relation that repeats one measurement."""
    ccs = CausalContextualityScenario(deepcopy(ccs_consistency_fail_data))
    with pytest.raises(ValueError, match="Duplicate measurement"):
        ccs.check_consistency()


def test_check_contexts_rejects_inconsistent_memberships(ccs_contexts_fail_data):
    """Reject a measurement whose local contexts do not match the cover."""
    ccs = CausalContextualityScenario(deepcopy(ccs_contexts_fail_data))
    assert ccs.check_contexts() is False


def test_check_cover_rejects_incomplete_cover(ccs_cover_fail_data):
    """Reject a cover that does not include every measurement."""
    ccs = CausalContextualityScenario(deepcopy(ccs_cover_fail_data))
    assert ccs.check_cover() is False


def test_check_leaves_rejects_incorrect_annotation(ccs_leaves_fail_data):
    """Reject an outcome whose leaf flag is incorrect."""
    ccs = CausalContextualityScenario(deepcopy(ccs_leaves_fail_data))
    with pytest.raises(ValueError, match="Leaves are not valid"):
        ccs.check_leaves()


def test_check_unique_values_rejects_duplicate_outcomes(ccs_unique_values_fail_data):
    """Reject repeated outcome values within one measurement."""
    ccs = CausalContextualityScenario(deepcopy(ccs_unique_values_fail_data))
    assert ccs.check_unique_values() is False


def test_check_anti_chain_rejects_nested_cover(ccs_antichain_fail_data):
    """Reject a cover that contains a strict subset relation."""
    ccs = CausalContextualityScenario(deepcopy(ccs_antichain_fail_data))
    assert ccs.check_anti_chain() is False


def test_check_enabling_events_no_enabling_relations():
    """Ensure flat scenarios validate."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B"]],
    }
    ccs = CausalContextualityScenario(data)
    assert ccs.check_enabling_events() is True


def test_check_enabling_events_perfectly_consistent():
    """Test conistent enabling events and measurement outcomes."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B"]],
    }
    ccs = CausalContextualityScenario(data)
    assert ccs.check_enabling_events() is True


def test_check_enabling_events_multiple_events_in_single_relation():
    """Exercise relations that contain multiple compound event constraints."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}, {"m": "B", "v": 1}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C"]],
    }
    ccs = CausalContextualityScenario(data)
    assert ccs.check_enabling_events() is True


def test_check_enabling_events_inconsistent_value():
    """Check that inconsistent value in relation fails."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "B",
                "e": [[{"m": "A", "v": 99}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B"]],
    }
    ccs = CausalContextualityScenario(data)
    assert ccs.check_enabling_events() is False


def test_check_enabling_events_missing_measurement_reference():
    """Force invalid measurement to fail."""
    data = {
        "ms": [{"m": "B", "e": [[{"m": "NonExistentMeasurement", "v": 0}]], "o": [{"v": 0}]}],
        "c": [["B"]],
    }
    ccs = CausalContextualityScenario(data)

    with pytest.raises(KeyError):
        ccs.check_enabling_events()


def test_all_checks_pass(secured_ccs_data):
    """Verify that the full secured CCS pipeline passes."""
    ccs = CausallySecuredScenario(deepcopy(secured_ccs_data))
    assert ccs.all_checks() is True


def test_all_checks_rejects_nested_cover(ccs_all_checks_fail_data):
    """Reject a CCS that fails all_checks through the anti-chain test."""
    ccs = CausalContextualityScenario(deepcopy(ccs_all_checks_fail_data))
    with pytest.raises(ValueError, match="Inconsistency detected: check_anti_chain failed"):
        ccs.all_checks()


def test_stable_checks_pass(stable_ccs_data):
    """Verify that the stable CCS check accepts compatible bridges."""
    ccs = StableCausalContextualityScenario(deepcopy(stable_ccs_data))
    assert ccs.check_stability() is True


def test_stable_check_rejects_shared_facet(stable_ccs_fail_data):
    """Reject a measurement whose bridges can coexist in one facet."""
    ccs = StableCausalContextualityScenario(deepcopy(stable_ccs_fail_data))
    with pytest.raises(ValueError, match="unstable and cannot be duplicated"):
        ccs.check_stability()


def test_secured_checks_pass(secured_ccs_data):
    """Verify the causally secured checks on a minimal valid instance."""
    ccs = CausallySecuredScenario(deepcopy(secured_ccs_data))
    assert ccs.check_unique_causal_bridges() is True
    assert ccs.check_causally_secured_cover() is True
    assert ccs.check_local_covers_clean() is True
    assert ccs.check_no_cycles() is True


def test_check_unique_causal_bridges_rejects_non_contradictory_bridges(
    secured_unique_bridges_fail_data,
):
    """Reject two bridges that can be active at the same time."""
    ccs = CausallySecuredScenario(deepcopy(secured_unique_bridges_fail_data))
    assert ccs.check_unique_causal_bridges() is False


def test_check_causally_secured_cover_rejects_missing_support(
    secured_causal_cover_fail_data,
):
    """Reject a cover that omits the support of an enabled measurement."""
    ccs = CausallySecuredScenario(deepcopy(secured_causal_cover_fail_data))
    assert ccs.check_causally_secured_cover() is False


def test_check_local_covers_clean_rejects_split_support(
    secured_local_covers_fail_data,
):
    """Reject a cover that splits a bridge's support across local contexts."""
    ccs = CausallySecuredScenario(deepcopy(secured_local_covers_fail_data))
    assert ccs.check_local_covers_clean() is False


def test_check_no_cycles_rejects_cycle(secured_cycle_fail_data):
    """Reject a causal graph with a directed cycle."""
    ccs = CausallySecuredScenario(deepcopy(secured_cycle_fail_data))
    assert ccs.check_no_cycles() is False


def test_stability_with_two_duplications(two_duplications_ccs_data):
    """Check that two measurements are duplicated when deduplicating causal bridges."""
    ccs = StableCausalContextualityScenario(deepcopy(two_duplications_ccs_data))
    deduped = ccs.deduplicate_causal_bridges()
    measurement_splits = defaultdict(int)
    for measurement, measurement_data in deduped.measurements.items():
        assert len(measurement_data["e"]) <= 1
        measurement_splits[measurement[0]] += 1
    expected = set([("A", 1), ("B", 1), ("C", 2), ("D", 2)])
    got = set(item for item in measurement_splits.items())
    assert got == expected
    assert len(deduped.measurements) == len(ccs.measurements) + 2
    assert set(measurement for context in deduped.cover for measurement in context) == set(
        deduped.measurements
    )
    assert deduped.everything()


def test_stability_one_duplication(
    one_duplication_based_on_enabling_relations, one_duplication_based_on_cover
):
    """Test only one duplicate measurement is created when enabling relations disallow more."""
    data_array = [one_duplication_based_on_enabling_relations, one_duplication_based_on_cover]
    for data in data_array:
        ccs = StableCausalContextualityScenario(deepcopy(data))
        deduped = ccs.deduplicate_causal_bridges()
        measurement_splits = defaultdict(int)
        for measurement, measurement_data in deduped.measurements.items():
            assert len(measurement_data["e"]) <= 1
            measurement_splits[measurement[0]] += 1
        expected = set([("A", 1), ("B", 1), ("C", 2), ("D", 1)])
        got = set(item for item in measurement_splits.items())
        assert got == expected
        assert len(deduped.measurements) == len(ccs.measurements) + 1
        assert set(measurement for context in deduped.cover for measurement in context) == set(
            deduped.measurements
        )
        assert deduped.everything()


def test_multiple_duplications():
    """Duplicate multiple times and check consistency."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "D", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "E",
                "e": [[{"m": "A", "v": 0}], [{"m": "B", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "F",
                "e": [[{"m": "C", "v": 0}], [{"m": "D", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "G",
                "e": [[{"m": "E", "v": 0}], [{"m": "F", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "E", "G"], ["B", "E", "G"], ["C", "F", "G"], ["D", "F", "G"]],
    }
    ccs = StableCausalContextualityScenario(data)
    deduped = ccs.deduplicate_causal_bridges()
    measurement_splits = defaultdict(int)
    for measurement, measurement_data in deduped.measurements.items():
        assert len(measurement_data["e"]) <= 1
        measurement_splits[measurement[0]] += 1
    expected = set([("A", 1), ("B", 1), ("C", 1), ("D", 1), ("E", 2), ("F", 2), ("G", 4)])
    got = set(item for item in measurement_splits.items())
    assert got == expected
    assert len(deduped.measurements) == len(ccs.measurements) + 5
    assert set(measurement for context in deduped.cover for measurement in context) == set(
        deduped.measurements
    )
    for c1 in deduped.cover:
        for c2 in deduped.cover:
            if c1 == c2:
                continue
            assert not c1 & c2
    assert deduped.everything()


def test_get_transitive_enabling_continues_on_seen_nodes():
    """Visit the same node twice in the transitive closure queue."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {"m": "C", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {
                "m": "D",
                "e": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C", "D"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    closure = ccs._get_transitive_enabling("D")
    assert closure == {"B": 0, "C": 0, "A": 0}


def test_get_transitive_enabling_returns_none_on_conflict():
    """Reject inconsistent transitive enabling requirements."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {"m": "C", "e": [[{"m": "A", "v": 1}]], "o": [{"v": 0}]},
            {
                "m": "D",
                "e": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C", "D"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    assert ccs._get_transitive_enabling("D") is None


def test_check_causally_secured_cover_returns_false_for_inconsistent_closure():
    """Reject a cover when a transitive closure is internally inconsistent."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {"m": "C", "e": [[{"m": "A", "v": 1}]], "o": [{"v": 0}]},
            {
                "m": "D",
                "e": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C", "D"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    assert ccs.check_causally_secured_cover() is False


def test_check_causally_secured_cover_returns_false_for_nonpropagating_facet():
    """Reject a facet that does not contain its enabling support."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["B"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    assert ccs.check_causally_secured_cover() is False


def test_check_causally_secured_cover_returns_false_for_conflicting_histories():
    """Reject a facet containing two measurements with incompatible transitive closures."""
    data = {
        "ms": [
            {"m": "R", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "X", "e": [[{"m": "R", "v": 0}]], "o": [{"v": 0}]},
            {"m": "Y", "e": [[{"m": "R", "v": 1}]], "o": [{"v": 0}]},
        ],
        "c": [["R", "X", "Y"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    assert ccs.check_causally_secured_cover() is False


def test_check_unique_causal_bridges_returns_false_on_empty_relation():
    """Reject an empty enabling relation paired with another relation."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {
                "m": "B",
                "e": [[], [{"m": "A", "v": 0}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))
    assert ccs.check_unique_causal_bridges() is False


def test_check_unique_causal_bridges_hits_contradiction_branch():
    """Detect a contradiction between two enabling relations."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [], "o": [{"v": 0}]},
            {
                "m": "B",
                "e": [
                    [{"m": "A", "v": 0}, {"m": "C", "v": 0}],
                    [{"m": "A", "v": 1}, {"m": "C", "v": 0}],
                ],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))
    assert ccs.check_unique_causal_bridges() is True


def test_value_label_and_context_label_branches():
    """Cover the special label formatting branches."""
    assert CausallySecuredScenario._value_label({"v": 7}) == "7"
    assert CausallySecuredScenario._value_label("plain_text") == "plain_text"
    assert CausallySecuredScenario._context_label(frozenset()) == "{}"


def test_to_spacetime_game_rejects_multiple_enabling_relations():
    """Reject a measurement with more than one enabling relation."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    with pytest.raises(ValueError, match="Multiple enabling relations"):
        ccs.to_spacetime_game()


def test_to_spacetime_game_rejects_empty_local_cover():
    """Reject a scenario whose local cover restriction becomes empty."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["A"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    with pytest.raises(ValueError, match="Empty local cover"):
        ccs.to_spacetime_game()


def test_to_spacetime_game_hits_try_create_bob_nodes_break_and_continue():
    """Exercise the candidate-missing and skip-existing-bridge branches."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {"m": "C", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
            {
                "m": "D",
                "e": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
                "o": [{"v": 0}],
            },
        ],
        "c": [["A", "B", "C", "D"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))
    game = ccs.to_spacetime_game()

    assert game["ps"] == ["Bob", "Alfred"]
    assert any(iset["p"] == "Bob" for iset in game["is"])
    assert any(iset["p"] == "Alfred" for iset in game["is"])


def test_to_spacetime_game_rejects_missing_root_local_cover():
    """Reject a scenario with no root local cover."""
    data = {
        "ms": [
            {"m": "A", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["A", "B"]],
    }
    ccs = CausallySecuredScenario(deepcopy(data))

    with pytest.raises(ValueError, match="No root local cover"):
        ccs.to_spacetime_game()


@pytest.fixture
def clean_scenario_data():
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "A", "v": 1}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B"], ["A", "C"]],
    }


@pytest.fixture
def dirty_inconsistent_only_data():
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}, {"m": "A", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B"]],
    }


@pytest.fixture
def dirty_facet_overlap_data():
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "B", "v": 1}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        # C's closure is {A, B} but no facet contains both
        "c": [["A", "B"], ["A", "C"]],
    }


@pytest.fixture
def clean_closures_ccs_data():
    """Build a CCS where all cleanliness checks pass."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "A", "v": 1}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B"], ["A", "C"]],
    }


@pytest.fixture
def dirty_unreachable_relation_data():
    """Build a CCS with one unreachable enabling relation but reachable measurements."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "C",
                "e": [
                    [{"m": "A", "v": 0}],
                    [{"m": "B", "v": 0}],
                    [{"m": "A", "v": 0}, {"m": "B", "v": 0}],
                ],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "C"], ["B", "C"]],
    }


@pytest.fixture
def dirty_facet_unreachable_measurement_data():
    """Build a CCS with a facet that cannot realize all of its measurements."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "D",
                "e": [[{"m": "C", "v": 1}], [{"m": "A", "v": 1}, {"m": "B", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B", "D"], ["A", "C", "D"], ["B", "C", "D"]],
    }


@pytest.fixture
def dirty_not_transitively_closed_data():
    """Build a CCS whose enabling relations are not transitively closed."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B", "C"]],
    }


@pytest.fixture
def helper_conflict_data():
    """Build a CCS whose relation closure becomes inconsistent."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "A", "v": 1}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "D", "e": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B", "C", "D"]],
    }


def test_relation_transitive_closures_returns_empty_for_root_measurement(clean_closures_ccs_data):
    """Return the empty closure for a root relation."""
    ccs = CausalContextualityScenario(deepcopy(clean_closures_ccs_data))
    closures_by_measurement = utils.compute_transitive_closures(ccs.data, consistent=True)

    assert ccs._relation_transitive_closures([], closures_by_measurement) == {frozenset()}


def test_relation_transitive_closures_returns_empty_on_conflict(helper_conflict_data):
    """Drop an impossible relation closure."""
    ccs = CausalContextualityScenario(deepcopy(helper_conflict_data))
    closures_by_measurement = utils.compute_transitive_closures(ccs.data, consistent=True)

    relation = [{"m": "B", "v": 0}, {"m": "C", "v": 0}]
    assert ccs._relation_transitive_closures(relation, closures_by_measurement) == set()


def test_check_all_enabling_relations_reachable_passes_on_clean_data(clean_closures_ccs_data):
    """Accept a scenario whose every enabling relation is reachable."""
    ccs = CausalContextualityScenario(deepcopy(clean_closures_ccs_data))
    assert ccs._check_all_enabling_relations_reachable() is True


def test_check_all_enabling_relations_reachable_fails_on_unreachable_relation(
    dirty_unreachable_relation_data,
):
    """Reject a scenario with an unreachable enabling relation."""
    ccs = CausalContextualityScenario(deepcopy(dirty_unreachable_relation_data))
    assert ccs._check_all_enabling_relations_reachable() is False


def test_check_facet_measurements_reachable_passes_on_clean_data(clean_closures_ccs_data):
    """Accept a scenario whose every facet can realize all of its measurements."""
    ccs = CausalContextualityScenario(deepcopy(clean_closures_ccs_data))
    assert ccs._check_facet_measurements_reachable() is True


def test_check_facet_measurements_reachable_fails_on_dirty_facet(
    dirty_facet_unreachable_measurement_data,
):
    """Reject a facet that cannot realize all of its measurements."""
    ccs = CausalContextualityScenario(deepcopy(dirty_facet_unreachable_measurement_data))
    assert ccs._check_facet_measurements_reachable() is False


def test_check_transitively_closed_enabling_relations_passes_on_closed_data(
    clean_closures_ccs_data,
):
    """Accept already transitively closed enabling relations."""
    ccs = CausalContextualityScenario(deepcopy(clean_closures_ccs_data))
    assert ccs._check_transitively_closed_enabling_relations() is True


def test_check_transitively_closed_enabling_relations_fails_on_open_data(
    dirty_not_transitively_closed_data,
):
    """Reject enabling relations that are not transitively closed."""
    ccs = CausalContextualityScenario(deepcopy(dirty_not_transitively_closed_data))
    assert ccs._check_transitively_closed_enabling_relations() is False


def test_transitively_close_enabling_relations_closes_in_place(dirty_not_transitively_closed_data):
    """Close enabling relations transitively in place."""
    ccs = CausalContextualityScenario(deepcopy(dirty_not_transitively_closed_data))
    returned = ccs.transitively_close_enabling_relations()

    assert returned is ccs
    assert ccs._check_transitively_closed_enabling_relations() is True
    assert ccs.data["ms"][0]["e"] == []
    assert ccs.data["ms"][1]["e"] == [[{"m": "A", "v": 0}]]
    assert ccs.data["ms"][2]["e"] == [[{"m": "A", "v": 0}, {"m": "B", "v": 0}]]


def test_transitively_close_enabling_relations_handles_multiple_branches():
    """Close multiple enabling branches consistently."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
            {"m": "C", "e": [[{"m": "B", "v": 1}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B", "C"]],
    }
    ccs = CausalContextualityScenario(deepcopy(data))

    ccs.transitively_close_enabling_relations()

    assert ccs.data["ms"][1]["e"] == [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]]
    assert ccs.data["ms"][2]["e"] == [
        [{"m": "A", "v": 0}, {"m": "B", "v": 1}],
        [{"m": "A", "v": 1}, {"m": "B", "v": 1}],
    ]


def test_is_scenario_clean_passes_on_clean_data(clean_closures_ccs_data):
    """Accept a scenario that satisfies all cleanliness criteria."""
    ccs = CausalContextualityScenario(deepcopy(clean_closures_ccs_data))
    assert ccs.is_scenario_clean() is True


def test_is_scenario_clean_fails_on_unreachable_relation(dirty_unreachable_relation_data):
    """Reject a scenario with an unreachable relation."""
    ccs = CausalContextualityScenario(deepcopy(dirty_unreachable_relation_data))
    assert ccs.is_scenario_clean() is False


def test_is_scenario_clean_fails_on_dirty_facet(dirty_facet_unreachable_measurement_data):
    """Reject a scenario with an unreachable measurement inside a facet."""
    ccs = CausalContextualityScenario(deepcopy(dirty_facet_unreachable_measurement_data))
    assert ccs.is_scenario_clean() is False


def test_is_scenario_clean_fails_on_open_enabling_relations(dirty_not_transitively_closed_data):
    """Reject a scenario whose enabling relations are not transitively closed."""
    ccs = CausalContextualityScenario(deepcopy(dirty_not_transitively_closed_data))
    assert ccs.is_scenario_clean() is False


@pytest.fixture
def equivalent_ccs_data_by_a_root():
    """Build a CCS whose child is enabled by the first root measurement."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["A", "B"]],
            },
            {
                "m": "B",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["A", "C"]],
            },
        ],
        "c": [["A", "B"], ["A", "C"]],
    }


@pytest.fixture
def equivalent_ccs_data_by_b_root():
    """Build a CCS equivalent to the A-root scenario up to relabeling."""
    return {
        "ms": [
            {
                "m": "X",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["X", "Y"]],
            },
            {
                "m": "Y",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "Z",
                "e": [[{"m": "X", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["X", "Z"]],
            },
        ],
        "c": [["X", "Y"], ["X", "Z"]],
    }


@pytest.fixture
def non_equivalent_two_bridge_data():
    """Build a CCS whose child has two distinct enabling relations."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}], [{"m": "B", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B", "C"]],
    }


@pytest.fixture
def dirty_not_transitively_closed_canonical_data():
    """Build a CCS whose enabling relations need transitive closure."""
    return {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            {"m": "C", "e": [[{"m": "B", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B", "C"]],
        "h": {"ms": "stale", "o": "stale", "e": "stale", "c": "stale"},
    }


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, "A"),
        (1, "B"),
        (25, "Z"),
        (26, "AA"),
        (27, "AB"),
    ],
)
def test_excel_name_covers_single_and_multi_letter_indices(index, expected):
    """Convert zero-based indices into canonical measurement names."""
    assert CausalContextualityScenario._excel_name(index) == expected


def test_excel_name_rejects_negative_indices():
    """Reject negative indices when creating canonical measurement names."""
    with pytest.raises(ValueError, match="index must be non-negative"):
        CausalContextualityScenario._excel_name(-1)


def test_canonical_signature_sorts_relations_within_measurement():
    """Serialize one measurement's enabling relations in a deterministic order."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {
                "m": "C",
                "e": [[{"m": "B", "v": 0}], [{"m": "A", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B", "C"]],
    }
    ccs = CausalContextualityScenario(deepcopy(data))

    signature = ccs._canonical_signature_for_order(["A", "B", "C"])

    assert signature == (
        tuple(),
        tuple(),
        ((("A", 1),), (("B", 0),)),
    )


def test_apply_measurement_renaming_relabels_measurements_cover_and_local_contexts():
    """Rewrite a scenario in place under a canonical measurement renaming."""
    data = {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["A", "B"]],
            },
            {
                "m": "B",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["A", "C"]],
            },
        ],
        "c": [["A", "B"], ["A", "C"]],
    }
    ccs = CausalContextualityScenario(deepcopy(data))

    ccs._apply_measurement_renaming(["B", "A", "C"])

    assert [measurement["m"] for measurement in ccs.data["ms"]] == ["A", "B", "C"]
    assert ccs.data["ms"][0]["e"] == []
    assert ccs.data["ms"][1]["c"] == [["A", "B"]]
    assert ccs.data["ms"][2]["e"] == [[{"m": "B", "v": 0}]]
    assert ccs.data["ms"][2]["c"] == [["B", "C"]]
    assert ccs.data["c"] == [["A", "B"], ["B", "C"]]
    assert ccs.cover == {frozenset({"A", "B"}), frozenset({"B", "C"})}
    assert ccs.measurements["B"]["m"] == "B"


def test_canonicalize_enabling_equivalence_maps_equivalent_structures_to_same_form(
    equivalent_ccs_data_by_a_root,
    equivalent_ccs_data_by_b_root,
    monkeypatch,
):
    """Collapse two relabeled but physically equivalent scenarios to the same canonical form."""

    def fake_topological_orders(nodes, parents):
        """Yield the two legal linearizations for a two-root one-child graph."""
        yield ("A", "B", "C") if "A" in nodes else ("X", "Y", "Z")
        yield ("B", "A", "C") if "A" in nodes else ("Y", "X", "Z")

    monkeypatch.setattr(utils, "_topological_orders", fake_topological_orders)

    ccs_a = CausalContextualityScenario(deepcopy(equivalent_ccs_data_by_a_root))
    ccs_b = CausalContextualityScenario(deepcopy(equivalent_ccs_data_by_b_root))

    returned_a = ccs_a.canonicalize_enabling_equivalence()
    returned_b = ccs_b.canonicalize_enabling_equivalence()

    assert returned_a is ccs_a
    assert returned_b is ccs_b
    assert ccs_a.data == ccs_b.data
    assert [measurement["m"] for measurement in ccs_a.data["ms"]] == ["A", "B", "C"]
    assert ccs_a.data["ms"][2]["e"] == [[{"m": "A", "v": 0}]]


def test_canonicalize_enabling_equivalence_distinguishes_non_equivalent_structures(
    equivalent_ccs_data_by_a_root,
    non_equivalent_two_bridge_data,
    monkeypatch,
):
    """Keep scenarios with different enabling structure in different equivalence classes."""

    def fake_topological_orders(nodes, parents):
        """Yield the two legal linearizations for a two-root one-child graph."""
        yield ("A", "B", "C")
        yield ("B", "A", "C")

    monkeypatch.setattr(utils, "_topological_orders", fake_topological_orders)

    ccs_equiv = CausalContextualityScenario(deepcopy(equivalent_ccs_data_by_a_root))
    ccs_non_equiv = CausalContextualityScenario(deepcopy(non_equivalent_two_bridge_data))

    ccs_equiv.canonicalize_enabling_equivalence()
    ccs_non_equiv.canonicalize_enabling_equivalence()

    assert ccs_equiv.data != ccs_non_equiv.data
    assert len(ccs_equiv.data["ms"][2]["e"]) == 1
    assert len(ccs_non_equiv.data["ms"][2]["e"]) == 2


def test_canonicalize_enabling_equivalence_closes_transitively_and_refreshes_human_readable(
    dirty_not_transitively_closed_canonical_data,
    monkeypatch,
):
    """Close the scenario transitively and recompute the human-readable representation."""

    def fake_topological_orders(nodes, parents):
        """Yield a single linearization for the closed chain."""
        yield tuple(sorted(nodes))

    monkeypatch.setattr(utils, "_topological_orders", fake_topological_orders)

    ccs = CausalContextualityScenario(deepcopy(dirty_not_transitively_closed_canonical_data))
    assert ccs._check_transitively_closed_enabling_relations() is False

    ccs.canonicalize_enabling_equivalence(ensure_transitively_closed=True)

    assert ccs._check_transitively_closed_enabling_relations() is True
    assert ccs.data["h"]["ms"] == "{A, B, C}"
    assert "(A,0)" in ccs.data["h"]["e"]
    assert "(B,0)" in ccs.data["h"]["e"]


def test_canonicalize_enabling_equivalence_skips_closure_when_not_requested(
    equivalent_ccs_data_by_a_root,
    monkeypatch,
):
    """Leave an already closed scenario untouched when closure is disabled."""

    def fake_topological_orders(nodes, parents):
        """Yield the only legal linearization for the test scenario."""
        yield ("A", "B", "C")

    monkeypatch.setattr(utils, "_topological_orders", fake_topological_orders)

    ccs = CausalContextualityScenario(deepcopy(equivalent_ccs_data_by_a_root))
    monkeypatch.setattr(
        ccs,
        "transitively_close_enabling_relations",
        lambda: pytest.fail("closure should not be called"),
    )

    ccs.canonicalize_enabling_equivalence(ensure_transitively_closed=False)

    assert ccs.data["ms"][2]["e"] == [[{"m": "A", "v": 0}]]


def test_canonicalize_enabling_equivalence_raises_when_no_legal_orders(
    equivalent_ccs_data_by_a_root,
    monkeypatch,
):
    """Reject scenarios that do not admit any legal measurement order."""
    monkeypatch.setattr(utils, "_topological_orders", lambda _nodes, _parents: iter(()))

    ccs = CausalContextualityScenario(deepcopy(equivalent_ccs_data_by_a_root))

    with pytest.raises(ValueError, match="No valid topological order exists for this scenario"):
        ccs.canonicalize_enabling_equivalence()


def _empty_ccs_data():
    """Build a minimal CCS instance with one flat measurement."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            }
        ],
        "c": [["A"]],
    }


def _branching_ccs_data():
    """Build a CCS instance with a genuine branching dependency graph."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "B",
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "C",
                "e": [[{"m": "A", "v": 1}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B", "C"]],
    }


def _self_reference_ccs_data():
    """Build a CCS instance whose enabling graph contains a self-reference."""
    return {
        "ms": [
            {
                "m": "A",
                "e": [],
                "o": [{"v": 0}, {"v": 1}],
            },
            {
                "m": "B",
                "e": [[{"m": "B", "v": 0}, {"m": "A", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
            },
        ],
        "c": [["A", "B"]],
        "h": {
            "ms": "{A, B}",
            "o": "O_A = {0, 1}, O_B = {0, 1}",
            "e": "∅ ⊢ A, {(B,0),(A,0)} ⊢ B",
            "c": "{{A, B}}",
        },
    }


def test_relation_transitive_closures_skips_conflicting_parent_value():
    """Drop partial closures that would assign incompatible values to a parent."""
    ccs = CausalContextualityScenario(_empty_ccs_data())
    relation = [{"m": "A", "v": 0}, {"m": "A", "v": 1}]
    closures_by_measurement = {"A": {frozenset()}}

    result = ccs._relation_transitive_closures(relation, closures_by_measurement)
    assert result == set()


def test_relation_transitive_closures_builds_expected_closure():
    """Combine compatible parent closures into a closure set."""
    ccs = CausalContextualityScenario(_branching_ccs_data())
    relation = [{"m": "A", "v": 0}, {"m": "B", "v": 1}]
    closures_by_measurement = {
        "A": {frozenset()},
        "B": {frozenset({("A", 0)})},
    }

    result = ccs._relation_transitive_closures(relation, closures_by_measurement)
    assert result == {frozenset({("A", 0), ("B", 1)})}


def test_check_all_enabling_relations_reachable_returns_true(monkeypatch):
    """Accept a relation whose support fits inside a facet."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A", "B"]],
    }
    ccs = CausalContextualityScenario(data)

    monkeypatch.setattr(
        utils,
        "compute_transitive_closures",
        lambda data, consistent=True: {
            "A": {frozenset()},
            "B": {frozenset({("A", 0)})},
        },
    )

    assert ccs._check_all_enabling_relations_reachable() is True


def test_check_all_enabling_relations_reachable_returns_false(monkeypatch):
    """Reject a relation whose support does not fit inside any facet."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
        ],
        "c": [["A"]],
    }
    ccs = CausalContextualityScenario(data)

    monkeypatch.setattr(
        utils,
        "compute_transitive_closures",
        lambda data, consistent=True: {
            "A": {frozenset()},
            "B": {frozenset({("A", 0)})},
        },
    )

    assert ccs._check_all_enabling_relations_reachable() is False


def test_canonicalize_enabling_equivalence_ignores_self_reference():
    """Canonicalize a scenario whose dependency graph contains a self-reference."""
    ccs = CausalContextualityScenario(deepcopy(_self_reference_ccs_data()))

    ccs.canonicalize_enabling_equivalence(ensure_transitively_closed=False)

    assert [measurement["m"] for measurement in ccs.data["ms"]] == ["A", "B"]
    assert ccs.data["c"] == [["A", "B"]]
    assert ccs.data["ms"][1]["e"] == [[{"m": "A", "v": 0}, {"m": "B", "v": 0}]]


def test_canonicalize_enabling_equivalence_recomputes_human_readable_when_closed(monkeypatch):
    """Close the scenario first and rebuild the human-readable representation."""
    data = _empty_ccs_data()
    data["h"] = {
        "ms": "{A}",
        "o": "O_A = {0, 1}",
        "e": "∅ ⊢ A",
        "c": "{{A}}",
    }
    ccs = CausalContextualityScenario(deepcopy(data))

    calls = {"closed": 0, "human": 0}

    def fake_closed(self):
        """Count transitive-closure calls."""
        calls["closed"] += 1
        return self

    def fake_human(self):
        """Count human-readable rebuilds."""
        calls["human"] += 1
        return None

    monkeypatch.setattr(
        CausalContextualityScenario,
        "_check_transitively_closed_enabling_relations",
        lambda self: False,
    )
    monkeypatch.setattr(
        CausalContextualityScenario,
        "transitively_close_enabling_relations",
        fake_closed,
    )
    monkeypatch.setattr(
        CausalContextualityScenario,
        "add_human_readable",
        fake_human,
    )

    ccs.canonicalize_enabling_equivalence(ensure_transitively_closed=True)

    assert calls["closed"] == 1
    assert calls["human"] == 1
    assert ccs.data["ms"][0]["m"] == "A"


def test_canonicalize_enabling_equivalence_skips_closure_when_already_closed(monkeypatch):
    """Leave already closed scenarios untouched before canonicalization."""
    ccs = CausalContextualityScenario(deepcopy(_empty_ccs_data()))

    calls = {"closed": 0}

    def fake_closed(self):
        """Count transitive-closure calls."""
        calls["closed"] += 1
        return self

    monkeypatch.setattr(
        CausalContextualityScenario,
        "_check_transitively_closed_enabling_relations",
        lambda self: True,
    )
    monkeypatch.setattr(
        CausalContextualityScenario,
        "transitively_close_enabling_relations",
        fake_closed,
    )

    ccs.canonicalize_enabling_equivalence(ensure_transitively_closed=True)

    assert calls["closed"] == 0
    assert [measurement["m"] for measurement in ccs.data["ms"]] == ["A"]
