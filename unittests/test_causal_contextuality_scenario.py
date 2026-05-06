"""Unit tests for the causal_contextuality_scenario module."""

from copy import deepcopy
from unittest.mock import patch

import jsonschema
import pytest

from quantum_experiment_structures.causal_contextuality_scenario import (
    CausalContextualityScenario,
    CausallySecuredScenario,
    StableCausalContextualityScenario,
)
from quantum_experiment_structures.spacetime_game import AlternatingSpacetimeGame


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


def test_deduplicate_causal_bridges_raises_when_parent_copy_list_empty():
    """Force deduplication to fail when a parent copy list is empty."""
    data = {
        "ms": [
            {"m": "A", "e": [], "o": [{"v": 0}]},
            {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}]},
        ],
        "c": [["A", "B"]],
    }
    ccs = StableCausalContextualityScenario(deepcopy(data))

    with patch.object(
        StableCausalContextualityScenario,
        "_topological_order",
        return_value=["B", "A"],
    ):
        with pytest.raises(ValueError, match="has not been expanded yet"):
            ccs.deduplicate_causal_bridges()


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
