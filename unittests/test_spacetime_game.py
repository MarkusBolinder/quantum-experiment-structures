"""Pytest coverage for the core causal contextuality modules.

The tests focus on three layers spacetime game validation and add/check helpers.
"""

from copy import deepcopy
from unittest.mock import patch

import jsonschema
import pytest

from quantum_experiment_structures.causal_contextuality_scenario import CausallySecuredScenario
from quantum_experiment_structures.spacetime_game import (
    AlternatingSpacetimeGame,
    SpacetimeGame,
)


def _assignment_set(history):
    """Return the assignment set of a history."""
    return frozenset(tuple(assignment.values()) for assignment in history["h"])


def _strategy_set(strategy):
    """Return the assignment set of a strategy."""
    return frozenset(tuple(assignment.values()) for assignment in strategy)


def _group_for_player(groups, player):
    """Return the strategy group for a player."""
    return next(group for group in groups if group["p"] == player)


@pytest.fixture
def base_game_data():
    """Provide a minimal spacelike-separated base game with three players."""

    return {
        "ps": ["Alice", "Bob", "Charlie"],
        "as": ["x", "y", "z"],
        "is": [
            {
                "i": "IA1",
                "p": "Alice",
                "a": ["x", "y"],
                "ns": [{"n": "NA1", "ps": []}],
            },
            {
                "i": "IA2",
                "p": "Alice",
                "a": ["x", "y"],
                "ns": [{"n": "NA2", "ps": []}],
            },
            {
                "i": "IB1",
                "p": "Bob",
                "a": ["z"],
                "ns": [{"n": "NB1", "ps": []}],
            },
            {
                "i": "IC1",
                "p": "Charlie",
                "a": ["z"],
                "ns": [{"n": "NC1", "ps": []}],
            },
        ],
    }


@pytest.fixture
def history_game_data():
    """Provide a minimal game with one enabled child information set."""

    return {
        "ps": ["Bob", "Alfred"],
        "as": ["L", "R", "a"],
        "is": [
            {
                "i": "IB",
                "p": "Bob",
                "a": ["L", "R"],
                "ns": [{"n": "NB", "ps": []}],
            },
            {
                "i": "IA",
                "p": "Alfred",
                "a": ["a"],
                "ns": [{"n": "NA", "ps": [{"p": "NB", "a": "L"}]}],
            },
        ],
    }


@pytest.fixture
def strategy_game_data():
    """Provide a base game that yields multiple full strategies for one player."""

    return {
        "ps": ["Alice", "Bob", "Charlie"],
        "as": ["x", "y", "z"],
        "is": [
            {
                "i": "IA1",
                "p": "Alice",
                "a": ["x", "y"],
                "ns": [{"n": "NA1", "ps": []}],
            },
            {
                "i": "IA2",
                "p": "Alice",
                "a": ["x", "y"],
                "ns": [{"n": "NA2", "ps": []}],
            },
            {
                "i": "IB1",
                "p": "Bob",
                "a": ["z"],
                "ns": [{"n": "NB1", "ps": []}],
            },
            {
                "i": "IC1",
                "p": "Charlie",
                "a": ["z"],
                "ns": [{"n": "NC1", "ps": []}],
            },
        ],
    }


@pytest.fixture
def valid_alternating_game_data():
    """Provide a minimal valid alternating game with a nontrivial reduced strategy split."""

    return {
        "ps": ["Bob", "Alfred"],
        "as": ["L", "R", "a", "b"],
        "is": [
            {
                "i": "IB0",
                "p": "Bob",
                "a": ["L", "R"],
                "ns": [{"n": "NB0", "ps": []}],
            },
            {
                "i": "IA0",
                "p": "Alfred",
                "a": ["a"],
                "ns": [{"n": "NA0", "ps": [{"p": "NB0", "a": "L"}]}],
            },
            {
                "i": "IA1",
                "p": "Alfred",
                "a": ["a"],
                "ns": [{"n": "NA1", "ps": [{"p": "NB0", "a": "R"}]}],
            },
            {
                "i": "IB1",
                "p": "Bob",
                "a": ["b"],
                "ns": [{"n": "NB1", "ps": [{"p": "NA0", "a": "a"}]}],
            },
            {
                "i": "IA2",
                "p": "Alfred",
                "a": ["b"],
                "ns": [{"n": "NA2", "ps": [{"p": "NB1", "a": "b"}]}],
            },
        ],
        "z": [
            {
                "z": "Z0",
                "h": [
                    {"i": "IB0", "a": "L"},
                    {"i": "IA0", "a": "a"},
                    {"i": "IB1", "a": "b"},
                    {"i": "IA2", "a": "b"},
                ],
                "u": [{"p": "Bob", "v": 0}, {"p": "Alfred", "v": 0}],
            }
        ],
    }


@pytest.fixture
def two_history_history_data():
    """Provide a valid history game for totality and co-totality tests."""

    return {
        "ps": ["Bob", "Alfred"],
        "as": ["L", "R", "a"],
        "is": [
            {
                "i": "IB",
                "p": "Bob",
                "a": ["L", "R"],
                "ns": [{"n": "NB", "ps": []}],
            },
            {
                "i": "IA",
                "p": "Alfred",
                "a": ["a"],
                "ns": [{"n": "NA", "ps": [{"p": "NB", "a": "L"}]}],
            },
        ],
        "z": [
            {
                "z": "Z0",
                "h": [{"i": "IB", "a": "L"}, {"i": "IA", "a": "a"}],
                "s": ["IB", "IA"],
                "u": [{"p": "Bob", "v": 1}, {"p": "Alfred", "v": 0}],
            }
        ],
    }


@pytest.fixture
def valid_spacelike_separated_game_data():
    """Provide a base game that explicitly contains two disconnected roots."""

    return {
        "ps": ["Alice", "Bob", "Charlie"],
        "as": ["x", "y", "z"],
        "is": [
            {
                "i": "IA",
                "p": "Alice",
                "a": ["x"],
                "ns": [{"n": "NA", "ps": []}],
            },
            {
                "i": "IB",
                "p": "Bob",
                "a": ["y"],
                "ns": [{"n": "NB", "ps": []}],
            },
            {
                "i": "IC",
                "p": "Charlie",
                "a": ["z"],
                "ns": [{"n": "NC", "ps": []}],
            },
        ],
    }


@pytest.fixture
def valid_ccs_data():
    """Return a small valid CCS fixture."""
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
                "e": [[{"m": "A", "v": 0}]],
                "o": [{"v": 0}, {"v": 1}],
                "c": [["A", "B"]],
            },
        ],
        "c": [["A", "B"]],
    }


@pytest.fixture
def valid_game_data():
    """Return a small valid spacetime game fixture."""
    return {
        "ps": ["Alice", "Bob"],
        "as": ["play", "pass"],
        "is": [
            {
                "i": "I1",
                "p": "Alice",
                "a": ["play", "pass"],
                "ns": [{"n": "N1", "ps": []}],
            },
            {
                "i": "I2",
                "p": "Bob",
                "a": ["play", "pass"],
                "ns": [{"n": "N2", "ps": [{"p": "N1", "a": "play"}]}],
            },
        ],
        "z": [
            {
                "z": "H1",
                "h": [{"i": "I1", "a": "play"}, {"i": "I2", "a": "pass"}],
                "s": ["I1", "I2"],
                "u": [{"p": "Alice", "v": 1}, {"p": "Bob", "v": 0}],
            }
        ],
    }


def _measurement(axis_index, n_outcomes):
    """Build one PMF measurement entry."""
    return {
        "MeasurementAxisIndex": axis_index,
        "CPMaps": [
            {
                "MeasurementOutcomeIndex": outcome_index,
                "CPMap": [[float(outcome_index)]],
            }
            for outcome_index in range(n_outcomes)
        ],
    }


def _lab(index, name, outcome_counts):
    """Build one PMF lab entry."""
    return {
        "Name": name,
        "Index": index,
        "NumberOfInQubits": 0,
        "NumberOfOutQubits": 0,
        "Measurements": [
            _measurement(axis_index, n_outcomes)
            for axis_index, n_outcomes in enumerate(outcome_counts)
        ],
    }


def _pmf_data(labs, wires):
    """Build a minimal process-matrix framework payload."""
    return {"ProcessMatrixFramework": {"Labs": labs, "Wires": wires}}


def _history(history_id, assignments, utilities):
    """Build one spacetime-game history entry."""
    return {
        "z": history_id,
        "h": [{"i": iset, "a": action} for iset, action in assignments],
        "u": [{"p": player, "v": value} for player, value in utilities.items()],
    }


def _leaf_payoffs(tree):
    """Collect all leaf payoffs from an extensive-form game tree."""
    if tree["kind"] == "outcome":
        return [tuple(tree["payoffs"])]

    leaves = []
    for child in tree["Children"]:
        leaves.extend(_leaf_payoffs(child))
    return leaves


@pytest.fixture
def pmf_no_links_data():
    """Return a minimal process matrix without causal links."""
    return _pmf_data(
        labs=[
            _lab(0, "Start", []),
            _lab(1, "End", []),
            _lab(2, "A", [2]),
            _lab(3, "B", [1]),
        ],
        wires=[],
    )


@pytest.fixture
def pmf_causal_link_data():
    """Return a minimal process matrix with one causal link."""
    return _pmf_data(
        labs=[
            _lab(0, "Start", []),
            _lab(1, "End", []),
            _lab(2, "A", [2]),
            _lab(3, "B", [1]),
        ],
        wires=[
            {
                "From": {"LabIdx": 2, "OutQubitLocalIdx": 0},
                "To": {"LabIdx": 3, "InQubitLocalIdx": 0},
            },
        ],
    )


@pytest.fixture
def pmf_mixed_data():
    """Return a minimal process matrix with one causal and one spacelike lab."""
    return _pmf_data(
        labs=[
            _lab(0, "Start", []),
            _lab(1, "End", []),
            _lab(2, "A", [2]),
            _lab(3, "B", [1]),
            _lab(4, "C", [1]),
        ],
        wires=[
            {
                "From": {"LabIdx": 2, "OutQubitLocalIdx": 0},
                "To": {"LabIdx": 3, "InQubitLocalIdx": 0},
            },
        ],
    )


@pytest.fixture
def pmf_cycle_data():
    """Return a minimal process matrix with a directed cycle."""
    return _pmf_data(
        labs=[
            _lab(0, "Start", []),
            _lab(1, "End", []),
            _lab(2, "A", [1]),
            _lab(3, "B", [1]),
        ],
        wires=[
            {
                "From": {"LabIdx": 2, "OutQubitLocalIdx": 0},
                "To": {"LabIdx": 3, "InQubitLocalIdx": 0},
            },
            {
                "From": {"LabIdx": 3, "OutQubitLocalIdx": 0},
                "To": {"LabIdx": 2, "InQubitLocalIdx": 0},
            },
        ],
    )


@pytest.fixture
def timelike_game_data():
    """Return a minimal timelike spacetime-game fixture."""
    return {
        "ps": ["Alice", "Bob"],
        "as": ["go", "wait", "x", "y"],
        "is": [
            {"i": "I1", "p": "Alice", "a": ["go", "wait"], "ns": [{"n": "N1", "ps": []}]},
            {
                "i": "I2",
                "p": "Bob",
                "a": ["x", "y"],
                "ns": [{"n": "N2", "ps": [{"p": "N1", "a": "go"}]}],
            },
        ],
        "z": [
            _history("H_go_x", [("I1", "go"), ("I2", "x")], {"Alice": 1, "Bob": 0}),
            _history("H_go_y", [("I1", "go"), ("I2", "y")], {"Alice": 0, "Bob": 1}),
            _history("H_wait", [("I1", "wait")], {"Alice": 0, "Bob": 0}),
        ],
    }


@pytest.fixture
def spacelike_game_data():
    """Return a minimal spacelike spacetime-game fixture."""
    return {
        "ps": ["Alice", "Bob"],
        "as": ["a1", "a2", "b1", "b2"],
        "is": [
            {"i": "I1", "p": "Alice", "a": ["a1", "a2"], "ns": [{"n": "N1", "ps": []}]},
            {"i": "I2", "p": "Bob", "a": ["b1", "b2"], "ns": [{"n": "N2", "ps": []}]},
        ],
        "z": [
            _history("H_11", [("I1", "a1"), ("I2", "b1")], {"Alice": 1, "Bob": 0}),
            _history("H_12", [("I1", "a1"), ("I2", "b2")], {"Alice": 0, "Bob": 1}),
            _history("H_21", [("I1", "a2"), ("I2", "b1")], {"Alice": 1, "Bob": 0}),
            _history("H_22", [("I1", "a2"), ("I2", "b2")], {"Alice": 0, "Bob": 1}),
        ],
    }


@pytest.fixture
def imperfect_info_game_data():
    """Return a minimal imperfect-information spacetime-game fixture."""
    return {
        "ps": ["Alice", "Bob"],
        "as": ["a", "b", "x", "y"],
        "is": [
            {"i": "I1", "p": "Alice", "a": ["a", "b"], "ns": [{"n": "N1", "ps": []}]},
            {
                "i": "I2",
                "p": "Bob",
                "a": ["x", "y"],
                "ns": [
                    {"n": "N2a", "ps": [{"p": "N1", "a": "a"}]},
                    {"n": "N2b", "ps": [{"p": "N1", "a": "b"}]},
                ],
            },
        ],
        "z": [
            _history("H_ax", [("I1", "a"), ("I2", "x")], {"Alice": 1, "Bob": 0}),
            _history("H_ay", [("I1", "a"), ("I2", "y")], {"Alice": 2, "Bob": 0}),
            _history("H_bx", [("I1", "b"), ("I2", "x")], {"Alice": 0, "Bob": 1}),
            _history("H_by", [("I1", "b"), ("I2", "y")], {"Alice": 0, "Bob": 2}),
        ],
    }


@pytest.fixture
def combined_game_data():
    """Return a minimal mixed spacetime-game fixture."""
    return {
        "ps": ["Alice", "Bob", "Charlie"],
        "as": ["a", "b", "c", "d", "x", "y"],
        "is": [
            {"i": "I1", "p": "Alice", "a": ["a", "b"], "ns": [{"n": "N1", "ps": []}]},
            {"i": "I3", "p": "Charlie", "a": ["c", "d"], "ns": [{"n": "N3", "ps": []}]},
            {
                "i": "I2",
                "p": "Bob",
                "a": ["x", "y"],
                "ns": [
                    {"n": "N2a", "ps": [{"p": "N1", "a": "a"}]},
                    {"n": "N2b", "ps": [{"p": "N1", "a": "b"}]},
                ],
            },
        ],
    }


def test_game_init_validate_and_repr(valid_game_data):
    """Verify game initialization, validation, and representation."""
    game = SpacetimeGame(deepcopy(valid_game_data))
    assert game.players == {"Alice", "Bob"}
    assert game.info_sets["I1"]["p"] == "Alice"
    assert game.adj["N1"][0]["c"] == "N2"

    with patch("jsonschema.validate") as mock_validate:
        mock_validate.return_value = None
        assert game.validate() is True
        mock_validate.side_effect = jsonschema.ValidationError("boom")
        assert game.validate() is False

    assert isinstance(repr(game), str)


def test_game_information_and_graph_checks(valid_game_data):
    """Verify information-set and graph integrity checks."""
    game = SpacetimeGame(deepcopy(valid_game_data))
    assert game.check_information_sets_consistency() is True
    assert game.check_node_graph_integrity() is True
    assert game.check_no_cycles() is True
    assert game.check_totality_and_cototality() is True
    assert game.check_histories_consistency() is True


def test_game_information_and_history_failures(valid_game_data):
    """Reject invalid information sets and histories."""
    bad_players = deepcopy(valid_game_data)
    bad_players["is"][0]["p"] = "Eve"
    with pytest.raises(ValueError, match="Union of players"):
        SpacetimeGame(bad_players).check_information_sets_consistency()

    bad_actions = deepcopy(valid_game_data)
    bad_actions["as"] = ["play"]
    with pytest.raises(ValueError, match="Union of actions"):
        SpacetimeGame(bad_actions).check_information_sets_consistency()

    bad_history = deepcopy(valid_game_data)
    bad_history["z"][0]["h"][0]["i"] = "I99"
    with pytest.raises(ValueError, match="references unknown information set"):
        SpacetimeGame(bad_history).check_histories_consistency()

    dup_history = deepcopy(valid_game_data)
    dup_history["z"][0]["h"] = [{"i": "I1", "a": "play"}, {"i": "I1", "a": "pass"}]
    with pytest.raises(ValueError, match="assigned more than one action"):
        SpacetimeGame(dup_history).check_histories_consistency()

    bad_utility = deepcopy(valid_game_data)
    bad_utility["z"][0]["u"] = [{"p": "Alice", "v": 1}]
    with pytest.raises(ValueError, match="missing players"):
        SpacetimeGame(bad_utility).check_histories_consistency()


def test_game_graph_and_totality_failures(valid_game_data):
    """Reject broken parent links and totality violations."""
    bad_parent = deepcopy(valid_game_data)
    bad_parent["is"][1]["ns"][0]["ps"][0]["p"] = "NonExistent"
    with pytest.raises(ValueError, match="Parental problems"):
        SpacetimeGame(bad_parent).check_node_graph_integrity()

    bad_cycle = deepcopy(valid_game_data)
    bad_cycle["is"][0]["ns"][0]["ps"] = [{"p": "N2", "a": "play"}]
    with pytest.raises(ValueError, match="Cycle detected"):
        SpacetimeGame(bad_cycle).check_no_cycles()

    bad_totality = deepcopy(valid_game_data)
    bad_totality["is"].append(
        {"i": "I3", "p": "Alice", "a": ["play"], "ns": [{"n": "N3", "ps": []}]}
    )
    with pytest.raises(ValueError, match="Totality violation"):
        SpacetimeGame(bad_totality).check_totality_and_cototality()

    bad_cototality = deepcopy(valid_game_data)
    bad_cototality["is"][1]["ns"][0]["ps"] = [{"p": "N1", "a": "impossible_action"}]
    with pytest.raises(ValueError, match="Co-totality violation"):
        SpacetimeGame(bad_cototality).check_totality_and_cototality()


def test_game_strategies_and_adders(valid_game_data):
    """Verify strategy checks and automatic adders."""
    game = SpacetimeGame(deepcopy(valid_game_data))

    game.add_played_information_sets()
    assert game.data["z"][0]["s"] == ["I1", "I2"]

    game.add_strategies()
    assert game.check_number_of_strategies() is True
    assert game.check_strategies_consistency() is True

    game.data.pop("rs", None)
    game.add_reduced_strategies()
    assert game.check_reduced_strategies_consistency() is True

    game.add_human_readable()
    assert set(game.data["h"]) == {"ns", "es", "ps", "as", "is", "z", "u", "s"}
    assert "ns" in repr(game)


def test_game_strategy_failures(valid_game_data):
    """Reject invalid strategies and reduced strategies."""
    unknown_player = deepcopy(valid_game_data)
    unknown_player["s"] = [{"p": "Charlie", "s": []}]
    with pytest.raises(ValueError, match="unknown player"):
        SpacetimeGame(unknown_player).check_strategies_consistency()

    foreign_iset = deepcopy(valid_game_data)
    foreign_iset["s"] = [{"p": "Alice", "s": [[{"i": "I2", "a": "play"}]]}]
    with pytest.raises(ValueError, match="belonging to player"):
        SpacetimeGame(foreign_iset).check_strategies_consistency()

    duplicate_strategy = deepcopy(valid_game_data)
    duplicate_strategy["s"] = [
        {"p": "Alice", "s": [[{"i": "I1", "a": "play"}], [{"i": "I1", "a": "play"}]]},
        {"p": "Bob", "s": [[{"i": "I2", "a": "pass"}]]},
    ]
    with pytest.raises(ValueError, match="Duplicate strategy"):
        SpacetimeGame(duplicate_strategy).check_strategies_consistency()

    bad_reduced = deepcopy(valid_game_data)
    bad_reduced["rs"] = [{"p": "Alice", "s": [[{"i": "I1", "a": "⟂"}]]}]
    with pytest.raises(ValueError, match="reachable but assigned '⟂'"):
        SpacetimeGame(bad_reduced).check_reduced_strategies_consistency()

    bad_reduced_action = deepcopy(valid_game_data)
    bad_reduced_action["rs"] = [{"p": "Alice", "s": [[{"i": "I1", "a": "invalid"}]]}]
    with pytest.raises(ValueError, match="Invalid action"):
        SpacetimeGame(bad_reduced_action).check_reduced_strategies_consistency()


def test_game_to_json_and_everything(tmp_path, valid_game_data):
    """Write game data to disk and run the full validation pipeline."""
    game = SpacetimeGame(deepcopy(valid_game_data))
    game.to_json(tmp_path / "game")
    assert (tmp_path / "game.json").exists()

    game.append_to_json_lines(tmp_path / "game_lines")
    assert (tmp_path / "game_lines.jsonl").exists()

    assert game.everything() is True


def test_alternating_game_checks():
    """Verify the alternating-game checks on converted CCS output."""
    ccs = CausallySecuredScenario(
        {
            "ms": [
                {"m": "A", "e": [], "o": [{"v": 0}, {"v": 1}]},
                {"m": "B", "e": [[{"m": "A", "v": 0}]], "o": [{"v": 0}, {"v": 1}]},
            ],
            "c": [["A", "B"]],
        }
    )
    game = AlternatingSpacetimeGame(ccs.to_spacetime_game())

    assert game.check_2_players() is True
    assert game.check_bipartite() is True
    assert game.check_roots_and_leaves() is True
    assert game.check_singleton_bob_info_sets() is True
    assert game.check_bob_a() is True
    assert game.check_ba1() is True
    assert game.check_ba2() is True
    assert game.check_ba3() is True
    assert game.check_ab1() is True
    assert game.check_ab2() is True
    assert game.check_even_height() is True


def test_base_game_checks_pass(valid_spacelike_separated_game_data):
    """Verify the core base-game checks on a disconnected spacelike-separated fixture."""

    game = SpacetimeGame(deepcopy(valid_spacelike_separated_game_data))
    assert game.check_information_sets_consistency() is True
    assert game.check_node_graph_integrity() is True
    assert game.check_no_cycles() is True
    assert game.all_checks() is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda data: data.__setitem__("ps", ["Alice", "Bob"]), "Union of players"),
        (lambda data: data.__setitem__("as", ["x"]), "Union of actions"),
    ],
)
def test_base_information_set_consistency_failures(base_game_data, mutator, match):
    """Reject information-set mismatches against the global player and action arrays."""

    data = deepcopy(base_game_data)
    mutator(data)
    with pytest.raises(ValueError, match=match):
        SpacetimeGame(data).check_information_sets_consistency()


def test_base_node_graph_integrity_failure(base_game_data):
    """Reject a parent reference to a missing node."""

    data = deepcopy(base_game_data)
    data["is"][1]["ns"][0]["ps"] = [{"p": "N999", "a": "x"}]
    with pytest.raises(ValueError, match="Parental problems"):
        SpacetimeGame(data).check_node_graph_integrity()


def test_base_cycle_detection_failure(base_game_data):
    """Reject a simple cycle between two nodes."""

    data = deepcopy(base_game_data)
    data["is"][0]["ns"][0]["ps"] = [{"p": "NA2", "a": "x"}]
    data["is"][1]["ns"][0]["ps"] = [{"p": "NA1", "a": "x"}]
    game = SpacetimeGame(data)
    assert game.check_node_graph_integrity() is True
    with pytest.raises(ValueError, match="Cycle detected"):
        game.check_no_cycles()


def test_totality_and_cototality_pass(two_history_history_data):
    """Verify totality and co-totality on a minimal supported history."""

    game = SpacetimeGame(deepcopy(two_history_history_data))
    assert game.check_totality_and_cototality() is True
    assert game.check_histories_consistency() is True


@pytest.mark.parametrize(
    ("history", "match"),
    [
        ([{"i": "IB", "a": "L"}], "Totality violation"),
        ([{"i": "IB", "a": "R"}, {"i": "IA", "a": "a"}], "Co-totality violation"),
    ],
)
def test_totality_and_cototality_failures(two_history_history_data, history, match):
    """Reject histories that violate totality or co-totality."""

    data = deepcopy(two_history_history_data)
    data["z"][0]["h"] = history
    data["z"][0]["u"] = [{"p": "Bob", "v": 0}, {"p": "Alfred", "v": 0}]
    with pytest.raises(ValueError, match=match):
        SpacetimeGame(data).check_totality_and_cototality()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda data: data["z"][0]["h"].__setitem__(0, {"i": "IX", "a": "L"}),
            "unknown information set",
        ),
        (
            lambda data: data["z"][0]["h"].append({"i": "IB", "a": "R"}),
            "assigned more than one action",
        ),
        (lambda data: data["z"][0]["u"].pop(), "missing players"),
        (
            lambda data: data["z"][0]["h"].__setitem__(0, {"i": "IB", "a": "invalid"}),
            "is not playable",
        ),
        (lambda data: data["z"][0].__setitem__("s", ["IA"]), "does not match the information sets"),
    ],
)
def test_history_consistency_failures(two_history_history_data, mutator, match):
    """Reject malformed histories and utility lists."""

    data = deepcopy(two_history_history_data)
    mutator(data)
    with pytest.raises(ValueError, match=match):
        SpacetimeGame(data).check_histories_consistency()


def test_add_played_information_sets(two_history_history_data):
    """Populate played information sets on missing histories."""

    data = deepcopy(two_history_history_data)
    del data["z"][0]["s"]
    game = SpacetimeGame(data)
    game.add_played_information_sets()
    assert game.data["z"][0]["s"] == ["IB", "IA"]


def test_add_strategies_generates_full_cartesian_product(strategy_game_data):
    """Generate all full strategies for a player with two information sets."""

    game = SpacetimeGame(deepcopy(strategy_game_data))
    game.add_strategies()

    alice_group = _group_for_player(game.data["s"], "Alice")
    bob_group = _group_for_player(game.data["s"], "Bob")
    charlie_group = _group_for_player(game.data["s"], "Charlie")

    assert len(alice_group["s"]) == 4
    assert len(bob_group["s"]) == 1
    assert len(charlie_group["s"]) == 1

    alice_strategies = {_strategy_set(strategy) for strategy in alice_group["s"]}
    assert alice_strategies == {
        frozenset({("IA1", "x"), ("IA2", "x")}),
        frozenset({("IA1", "x"), ("IA2", "y")}),
        frozenset({("IA1", "y"), ("IA2", "x")}),
        frozenset({("IA1", "y"), ("IA2", "y")}),
    }
    assert game.check_number_of_strategies() is True
    assert game.check_strategies_consistency() is True


@pytest.mark.parametrize(
    ("strategy_groups", "match"),
    [
        ([{"p": "John Doe", "s": []}], "unknown player"),
        (
            [
                {
                    "p": "Alice",
                    "s": [[{"i": "IA1", "a": "x"}, {"i": "IA2", "a": "x"}, {"i": "IB1", "a": "z"}]],
                }
            ],
            "belonging to player",
        ),
        (
            [
                {
                    "p": "Alice",
                    "s": [
                        [
                            {"i": "IA1", "a": "x"},
                            {"i": "IA2", "a": "x"},
                        ],
                        [
                            {"i": "IA1", "a": "x"},
                            {"i": "IA2", "a": "x"},
                        ],
                    ],
                }
            ],
            "Duplicate strategy",
        ),
    ],
)
def test_full_strategy_consistency_failures(strategy_game_data, strategy_groups, match):
    """Reject malformed full-strategy groups."""

    data = deepcopy(strategy_game_data)
    data["s"] = strategy_groups
    with pytest.raises(ValueError, match=match):
        SpacetimeGame(data).check_strategies_consistency()


def test_number_of_strategies_failure(strategy_game_data):
    """Reject repeated or missing strategy groups."""

    repeated = deepcopy(strategy_game_data)
    repeated["s"] = [{"p": "Alice", "s": []}, {"p": "Alice", "s": []}]
    assert SpacetimeGame(repeated).check_number_of_strategies() is False

    missing = deepcopy(strategy_game_data)
    missing["s"] = [{"p": "Alice", "s": []}, {"p": "Bob", "s": []}]
    assert SpacetimeGame(missing).check_number_of_strategies() is False


def test_add_histories_generates_expected_complete_histories(valid_alternating_game_data):
    """Generate the expected complete histories for the branching alternating fixture."""

    game = AlternatingSpacetimeGame(deepcopy(valid_alternating_game_data))
    game.add_histories()

    assert len(game.data["z"]) == 2
    assert {_assignment_set(history) for history in game.data["z"]} == {
        frozenset({("IB0", "L"), ("IA0", "a"), ("IB1", "b"), ("IA2", "b")}),
        frozenset({("IB0", "R"), ("IA1", "a")}),
    }


def test_add_played_information_sets_on_generated_histories(valid_alternating_game_data):
    """Populate played information sets after generating histories."""

    game = AlternatingSpacetimeGame(deepcopy(valid_alternating_game_data))
    game.add_histories()
    for history in game.data["z"]:
        history.pop("s", None)

    game.add_played_information_sets()
    assert all("s" in history for history in game.data["z"])
    assert all(
        set(history["s"]) <= {"IB0", "IA0", "IA1", "IB1", "IA2"} for history in game.data["z"]
    )


def test_add_reduced_strategies_generates_expected_reachability_split(valid_alternating_game_data):
    """Generate the expected reduced strategies for the branching alternating fixture."""

    game = AlternatingSpacetimeGame(deepcopy(valid_alternating_game_data))
    game.add_reduced_strategies()

    bob_group = _group_for_player(game.data["rs"], "Bob")
    alfred_group = _group_for_player(game.data["rs"], "Alfred")

    bob_strategies = {_strategy_set(strategy) for strategy in bob_group["s"]}
    alfred_strategies = {_strategy_set(strategy) for strategy in alfred_group["s"]}

    assert bob_strategies == {
        frozenset({("IB0", "L"), ("IB1", "b")}),
        frozenset({("IB0", "R"), ("IB1", "⟂")}),
    }
    assert alfred_strategies == {
        frozenset({("IA0", "a"), ("IA1", "a"), ("IA2", "b")}),
    }
    assert game.check_reduced_strategies_consistency() is True


@pytest.mark.parametrize(
    ("reduced_groups", "match"),
    [
        ([{"p": "Eve", "s": []}], "unknown player"),
        ([{"p": "Bob", "s": [[{"i": "IA0", "a": "a"}]]}], "foreign info set"),
        ([{"p": "Bob", "s": [[{"i": "IB0", "a": "invalid"}]]}], "Invalid action"),
        ([{"p": "Bob", "s": [[{"i": "IB0", "a": "⟂"}]]}], "reachable but assigned '⟂'"),
        (
            [{"p": "Bob", "s": [[{"i": "IB0", "a": "R"}, {"i": "IB1", "a": "b"}]]}],
            "not reachable but assigned real action",
        ),
        (
            [{"p": "Bob", "s": [[{"i": "IB0", "a": "L"}], [{"i": "IB0", "a": "L"}]]}],
            "Duplicate reduced strategy",
        ),
    ],
)
def test_reduced_strategy_consistency_failures(valid_alternating_game_data, reduced_groups, match):
    """Reject malformed reduced strategies."""

    data = deepcopy(valid_alternating_game_data)
    data["rs"] = reduced_groups
    with pytest.raises(ValueError, match=match):
        AlternatingSpacetimeGame(data).check_reduced_strategies_consistency()


@pytest.mark.parametrize(
    "method_name",
    [
        "check_2_players",
        "check_bipartite",
        "check_roots_and_leaves",
        "check_singleton_bob_info_sets",
        "check_bob_a",
        "check_ba1",
        "check_ba2",
        "check_ba3",
        "check_ab1",
        "check_ab2",
        "check_even_height",
    ],
)
def test_alternating_checks_pass(valid_alternating_game_data, method_name):
    """Verify the alternating-game checks on a valid fixture."""

    game = AlternatingSpacetimeGame(deepcopy(valid_alternating_game_data))
    assert getattr(game, method_name)() is True


def test_alternating_check_2_players_failure(valid_alternating_game_data):
    """Reject a game with three players."""

    data = deepcopy(valid_alternating_game_data)
    data["ps"] = ["Bob", "Alfred", "Charlie"]
    assert AlternatingSpacetimeGame(data).check_2_players() is False


def test_alternating_check_bipartite_failure(valid_alternating_game_data):
    """Reject an edge between nodes played by the same player."""

    data = deepcopy(valid_alternating_game_data)
    data["is"].append(
        {
            "i": "IBX",
            "p": "Bob",
            "a": ["b"],
            "ns": [{"n": "NBX", "ps": [{"p": "NB0", "a": "L"}]}],
        }
    )
    assert AlternatingSpacetimeGame(data).check_bipartite() is False


def test_alternating_check_roots_and_leaves_failure(valid_alternating_game_data):
    """Reject a Bob leaf node."""

    data = deepcopy(valid_alternating_game_data)
    data["is"].append(
        {
            "i": "IBX",
            "p": "Bob",
            "a": ["b"],
            "ns": [{"n": "NBX", "ps": []}],
        }
    )
    assert AlternatingSpacetimeGame(data).check_roots_and_leaves() is False


def test_alternating_check_singleton_bob_info_sets_failure(valid_alternating_game_data):
    """Reject a Bob information set with two nodes."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][0]["ns"].append({"n": "NB0b", "ps": []})
    assert AlternatingSpacetimeGame(data).check_singleton_bob_info_sets() is False


def test_alternating_check_bob_a_failure(valid_alternating_game_data):
    """Reject a Bob node that does not use all available actions."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][0]["a"] = ["L", "R", "extra"]
    assert AlternatingSpacetimeGame(data).check_bob_a() is False


def test_alternating_check_ba1_failure(valid_alternating_game_data):
    """Reject an Alfred node with two parents."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][1]["ns"][0]["ps"] = [{"p": "NB0", "a": "L"}, {"p": "NB1", "a": "b"}]
    assert AlternatingSpacetimeGame(data).check_ba1() is False


def test_alternating_check_ba2_failure(valid_alternating_game_data):
    """Reject two Alfred nodes with the same info set under one Bob action."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][1]["ns"].append({"n": "NA0b", "ps": [{"p": "NB0", "a": "L"}]})
    assert AlternatingSpacetimeGame(data).check_ba2() is False


def test_alternating_check_ba3_failure(valid_alternating_game_data):
    """Reject two different Bob labels pointing to the same Alfred info sets."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][1]["ns"].append({"n": "NA0b", "ps": [{"p": "NB0", "a": "R"}]})
    data["is"][2]["ns"].append({"n": "NA1b", "ps": [{"p": "NB0", "a": "L"}]})
    assert AlternatingSpacetimeGame(data).check_ba3() is False


def test_alternating_check_ab1_failure(valid_alternating_game_data):
    """Reject Alfred nodes in the same information set with different outgoing edges."""

    data = deepcopy(valid_alternating_game_data)
    data["is"][1]["ns"].append({"n": "NA0b", "ps": [{"p": "NB0", "a": "L"}]})
    assert AlternatingSpacetimeGame(data).check_ab1() is False


def test_alternating_check_ab2_failure(valid_alternating_game_data):
    """Reject duplicate Bob bridges."""

    data = deepcopy(valid_alternating_game_data)
    data["is"].append(
        {
            "i": "IB2",
            "p": "Bob",
            "a": ["b"],
            "ns": [{"n": "NB2", "ps": []}],
        }
    )
    assert AlternatingSpacetimeGame(data).check_ab2() is False


def test_alternating_check_even_height_failure(valid_alternating_game_data):
    """Reject a Bob leaf that breaks the parity condition."""

    data = deepcopy(valid_alternating_game_data)
    even_game = AlternatingSpacetimeGame(data)
    assert even_game.check_even_height() is True
    data["is"].append(
        {
            "i": "IBX",
            "p": "Bob",
            "a": ["b"],
            "ns": [{"n": "NBX", "ps": []}],
        }
    )
    odd_game = AlternatingSpacetimeGame(data)
    assert odd_game.check_even_height() is False


def test_alternating_all_adds_and_all_checks(valid_alternating_game_data):
    """Populate all alternating-game additions and verify the resulting checks."""

    game = AlternatingSpacetimeGame(deepcopy(valid_alternating_game_data))
    game.all_adds()

    assert "z" in game.data
    assert "s" in game.data
    assert "rs" in game.data
    assert "h" in game.data
    assert game.all_checks() is True


def test_from_process_matrix_without_causal_links(pmf_no_links_data):
    """Verify that spacelike labs stay independent in the conversion."""
    game = SpacetimeGame.from_process_matrix(deepcopy(pmf_no_links_data))

    assert game.players == {"A", "B", "Nature"}
    assert all(len(iset["ns"]) == 1 for iset in game.data["is"])
    assert sum(1 for iset in game.data["is"] if iset["p"] == "A") == 1
    assert sum(1 for iset in game.data["is"] if iset["p"] == "B") == 1
    assert sum(1 for iset in game.data["is"] if iset["p"] == "Nature") == 2
    assert sum(1 for iset in game.data["is"] if not iset["ns"][0]["ps"]) == 2


def test_from_process_matrix_with_causal_links_duplicates_contexts(pmf_causal_link_data):
    """Verify that causal links duplicate downstream lab contexts."""
    game = SpacetimeGame.from_process_matrix(deepcopy(pmf_causal_link_data))

    assert game.players == {"A", "B", "Nature"}
    assert all(len(iset["ns"]) == 1 for iset in game.data["is"])
    assert sum(1 for iset in game.data["is"] if iset["p"] == "A") == 1
    assert sum(1 for iset in game.data["is"] if iset["p"] == "B") == 2
    assert sum(1 for iset in game.data["is"] if iset["p"] == "Nature") == 3
    assert sum(1 for iset in game.data["is"] if not iset["ns"][0]["ps"]) == 1


def test_from_process_matrix_mixed_causal_and_spacelike_structure(pmf_mixed_data):
    """Verify that mixed causal structure preserves both links and independence."""
    game = SpacetimeGame.from_process_matrix(deepcopy(pmf_mixed_data))

    assert game.players == {"A", "B", "C", "Nature"}
    assert sum(1 for iset in game.data["is"] if iset["p"] == "A") == 1
    assert sum(1 for iset in game.data["is"] if iset["p"] == "B") == 2
    assert sum(1 for iset in game.data["is"] if iset["p"] == "C") == 1
    assert sum(1 for iset in game.data["is"] if iset["p"] == "Nature") == 4
    assert sum(1 for iset in game.data["is"] if not iset["ns"][0]["ps"]) == 2


def test_from_process_matrix_rejects_cycles(pmf_cycle_data):
    """Reject a process matrix whose experimental-lab graph contains a cycle."""
    with pytest.raises(ValueError, match="directed cycle"):
        SpacetimeGame.from_process_matrix(deepcopy(pmf_cycle_data))


def test_convert_to_extensive_game_timelike_and_utilities(timelike_game_data):
    """Verify timelike separation and terminal utilities in the extensive conversion."""
    game = SpacetimeGame(deepcopy(timelike_game_data))
    player_order = list(game.players)

    tree = game.convert_to_extensive_game(linearization=["N1", "N2"], match_utility=True)

    assert tree["kind"] == "choice"
    assert len(tree["Children"]) == 2
    assert tree["Children"][0]["kind"] == "choice"
    assert tree["Children"][1]["kind"] == "outcome"

    left_leaves = set(_leaf_payoffs(tree["Children"][0]))
    right_leaves = set(_leaf_payoffs(tree["Children"][1]))

    expected_left = {
        tuple({"Alice": 1, "Bob": 0}[player] for player in player_order),
        tuple({"Alice": 0, "Bob": 1}[player] for player in player_order),
    }
    expected_right = {tuple({"Alice": 0, "Bob": 0}[player] for player in player_order)}

    assert left_leaves == expected_left
    assert right_leaves == expected_right


def test_convert_to_extensive_game_spacelike_independence(spacelike_game_data):
    """Verify spacelike separation keeps the second root choice independent of the first."""
    game = SpacetimeGame(deepcopy(spacelike_game_data))
    player_order = list(game.players)

    tree = game.convert_to_extensive_game(linearization=["N1", "N2"], match_utility=True)

    assert tree["kind"] == "choice"
    assert len(tree["Children"]) == 2
    assert tree["Children"][0]["kind"] == tree["Children"][1]["kind"] == "choice"
    assert tree["Children"][0]["information-set"] == tree["Children"][1]["information-set"]

    left_leaves = set(_leaf_payoffs(tree["Children"][0]))
    right_leaves = set(_leaf_payoffs(tree["Children"][1]))
    expected = {
        tuple({"Alice": 1, "Bob": 0}[player] for player in player_order),
        tuple({"Alice": 0, "Bob": 1}[player] for player in player_order),
    }
    assert left_leaves == right_leaves == expected


def test_convert_to_extensive_game_imperfect_information(imperfect_info_game_data):
    """Verify imperfect information reuses the same information-set label across branches."""
    game = SpacetimeGame(deepcopy(imperfect_info_game_data))
    player_order = list(game.players)

    tree = game.convert_to_extensive_game(linearization=["N1", "N2a", "N2b"], match_utility=True)

    assert tree["kind"] == "choice"
    assert len(tree["Children"]) == 2
    assert tree["Children"][0]["kind"] == "choice"
    assert tree["Children"][1]["kind"] == "choice"
    assert tree["Children"][0]["information-set"] == tree["Children"][1]["information-set"]

    left_leaves = set(_leaf_payoffs(tree["Children"][0]))
    right_leaves = set(_leaf_payoffs(tree["Children"][1]))

    expected_left = {
        tuple({"Alice": 1, "Bob": 0}[player] for player in player_order),
        tuple({"Alice": 2, "Bob": 0}[player] for player in player_order),
    }
    expected_right = {
        tuple({"Alice": 0, "Bob": 1}[player] for player in player_order),
        tuple({"Alice": 0, "Bob": 2}[player] for player in player_order),
    }
    assert left_leaves == expected_left
    assert right_leaves == expected_right


def test_convert_to_extensive_game_combined_structure(combined_game_data):
    """Verify mixed timelike, spacelike, and imperfect-information structure."""
    game = SpacetimeGame(deepcopy(combined_game_data))

    tree = game.convert_to_extensive_game(
        linearization=["N1", "N3", "N2a", "N2b"], match_utility=False
    )

    assert tree["kind"] == "choice"
    assert len(tree["Children"]) == 2

    left_branch, right_branch = tree["Children"]
    assert left_branch["kind"] == right_branch["kind"] == "choice"
    assert left_branch["information-set"] == right_branch["information-set"]
    assert len(left_branch["Children"]) == 2
    assert len(right_branch["Children"]) == 2
    assert left_branch["Children"][0]["kind"] == right_branch["Children"][0]["kind"] == "choice"
    assert (
        left_branch["Children"][0]["information-set"]
        == right_branch["Children"][0]["information-set"]
    )


def test_strategies_must_cover_all_player_information_sets(valid_game_data):
    """Reject incomplete strategies and accept complete ones."""
    complete = deepcopy(valid_game_data)
    complete["s"] = [
        {
            "p": "Alice",
            "s": [[{"i": "I1", "a": "play"}]],
        },
        {
            "p": "Bob",
            "s": [[{"i": "I2", "a": "pass"}]],
        },
    ]
    assert SpacetimeGame(complete).check_strategies_consistency() is True

    incomplete = deepcopy(valid_game_data)
    incomplete["s"] = [
        {
            "p": "Alice",
            "s": [[]],  # missing the action for I1.
        },
        {
            "p": "Bob",
            "s": [[{"i": "I2", "a": "pass"}]],
        },
    ]

    with pytest.raises(ValueError, match="does not assign exactly one action"):
        SpacetimeGame(incomplete).check_strategies_consistency()
