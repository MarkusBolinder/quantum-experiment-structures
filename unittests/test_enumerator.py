"""Unit tests for the enumerator module."""

import pytest

import quantum_experiment_structures.enumerator as enumerator_module
from quantum_experiment_structures.enumerator import CCSEnumerator


OUTCOMES = [{"v": 0}, {"v": 1}]


def _ms(measurement, enabling_relation):
    """Return measurement dict."""
    return {"m": measurement, "e": enabling_relation, "o": OUTCOMES}


def test_init_rejects_n_less_than_one():
    """Raise when n is too small."""
    with pytest.raises(ValueError, match="n must be at least 1"):
        CCSEnumerator(n=0)


def test_init_rejects_names_with_wrong_length():
    """Trigger error when input names has wrong length."""
    with pytest.raises(ValueError, match="names must have length n"):
        CCSEnumerator(n=2, names=["A"])


def test_init_rejects_missing_static_covers(monkeypatch):
    """Reject n when no statically stored covers are available for it."""
    monkeypatch.setattr(enumerator_module, "LOCAL_COVERS", [[]], raising=False)

    with pytest.raises(ValueError, match="No static covers available for n=2"):
        CCSEnumerator(n=2)


def test_init_accepts_custom_names_and_covers():
    """Accept custom names and covers."""
    custom_covers = [[[0]], [[0, 1]]]
    enum = CCSEnumerator(n=1, names=["X"], covers=custom_covers)

    assert enum.n_measurements == 1
    assert enum.names == ["X"]
    assert enum.covers == custom_covers
    assert enum.allow_duplicates is False


def test_default_names_generation():
    """Generate default names when necessary."""
    enum = CCSEnumerator(n=2, covers=[[[0, 1]]])

    assert enum.names == ["A", "B"]


def test_rename_cover_uses_measurement_names():
    """Test method maps linearization indices to measurement names."""
    enum = CCSEnumerator(n=3, names=["X", "Y", "Z"], covers=[[[0, 1, 2]]])

    assert enum.rename_cover([[0, 2], [1]]) == [["X", "Z"], ["Y"]]


def test_get_all_nonempty_subsets():
    """Get all the nonempty subsets."""
    events = (("A", 0), ("A", 1))
    subsets = CCSEnumerator._get_all_nonempty_subsets(events)

    assert subsets == (
        frozenset({events[0]}),
        frozenset({events[1]}),
        frozenset({events[0], events[1]}),
    )


def test_context_key_orders_by_length_then_lexicographic():
    """Validate correctness of created context keys."""
    assert CCSEnumerator._context_key([("B", 1)]) == (1, (("B", 1),))
    assert CCSEnumerator._context_key([("A", 1), ("A", 0)]) == (
        2,
        (("A", 0), ("A", 1)),
    )


def test_incomparable_detects_subset_relation():
    """Trigger on inconsistent set of events."""
    a = frozenset({("A", 0)})
    b = frozenset({("A", 0), ("A", 1)})
    c = frozenset({("B", 0)})

    assert CCSEnumerator._incomparable(a, c) is True
    assert CCSEnumerator._incomparable(a, b) is False
    assert CCSEnumerator._incomparable(b, a) is False


def test_has_duplicate_measurements():
    """Test helper method for detecting inconsistent set of events."""
    assert CCSEnumerator._has_duplicate_measurements({("A", 0), ("A", 1)})
    assert not CCSEnumerator._has_duplicate_measurements({("A", 0), ("B", 1)})


def test_create_valid_enabling_relation_sorts_contexts_and_events():
    """Test sorting of covers and enabling relations."""
    enum = CCSEnumerator(n=2, names=["A", "B"], covers=[[[0, 1]]])
    relation = (
        frozenset({("B", 1), ("A", 0)}),
        frozenset({("A", 1)}),
    )

    assert enum._create_valid_enabling_relation(relation) == [
        [{"m": "A", "v": 1}],
        [{"m": "A", "v": 0}, {"m": "B", "v": 1}],
    ]


@pytest.mark.parametrize(
    ("allow_duplicates", "expected"),
    [
        (
            True,
            (
                frozenset({("A", 0)}),
                frozenset({("A", 1)}),
                frozenset({("A", 0), ("A", 1)}),
            ),
        ),
        (
            False,
            (
                frozenset({("A", 0)}),
                frozenset({("A", 1)}),
            ),
        ),
    ],
)
def test_candidate_contexts_respects_duplicate_setting(allow_duplicates, expected):
    """Test that the 'allow_duplicates' parameter works properly."""
    enum = CCSEnumerator(
        n=2, names=["A", "B"], covers=[[[0, 1]]], allow_duplicates=allow_duplicates
    )

    assert enum._candidate_contexts(("A",)) == expected


@pytest.mark.parametrize(
    ("allow_duplicates", "expected"),
    [
        (
            True,
            [
                [],
                [[{"m": "A", "v": 0}, {"m": "A", "v": 1}]],
                [[{"m": "A", "v": 1}]],
                [[{"m": "A", "v": 0}]],
                [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]],
            ],
        ),
        (
            False,
            [
                [],
                [[{"m": "A", "v": 1}]],
                [[{"m": "A", "v": 0}]],
                [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]],
            ],
        ),
    ],
)
def test_iter_enabling_relations_for_one_prior_measurement(allow_duplicates, expected):
    """Ensure the correct enabling relations are enumerated."""
    enum = CCSEnumerator(
        n=2, names=["A", "B"], covers=[[[0, 1]]], allow_duplicates=allow_duplicates
    )

    assert list(enum.iter_enabling_relations(("A",))) == expected


def test_iter_enabling_relations_with_no_prior_measurements():
    """Test flat scenario case."""
    enum = CCSEnumerator(n=1, names=["A"], covers=[[[0]]])

    assert list(enum.iter_enabling_relations(())) == [[]]


def test_enumerate_matches_expected_output_for_n_1_with_duplicates():
    """Validate results for n = 1 with duplicates allowed."""
    enum = CCSEnumerator(n=1, allow_duplicates=True)

    assert list(enum.enumerate()) == [
        {
            "ms": [
                {
                    "m": "A",
                    "e": [],
                    "o": OUTCOMES,
                }
            ],
            "c": [["A"]],
        }
    ]


def test_enumerate_matches_expected_output_for_n_1_without_duplicates():
    """Validate results for n = 1 with duplicates not allowed."""
    enum = CCSEnumerator(n=1, allow_duplicates=False)

    assert list(enum.enumerate()) == [
        {
            "ms": [
                {
                    "m": "A",
                    "e": [],
                    "o": OUTCOMES,
                }
            ],
            "c": [["A"]],
        }
    ]


def test_enumerate_matches_expected_output_for_n_2_with_duplicates():
    """Validate results for n = 2 with duplicates allowed."""
    enum = CCSEnumerator(n=2, allow_duplicates=True)

    expected = [
        {
            "ms": [
                _ms("A", []),
                _ms("B", []),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}, {"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 1}]]),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 0}]]),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}],
                        [{"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", []),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}, {"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 1}]]),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 0}]]),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}],
                        [{"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A"], ["B"]],
        },
    ]

    assert list(enum.enumerate()) == expected


def test_enumerate_matches_expected_output_for_n_2_without_duplicates():
    """Validate results for n = 2 with duplicates not allowed."""
    enum = CCSEnumerator(n=2, allow_duplicates=False)

    expected = [
        {
            "ms": [
                _ms("A", []),
                _ms("B", []),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 1}]]),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 0}]]),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}],
                        [{"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A", "B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", []),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 1}]]),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms("B", [[{"m": "A", "v": 0}]]),
            ],
            "c": [["A"], ["B"]],
        },
        {
            "ms": [
                _ms("A", []),
                _ms(
                    "B",
                    [
                        [{"m": "A", "v": 0}],
                        [{"m": "A", "v": 1}],
                    ],
                ),
            ],
            "c": [["A"], ["B"]],
        },
    ]

    assert list(enum.enumerate()) == expected
