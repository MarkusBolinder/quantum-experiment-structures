"""Pytest coverage for the core causal contextuality modules.

The tests focus on three layers:
1. Causal contextuality scenarios and their stable / secured subclasses.
2. Random scenario generation.
3. Spacetime game validation and add/check helpers.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import quantum_experiment_structures.generator as generator_module
from quantum_experiment_structures.generator import CCSGenerator
from quantum_experiment_structures.utils import utils


class ScriptedRNG:
    """Provide deterministic random values for generator tests."""

    def __init__(
        self,
        randint_value=0,
        randint_values=None,
        random_value=0.0,
        random_values=None,
        choice_value=0,
        choice_values=None,
        randrange_value=0,
        randrange_values=None,
        reverse_shuffle=False,
    ):
        self._randint_value = randint_value
        self._randint_values = iter(randint_values or [])
        self._random_value = random_value
        self._random_values = iter(random_values or [])
        self._choice_value = choice_value
        self._choice_values = iter(choice_values or [])
        self._randrange_value = randrange_value
        self._randrange_values = iter(randrange_values or [])
        self._reverse_shuffle = reverse_shuffle

    def randint(self, _a, _b):
        """Return the next scripted integer."""
        try:
            return next(self._randint_values)
        except StopIteration:
            return self._randint_value

    def random(self):
        """Return the scripted floating-point value."""
        try:
            return next(self._random_values)
        except StopIteration:
            return self._random_value

    def sample(self, population, k):
        """Return the first k items from the population."""
        return list(population)[:k]

    def choice(self, population):
        """Return the first item from the population."""
        try:
            return next(self._choice_values)
        except StopIteration:
            return self._choice_value

    def shuffle(self, items):
        """Reverse the sequence when requested."""
        if self._reverse_shuffle:
            items.reverse()

    def randrange(self, n):
        """Return the first valid index."""
        try:
            return next(self._randrange_values)
        except StopIteration:
            return self._randrange_value


class FakeCCS:
    """Stand in for CausalContextualityScenario during generator tests."""

    instances = []

    def __init__(self, data):
        self.data = deepcopy(data)
        self.validated = False
        FakeCCS.instances.append(self)

    def everything(self):
        """Pretend to validate the scenario successfully."""
        self.validated = True
        return True

    def to_json(self, filename, indent=None):
        """Write the stored scenario data to JSON."""
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".json")
        path.write_text(json.dumps(self.data, indent=indent))


@pytest.fixture
def default_settings():
    """Return a compact generator configuration."""
    return {
        "n_measurements_range": [2, 3],
        "n_values_range": [2, 3],
        "n_contexts_range": [1, 2],
        "context_size_range": [1, 2],
        "n_alternatives_range": [1, 2],
        "enabling_relation_size_range": [1, 2],
        "n_samples_per_causal_structure": 1,
        "p_has_enabled": 0.5,
        "n_alternatives_mean": 1.0,
        "enabling_relation_size_mean": 1.0,
        "no_lexicographic_order": False,
        "n_scenarios": 2,
        "batch_size": 1,
        "seed": 42,
        "output_dir": None,
    }


@pytest.fixture
def fixed_measurement_outcomes():
    """Return a fixed measurement-outcome map."""
    return {"A": [0, 1], "B": [0], "C": [0, 1, 2]}


def test_generator_init_and_ranges(default_settings):
    """Initialize the generator and reject descending ranges."""
    gen = CCSGenerator(**deepcopy(default_settings))
    assert gen.seed == 42
    assert gen.n_scenarios == 2

    invalid = deepcopy(default_settings)
    invalid["n_measurements_range"] = [5, 2]
    with pytest.raises(ValueError, match="Invalid range of values"):
        CCSGenerator(**invalid)


def test_generator_measurement_names_and_outcomes(default_settings):
    """Generate measurement names and fixed outcomes deterministically."""
    gen = CCSGenerator(**deepcopy(default_settings))
    assert gen._generate_measurement_names(0) == []
    assert gen._generate_measurement_names(3) == ["A", "B", "C"]
    assert gen._generate_measurement_names(28) == [chr(ord("A") + i) for i in range(26)] + [
        "AA",
        "AB",
    ]

    fixed = CCSGenerator(
        measurement_outcomes_dict={"A": [0, 1], "B": [2]}, **deepcopy(default_settings)
    )
    measurements, outcomes = fixed.sample_measurements_and_outcomes()
    assert measurements == ["A", "B"]
    assert outcomes == {"A": [0, 1], "B": [2]}


def test_generator_sampling_helpers(default_settings):
    """Verify context sampling, weighted sampling, and requirement merging."""
    gen = CCSGenerator(**deepcopy(default_settings))
    assert gen._weighted_count_sample(2.0, 1, 1) == 1
    sample = gen._weighted_count_sample(2.0, 1, 5)
    assert 1 <= sample <= 5

    merged = gen._merge_requirements({"A": 0}, {"B": 1})
    assert merged == {"A": 0, "B": 1}
    assert gen._merge_requirements({"A": 0}, {"A": 1}) is None

    contexts = gen.sample_contexts(["A", "B", "C"])
    covered = {measurement for context in contexts for measurement in context}
    assert covered == {"A", "B", "C"}
    assert all(1 <= len(context) <= 2 for context in contexts)


def test_generator_enabling_relations(default_settings):
    """Generate acyclic enabling relations with deterministic ordering."""
    gen = CCSGenerator(**deepcopy(default_settings))
    measurements = ["A", "B", "C"]
    outcomes = {"A": [0, 1], "B": [0, 1], "C": [0, 1]}
    enabling = gen.generate_enabling_relations(measurements, outcomes)

    order = {name: idx for idx, name in enumerate(measurements)}
    for target, relations in enabling.items():
        for relation in relations:
            parents = [event["m"] for event in relation]
            assert len(parents) == len(set(parents))
            for parent in parents:
                assert order[parent] < order[target]


def test_generator_cover_generation(default_settings):
    """Generate a causally secured cover from fixed bridges."""
    gen = CCSGenerator(**deepcopy(default_settings))
    measurements = ["A", "B", "C"]
    enabling_relations = {
        "A": [],
        "B": [[{"m": "A", "v": 0}]],
        "C": [[{"m": "A", "v": 0}]],
    }

    def fake_local_cover(rhs):
        return [[m] for m in rhs]

    gen.sample_local_cover = fake_local_cover  # type: ignore
    cover = gen.generate_causally_secured_cover(measurements, enabling_relations)
    assert sorted(map(sorted, cover)) == sorted([sorted(["A", "B"]), sorted(["A", "C"])])

    bad = deepcopy(enabling_relations)
    bad["C"] = [[{"m": "A", "v": 0}], [{"m": "A", "v": 1}]]
    with pytest.raises(ValueError, match="multiple enabling relations"):
        gen.generate_causally_secured_cover(measurements, bad)


def test_init_and_range_validation(default_settings):
    """Verify initialization and ascending-range validation."""
    gen = CCSGenerator(**deepcopy(default_settings))
    assert gen.seed == 42
    assert gen.n_scenarios == 2
    assert gen.batch_size == 1
    assert gen.output_dir is None

    bad = deepcopy(default_settings)
    bad["n_measurements_range"] = [4, 2]
    with pytest.raises(ValueError, match="Invalid range of values"):
        CCSGenerator(**bad)


def test_generate_measurement_names_and_tuple_letter_generator(default_settings):
    """Verify measurement-name and tuple-letter helpers."""
    gen = CCSGenerator(**deepcopy(default_settings))

    assert gen._generate_measurement_names(0) == []
    assert gen._generate_measurement_names(3) == ["A", "B", "C"]
    assert gen._generate_measurement_names(28)[-2:] == ["AA", "AB"]
    assert list(gen._tuple_letter_generator("AB", 1)) == [("A",), ("B",)]
    assert list(gen._tuple_letter_generator("AB", 2)) == [
        ("A", "A"),
        ("A", "B"),
        ("B", "A"),
        ("B", "B"),
    ]


def test_sample_measurements_and_outcomes_fixed_and_random(
    default_settings, fixed_measurement_outcomes
):
    """Verify fixed and sampled measurement-outcome selection."""
    fixed_gen = CCSGenerator(
        measurement_outcomes_dict=deepcopy(fixed_measurement_outcomes), **deepcopy(default_settings)
    )
    measurements, outcomes = fixed_gen.sample_measurements_and_outcomes()
    assert measurements == ["A", "B", "C"]
    assert outcomes == fixed_measurement_outcomes

    random_gen = CCSGenerator(**deepcopy(default_settings))
    random_gen._generate_measurement_names = lambda n: ["A", "B"]
    random_gen.rng = ScriptedRNG(randint_values=[2, 1, 2])  # type: ignore
    measurements, outcomes = random_gen.sample_measurements_and_outcomes()
    assert measurements == ["A", "B"]
    assert outcomes == {"A": [0], "B": [0, 1]}


def test_weighted_count_sample_boundaries(default_settings):
    """Verify lower-bound and probabilistic sampling branches."""
    gen = CCSGenerator(**deepcopy(default_settings))
    gen.rng = ScriptedRNG(random_value=0.0)  # type: ignore
    assert gen._weighted_count_sample(mean=1.0, min_k=1, max_k=1) == 1
    assert gen._weighted_count_sample(mean=1.0, min_k=1, max_k=3) == 1

    gen.rng = ScriptedRNG(random_value=1.0)  # type: ignore
    assert gen._weighted_count_sample(mean=1.0, min_k=1, max_k=3) == 3


def test_merge_requirements(default_settings):
    """Verify compatible and conflicting requirement merges."""
    gen = CCSGenerator(**deepcopy(default_settings))
    assert gen._merge_requirements({"A": 0}, {"B": 1}) == {"A": 0, "B": 1}
    assert gen._merge_requirements({"A": 0}, {"A": 0, "B": 1}) == {"A": 0, "B": 1}
    assert gen._merge_requirements({"A": 0}, {"A": 1}) is None


def test_generate_enabling_relations_without_and_with_ordering(default_settings):
    """Verify empty enabling relations and causal ordering constraints."""
    measurements = ["A", "B", "C"]
    outcomes = {"A": [0, 1], "B": [0, 1], "C": [0, 1]}

    gen = CCSGenerator(**deepcopy(default_settings))
    gen.settings["p_has_enabled"] = 0.0
    empty_relations = gen.generate_enabling_relations(measurements, outcomes)
    assert empty_relations == {"A": [], "B": [], "C": []}

    gen = CCSGenerator(**deepcopy(default_settings))
    gen.settings.update(
        {
            "p_has_enabled": 1.0,
            "n_alternatives_range": [1, 1],
            "enabling_relation_size_range": [1, 1],
            "n_alternatives_mean": 1.0,
            "enabling_relation_size_mean": 1.0,
            "no_lexicographic_order": False,
        }
    )
    gen._weighted_count_sample = lambda mean, min_k, max_k: 1
    gen.rng = ScriptedRNG(random_value=0.0)  # type: ignore
    ordered = gen.generate_enabling_relations(measurements, outcomes)
    assert ordered["A"] == []
    assert ordered["B"] == [[{"m": "A", "v": 0}]]
    assert ordered["C"] == [[{"m": "A", "v": 0}]]

    gen = CCSGenerator(**deepcopy(default_settings))
    gen.settings.update(
        {
            "p_has_enabled": 1.0,
            "n_alternatives_range": [1, 1],
            "enabling_relation_size_range": [1, 1],
            "n_alternatives_mean": 1.0,
            "enabling_relation_size_mean": 1.0,
            "no_lexicographic_order": True,
        }
    )
    gen._weighted_count_sample = lambda mean, min_k, max_k: 1
    gen.rng = ScriptedRNG(random_value=0.0, reverse_shuffle=True)  # type: ignore
    shuffled = gen.generate_enabling_relations(measurements, outcomes)
    assert shuffled["A"] == [[{"m": "B", "v": 0}]]
    assert shuffled["B"] == [[{"m": "C", "v": 0}]]
    assert shuffled["C"] == []


def test_generate_enabling_relations_deduplicates_repeated_relations(default_settings):
    """Verify duplicate enabling relations are skipped."""
    measurements = ["A", "B"]
    outcomes = {"A": [0, 1], "B": [0, 1]}

    gen = CCSGenerator(**deepcopy(default_settings))
    gen.settings.update(
        {
            "p_has_enabled": 1.0,
            "n_alternatives_range": [2, 2],
            "enabling_relation_size_range": [1, 1],
            "n_alternatives_mean": 1.0,
            "enabling_relation_size_mean": 1.0,
            "no_lexicographic_order": False,
        }
    )
    gen._weighted_count_sample = lambda mean, min_k, max_k: 2
    gen.rng = ScriptedRNG(random_value=0.0)  # type: ignore
    relations = gen.generate_enabling_relations(measurements, outcomes)
    assert relations["A"] == []
    assert relations["B"] == [[{"m": "A", "v": 0}]]


def test_sample_contexts_covers_all_measurements(default_settings):
    """Verify sampled contexts cover the input measurements."""
    gen = CCSGenerator(**deepcopy(default_settings))
    gen.settings["n_contexts_range"] = [1, 1]
    gen.settings["context_size_range"] = [1, 1]
    gen.rng = ScriptedRNG(randint_values=[1, 1, 1, 1])  # type: ignore
    contexts = gen.sample_contexts(["A", "B", "C"])

    covered = {m for context in contexts for m in context}
    assert covered == {"A", "B", "C"}
    assert len(contexts) == len({frozenset(c) for c in contexts})


def test_generate_local_cover_fallback_preserves_cover(default_settings):
    """Verify the MCMC local-cover fallback produces a valid cover."""
    gen = CCSGenerator(**deepcopy(default_settings))
    measurements = [chr(ord("A") + i) for i in range(10)]
    cover = gen._generate_local_cover(measurements, iterations=12)

    assert {m for context in cover for m in context} == set(measurements)
    assert utils.is_antichain(set(frozenset(context) for context in cover))


def test_generate_causally_secured_cover_simple(default_settings, monkeypatch):
    """Verify a simple secured cover with no causal links."""
    gen = CCSGenerator(**deepcopy(default_settings))

    measurements = ["A", "B"]
    relations = {"A": [], "B": []}
    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [list(rhs)])
    cover = gen.generate_causally_secured_cover(measurements, relations)
    assert cover == [["A", "B"]]


def test_generate_causally_secured_cover_rejects_bad_structures(default_settings):
    """Reject multiple bridges, cycles, and inconsistent closures."""
    gen = CCSGenerator(**deepcopy(default_settings))

    with pytest.raises(ValueError, match="multiple enabling relations"):
        gen.generate_causally_secured_cover(
            ["A", "B"],
            {"A": [[{"m": "B", "v": 0}], [{"m": "B", "v": 1}]], "B": []},
        )

    with pytest.raises(ValueError, match="Cyclic enabling relation detected"):
        gen.generate_causally_secured_cover(
            ["A", "B"],
            {"A": [[{"m": "B", "v": 0}]], "B": [[{"m": "A", "v": 0}]]},
        )

    with pytest.raises(ValueError, match="Inconsistent enabling relations detected"):
        gen.generate_causally_secured_cover(
            ["A", "B", "C"],
            {
                "A": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
                "B": [[{"m": "C", "v": 1}]],
                "C": [],
            },
        )


def test_ccs_generator_yields_validated_scenarios(default_settings, monkeypatch):
    """Verify the internal generator yields validated CCS objects."""
    FakeCCS.instances.clear()
    gen = CCSGenerator(**deepcopy(default_settings))
    monkeypatch.setattr(generator_module.qes, "CausalContextualityScenario", FakeCCS)
    monkeypatch.setattr(generator_module.utils, "create_anti_chain", lambda contexts: contexts)
    monkeypatch.setattr(
        gen, "sample_measurements_and_outcomes", lambda: (["A", "B"], {"A": [0], "B": [0, 1]})
    )
    monkeypatch.setattr(
        gen,
        "generate_enabling_relations",
        lambda measurements, outcomes: {"A": [], "B": []},
    )
    monkeypatch.setattr(gen, "sample_contexts", lambda measurements: [["A", "B"]])

    scenarios = list(gen._ccs_generator())
    assert len(scenarios) == 2
    assert all(isinstance(ccs, FakeCCS) for ccs in scenarios)
    assert all(ccs.validated for ccs in scenarios)  # type: ignore
    assert scenarios[0].data["ms"][0]["m"] == "A"
    assert scenarios[0].data["c"] == [["A", "B"]]


def test_generate_returns_generator_without_output(default_settings, monkeypatch):
    """Verify generate returns a generator when output is disabled."""
    gen = CCSGenerator(**deepcopy(default_settings))
    fake = FakeCCS({"ms": [], "c": []})
    monkeypatch.setattr(gen, "_ccs_generator", lambda: iter([fake]))

    produced = list(gen.generate())
    assert produced == [fake]


def test_generate_writes_json_and_jsonl(tmp_path, default_settings, monkeypatch):
    """Verify generate writes both JSON and JSONL outputs."""
    # One-file-per-scenario path.
    json_settings = deepcopy(default_settings)
    json_settings.update(
        {"output_dir": str(tmp_path / "json_out"), "batch_size": 1, "n_scenarios": 2}
    )
    gen = CCSGenerator(**json_settings)
    monkeypatch.setattr(
        gen,
        "_ccs_generator",
        lambda: iter([FakeCCS({"id": 0}), FakeCCS({"id": 1})]),
    )
    gen.generate()
    assert (tmp_path / "json_out" / "part_0.jsonl").exists()
    assert (tmp_path / "json_out" / "part_1.jsonl").exists()

    # Batched JSONL path.
    jsonl_settings = deepcopy(default_settings)
    jsonl_settings.update(
        {"output_dir": str(tmp_path / "jsonl_out"), "batch_size": 2, "n_scenarios": 3}
    )
    gen = CCSGenerator(**jsonl_settings)
    monkeypatch.setattr(
        gen,
        "_ccs_generator",
        lambda: iter([FakeCCS({"id": 0}), FakeCCS({"id": 1}), FakeCCS({"id": 2})]),
    )
    gen.generate()

    files = sorted((tmp_path / "jsonl_out").glob("part_*.jsonl"))
    assert len(files) == 2
    assert files[0].read_text().count("\n") == 2
    assert json.loads(files[0].read_text().splitlines()[0]) == {"id": 0}


def test_sample_contexts_runs_additional_sampling_loop(default_settings):
    """Exercise the extra cover-sampling loop."""
    gen = CCSGenerator(**default_settings)
    gen.settings["n_contexts_range"] = [3, 3]
    gen.settings["context_size_range"] = [3, 3]
    gen.rng = ScriptedRNG(randint_values=[3, 3, 3])  # type: ignore

    contexts = gen.sample_contexts(["A", "B", "C", "D"])

    assert set(m for c in contexts for m in c) == {"A", "B", "C", "D"}
    assert len(contexts) >= 2


def test_sample_contexts_raises_when_no_context_is_possible(default_settings):
    """Raise when the sampled cover stays empty."""
    gen = CCSGenerator(**default_settings)
    gen.settings["n_contexts_range"] = [0, 0]
    gen.rng = ScriptedRNG(randint_values=[0])  # type: ignore

    with pytest.raises(RuntimeError, match="Failed to sample any contexts"):
        gen.sample_contexts([])


def test_generate_enabling_relations_breaks_when_max_size_is_zero(default_settings):
    """Hit the early break in enabling-relation sampling."""
    gen = CCSGenerator(**default_settings)
    gen.settings["p_has_enabled"] = 1.0
    gen.settings["n_alternatives_range"] = [1, 1]
    gen.settings["enabling_relation_size_range"] = [0, 0]
    gen.rng = ScriptedRNG(random_values=[0.0, 0.0])  # type: ignore

    enabled = gen.generate_enabling_relations(["A", "B"], {"A": [0], "B": [0]})

    assert enabled == {"A": [], "B": []}


def test_generate_enabling_relations_skips_empty_outcomes_and_empty_events(default_settings):
    """Exercise the continue branches in enabling-relation generation."""
    gen = CCSGenerator(**default_settings)
    gen.settings["p_has_enabled"] = 1.0
    gen.settings["n_alternatives_range"] = [1, 1]
    gen.settings["enabling_relation_size_range"] = [1, 1]
    gen.rng = ScriptedRNG(random_values=[0.0, 0.0])  # type: ignore

    enabled = gen.generate_enabling_relations(["A", "B"], {"A": [], "B": [0]})

    assert enabled["A"] == []
    assert enabled["B"] == []


def test_generate_local_cover_adds_missing_singletons(default_settings):
    """Exercise the cleanup branch that restores missing measurements."""
    gen = CCSGenerator(**default_settings)
    gen.rng = ScriptedRNG(  # type: ignore
        random_values=[0.1, 0.5],
        randint_values=[2],
        randrange_values=[0],
        choice_values=["A"],
    )

    cover = gen._generate_local_cover(["A", "B", "C"], iterations=2)

    flattened = {m for context in cover for m in context}
    assert flattened == {"A", "B", "C"}
    assert ["A"] in cover


def test_sample_local_cover_uses_static_cover(default_settings):
    """Exercise the static local-cover lookup path."""
    gen = CCSGenerator(**default_settings)

    cover = gen.sample_local_cover(["A"])

    assert set(m for context in cover for m in context) == {"A"}
    assert utils.is_antichain(set(frozenset(context) for context in cover))


def test_generate_causally_secured_cover_rejects_conflicting_closure(default_settings):
    """Reject a transitive closure with conflicting requirements."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B", "C", "D"]
    enabling_relations = {
        "B": [[{"m": "A", "v": 0}]],
        "C": [[{"m": "A", "v": 1}]],
        "D": [[{"m": "B", "v": 0}, {"m": "C", "v": 0}]],
    }

    with pytest.raises(ValueError, match="Inconsistent enabling relations detected while closing"):
        gen.generate_causally_secured_cover(measurements, enabling_relations)


def test_generate_causally_secured_cover_rejects_unclean_block_when_resampling_fails(
    default_settings, monkeypatch
):
    """Reject a bad local cover when resampling cannot fix it."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A", "B"]])
    monkeypatch.setattr(gen, "_merge_requirements", lambda left, right: None)

    with pytest.raises(ValueError, match="Failed to sample a clean local cover"):
        gen.generate_causally_secured_cover(
            measurements,
            enabling_relations,
            allow_unclean_local_covers=False,
            max_partition_tries=1,
        )


def test_generate_causally_secured_cover_falls_back_to_singletons_when_unclean_allowed(
    default_settings, monkeypatch
):
    """Split an unclean block into singleton contexts when allowed."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A", "B"]])
    monkeypatch.setattr(gen, "_merge_requirements", lambda left, right: None)

    cover = gen.generate_causally_secured_cover(
        measurements,
        enabling_relations,
        allow_unclean_local_covers=True,
        max_partition_tries=1,
    )

    assert sorted(map(tuple, cover)) == [("A",), ("B",)]


def test_generate_causally_secured_cover_adds_missing_measurements_when_allowed(
    default_settings, monkeypatch
):
    """Append missing measurements to the final cover when unclean local covers are allowed."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A"]])

    cover = gen.generate_causally_secured_cover(
        measurements,
        enabling_relations,
        allow_unclean_local_covers=True,
        max_partition_tries=1,
    )

    assert sorted(map(tuple, cover)) == [("A",), ("B",)]


def test_sample_local_cover_uses_generated_cover_for_five_or_more_measurements(
    default_settings, monkeypatch
):
    """Use the non-cached local-cover generator for larger measurement sets."""
    from quantum_experiment_structures.data.local_covers import LOCAL_COVERS

    gen = CCSGenerator(**default_settings)
    measurements = [f"M{i}" for i in range(len(LOCAL_COVERS) + 1)]

    called = {"count": 0}

    def fake_generate_local_cover(ms):
        called["count"] += 1
        return [[m] for m in ms]

    monkeypatch.setattr(gen, "_generate_local_cover", fake_generate_local_cover)

    cover = gen.sample_local_cover(measurements)

    assert called["count"] == 1
    assert set(m for context in cover for m in context) == set(measurements)


def test_generate_causally_secured_cover_raises_when_no_clean_local_cover_can_be_sampled(
    default_settings, monkeypatch
):
    """Raise when every sampled local cover is rejected as unclean."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A", "B"]])
    monkeypatch.setattr(gen, "_merge_requirements", lambda left, right: None)

    with pytest.raises(ValueError, match="Failed to sample a clean local cover"):
        gen.generate_causally_secured_cover(
            measurements,
            enabling_relations,
            allow_unclean_local_covers=False,
            max_partition_tries=1,
        )


def test_generate_causally_secured_cover_raises_on_internally_inconsistent_block(
    default_settings, monkeypatch
):
    """Raise when a sampled block becomes inconsistent during conversion."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    call_count = {"n": 0}

    def merge_requirements(left, right):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return {}
        return None

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A", "B"]])
    monkeypatch.setattr(gen, "_merge_requirements", merge_requirements)

    with pytest.raises(ValueError, match="internally inconsistent"):
        gen.generate_causally_secured_cover(
            measurements,
            enabling_relations,
            allow_unclean_local_covers=False,
            max_partition_tries=1,
        )


def test_generate_causally_secured_cover_raises_when_final_cover_missing_measurements(
    default_settings, monkeypatch
):
    """Raise when the final cover misses a measurement and relaxation is disabled."""
    gen = CCSGenerator(**default_settings)
    measurements = ["A", "B"]
    enabling_relations = {}

    monkeypatch.setattr(gen, "sample_local_cover", lambda rhs: [["A"]])

    with pytest.raises(ValueError, match="does not cover all measurements"):
        gen.generate_causally_secured_cover(
            measurements,
            enabling_relations,
            allow_unclean_local_covers=False,
            max_partition_tries=1,
        )
