#!/usr/bin/env python3
"""Random causal contextuality scenario generator.

This module provides utilities for generating random causal contextuality
scenarios (CCSs). A scenario consists of measurements, outcomes, contexts
(sets of jointly measurable events), and causal enabling relations between
measurements.

The module can be used in two ways:

1. Command line interface (CLI) for batch generation of datasets using the script in
   quantum_experiment_structures/utils/ccs_generator_script.py
2. Python API for programmatic scenario generation with full control
   over parameters.

Generated scenarios can be written either as individual JSON files or as
JSON Lines files suitable for large datasets.

When using the Python API, ranges are represented by 2-tuples, but any iterable
containing two integers should work.

Example usage:
    Python API usage:

        from quantum_experiment_structures.generator import CCSGenerator

        kwargs = {
            "n_measurements_range": [4, 6],
            "n_values_range": [2, 2]  # all measurements will have two outcomes
            "n_contexts_range": [3, 5],
            "context_size_range": [2, 3],
            "n_scenarios": 5,
            "seed": 42,
        }

        generator = CCSGenerator(**kwargs)

        generator.generate()

        # the generator object contains the scenarios if they were not written to disk
"""

import json
import math
import os
from pathlib import Path
import random
import string

import quantum_experiment_structures as qes
from quantum_experiment_structures.data.local_covers import LOCAL_COVERS
from quantum_experiment_structures.data.schemas import CCS_GENERATOR_SETTINGS_SCHEMA
from quantum_experiment_structures.utils import utils

# NOTE: need to pip install local package to be able to run this as a script
# it can be run as a module using:
#   python -m quantum_experiment_structures.generator [OPTIONS]
# but NOT using:
#   python quantum_experiment_structures/generator.py [OPTIONS]

# TODO: add type checking/annotations


class CCSGenerator:
    """Generator for random causal contextuality scenarios.

    The generator samples measurements, outcomes, contexts, and causal enabling
    relations according to configurable distributions and ranges. It can be
    used either through the CLI or directly as a Python object.

    Attributes:
        output_dir (str | None):
            Directory where generated scenarios are written. If None,
            results are only returned in memory.
        batch_size (int | None):
            Number of scenarios written per JSON Lines file.
            If set to 1, scenarios are written as individual JSON files.
        n_scenarios (int):
            Number of scenarios to generate.
        seed (int | None):
            Random seed for reproducibility.
        rng (random.Random):
            Random number generator used internally for sampling.
        scenarios (list[dict]):
            Generated causal contextuality scenarios.
        measurement_outcomes_dict (dict[str, list[int]] | None): maps measurement names to their
            allowed outcome values and will be used instead of sampling measurements and outcomes
            randomly.

        settings (dict):
            Configuration dictionary controlling the generation process.
            Keys correspond to CLI options.

            n_measurements_range (str):
                Range for the number of measurements in the scenario.
                Accepts either 'k' or 'min:max'.
            n_values_range (str):
                Range for the number of outcomes per measurement.
            n_contexts_range (str):
                Range for the number of contexts to sample.
            context_size_range (str):
                Range for the size of each context (number of measurements
                appearing in a context).
            n_alternatives_range (str):
                Range for the number of alternative enabling relations
                per measurement.
            enabling_relation_size_range (str):
                Range for the size of a single enabling relation
                (number of measurements required to enable another).
            n_samples_per_causal_structure (int):
                Number of contextual covers generated for a fixed causal
                structure (set of enabling relations).
            p_has_enabled (float):
                Probability that a measurement has at least one enabling
                relation.
            n_alternatives_mean (float):
                Mean number of alternative enabling relations for a
                measurement.
            enabling_relation_size_mean (float):
                Mean size of an enabling relation.
            no_lexicographic_order (bool):
                If False, impose lexicographic ordering on measurements
                when constructing enabling relations to guarantee
                acyclicity. Otherwise, a random order is used.

    Notes:
        A causal contextuality scenario consists of:

        - Measurements: labeled measurement settings.
        - Values: possible outcomes for each measurement.
        - Contexts: sets of measurements that can be jointly performed.
        - Enabling relations: causal dependencies specifying which
          measurements must occur before others.

        The generator samples these components independently within
        the constraints imposed by the provided ranges and means.
    """

    def __init__(self, measurement_outcomes_dict=None, **kwargs):
        """Create an instance of the CCSGenerator that drives random scenario generation.

        The constructor initializes internal state (settings, RNG seeded from seed, output and
        batching options) and performs basic validation and defaulting. Many generation parameters
        are passed via keyword arguments and stored in self.settings; certain program-level options
        are popped out of self.settings and assigned to explicit attributes (seed, output_dir,
        batch_size, n_scenarios, etc).

        Args:
            measurement_outcomes_dict (Optional[dict[str, list[int]]]): If provided, this dict maps
                measurement names to their allowed outcome values and will be used instead of
                sampling measurements and outcomes randomly.
            **kwargs: Arbitrary keyword settings controlling generation. Important keys that may be
                expected in kwargs (and are pulled out during initialization) include:
                - seed: integer seed for the RNG (required).
                - output_dir: path-like output directory (or None to avoid file writes).
                - n_contexts_per_causal_structure: how many covers to sample per causal structure.
                - n_scenarios: total number of scenarios to generate.
                - batch_size: number of scenarios to flush per output file or write batch.
                Additional, generation parameters will be defaulted by _add_default_values().

        Side effects:
            - Creates self.rng as a random.Random instance seeded with seed.
            - Calls self._add_default_values() to populate missing options with defaults.
            - Calls self._check_ranges() to validate any range-valued settings.
            - Initializes self.scenarios as an empty list and sets self.measurement_outcomes_dict.
        """
        # TODO: think about in-memory processing vs. streaming the data
        self.settings = kwargs

        # insert default values and check values with the schema for the settings
        validator = utils.DefaultValuesValidator(CCS_GENERATOR_SETTINGS_SCHEMA)
        validator.validate(self.settings)

        # extract program level kwargs, i.e. those that are not specifying the generation of a
        # single causal cotextuality scenario
        self.seed = self.settings.pop("seed")
        self.rng = random.Random(self.seed)
        self.output_dir = self.settings.pop("output_dir")
        self.n_contexts_per_causal_structure = self.settings.pop("n_samples_per_causal_structure")
        self.n_scenarios = self.settings.pop("n_scenarios")
        self.batch_size = self.settings.pop("batch_size")

        # initialize other attributes
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.n_file_magnitude = math.ceil(math.log(self.n_scenarios / self.batch_size, 10))
            self._batch_number = 0  # used to keep track of the current file to write to

        self.scenarios = []
        self.measurement_outcomes_dict = measurement_outcomes_dict

        # correct and check the settings
        self._check_ranges()

    def _check_ranges(self):
        """Validate all settings in self.settings that are expressed as ranges.

        The only thing that is not checked by the schema is if the range is ascending, i.e. if the
        first value is smaller than or equal to the second value.

        Raises:
            ValueError: If any range violates if min_val > max_val. This prevents
                generating nonsensical parameters later in the pipeline.
        """
        for key, value in self.settings.items():
            if key.endswith("range"):
                min_val, max_val = value
                if min_val > max_val:
                    raise ValueError(f"Invalid range of values for {key}: {value}")

    def generate(self):
        """Generate random causal contextuality scenarios and validate them.

        This is the main generator wrapper that handles I/O for the generator.

        Returns:
            Generator object containing all generated scenarios if output_dir is None. If output_dir
            is not None, the scenarios will be written by iterating over the generator object,
            meaning that the returned generator will be empty.
        """
        ccs_generator = self._ccs_generator()

        def _get_path():
            return (
                Path(self.output_dir) / f"part_{self._batch_number:0{self.n_file_magnitude}}.jsonl"
            )

        if self.output_dir is not None:
            json_lines = self.batch_size > 1
            if json_lines:
                while True:
                    i = 0
                    path = _get_path()
                    is_open = False
                    try:
                        while i < self.batch_size:
                            ccs = next(ccs_generator)
                            f = path.open("a")
                            is_open = True
                            json.dump(ccs.data, f)
                            f.write("\n")
                            i += 1
                    except StopIteration:
                        # TODO: if we somehow do not get StopIteration, we will loop forever
                        # -- Could this happen?
                        break  # we reached the end of the generator: break out of outer loop
                    finally:
                        if is_open:
                            f.close()  # make sure to always close the file
                    self._batch_number += 1
            else:
                for ccs in ccs_generator:
                    path = _get_path()
                    ccs.to_json(path)
                    self._batch_number += 1
        # NOTE: the generator will be empty if output_dir is not None
        return ccs_generator

    def _ccs_generator(self):
        """Yield randomly generated causal contextuality scenarios.

        It repeatedly samples measurement sets and outcome ranges, constructs a random acyclic
        causal structure (enabling relations), samples one or more covers for that causal structure,
        instantiates a CausalContextualityScenario validates the resulting CCS and collects
        successful scenarios.

        Algorithm overview:
            1. Sample measurements and allowed outcomes via sample_measurements_and_outcomes().
            2. Create enabling relations for each measurement via generate_enabling_relations().
            3. For each causal structure, sample n_contexts_per_causal_structure covers using
               sample_contexts().
            4. Build a scenario dict with keys 'ms' (measurements metadata) and 'c' (contexts).
            5. Instantiate qes.CausalContextualityScenario(scenario) and validate by calling
               ccs.everything(). Only validated scenarios are appended to self.scenarios.
            6. When self.output_dir is set and keep an open file for appending the scenarios until
               self.batch_size is reached (then we open a new file) or we reach the end of the
               generator.
            7. Return the generator object, which will be empty if output_dir is set.
        """
        i = 0
        while i < self.n_scenarios:
            # 1) determine how many measurements and outcomes per measurement and which values
            measurements, outcomes = self.sample_measurements_and_outcomes()
            # 2) randomly create causal structure
            enabling_relations_dict = self.generate_enabling_relations(measurements, outcomes)

            for _ in range(self.n_contexts_per_causal_structure):
                # 3) randomly sample a number of subsets of the contexts and union must equal cover
                contexts = self.sample_contexts(measurements)
                cover = utils.create_anti_chain(contexts)

                # 4) construct the scenario dict
                scenario = {
                    "ms": [
                        {
                            "m": measurement,
                            "e": enabling_relations_dict[measurement],
                            "o": [
                                {
                                    "v": v,
                                    # calclate leaf below, in 5)
                                }
                                for v in outcomes[measurement]
                            ],
                            # calculate the memberships below, in 5)
                        }
                        for measurement in measurements
                    ],
                    "c": cover,
                }

                # 5) instantiate CCS and validate
                ccs = qes.CausalContextualityScenario(scenario)
                if ccs.everything():
                    # yield the validated scenario immediately
                    yield ccs

                    i += 1
                    if i >= self.n_scenarios:
                        # TODO: could just return here probably
                        break

    def _generate_measurement_names(self, n):
        """Create an ordered of measurement names of length n following a spreadsheet-like scheme.

        Names are generated lexicographically from the uppercase Latin alphabet:
        A, B, ..., Z, AA, AB, ..., AZ, BA, ..., etc.

        Args:
            n (int): Desired number of measurement names. If n <= 0 an empty list is returned.

        Returns:
            list[str]: A list of string names of length n (or fewer if n is truncated).
        """
        if n <= 0:
            return []
        names = []
        alphabet = string.ascii_uppercase
        width = 1
        while len(names) < n:
            for tup in self._tuple_letter_generator(alphabet, width):
                names.append("".join(tup))
                if len(names) >= n:
                    break
            width += 1
        # this can become very memory-intense if n is too large
        return names[:n]

    def _tuple_letter_generator(self, alphabet, width):
        """Yield lexicographic tuples of letters of a fixed width.

        This generator yields tuples such as ('A',), ('B',), ..., or for width=2 yields
        ('A','A'), ('A','B'), ... in lexicographic order. It is used by _generate_measurement_names
        to construct multi-letter measurement identifiers.

        Args:
            alphabet (str or Sequence[str]): Sequence of characters to use.
            width (int): The tuple width (number of character positions). Must be >= 1.

        Yields:
            tuple[str, ...]: Tuples of characters of length width in lexicographic order.
        """
        # TODO: this code may benefit from Numpy vector operations
        if width == 1:
            for a in alphabet:
                yield (a,)
            return
        pools = [alphabet] * width
        # start with first char for each position
        indices = [0] * width
        length = len(alphabet)
        while True:
            yield tuple(pools[i][indices[i]] for i in range(width))
            # increment last index
            i = width - 1
            while i >= 0:
                indices[i] += 1
                if indices[i] < length:
                    break
                indices[i] = 0
                i -= 1
            if i < 0:
                break

    def sample_contexts(self, measurements):
        """Sample a collection of contexts (list of measurement names) that forms a cover.

        The method probabilistically constructs a set of contexts such that every measurement
        appears in at least one sampled context (the procedure enforces coverage by initially
        sampling from the set of missing measurements). Additional contexts are sampled until the
        requested number of contexts is reached. Duplicate contexts are avoided.

        Behavior and options (read from self.settings):
            - 'n_contexts_range' (min_n_contexts, max_n_contexts): bounds on the target number of
              contexts.
            - 'context_size_range' (min_context_size, max_context_size): allowed sizes for
              individual contexts.
            - If the number of missing measurements requires it, the method will add more contexts
              than the nominal upper bound to ensure coverage (i.e., it prioritizes full coverage).
            - The last context is not guaranteed to be smaller or larger than max_context_size
              unless a special 'use_remaining_as_last' behavior is enabled externally.

        Args:
            measurements (Sequence[str]): sequence of measurement names to create a cover over.

        Returns:
            list[set[str]]: A list of contexts, each context is a frozenset of measurement names.

        Raises:
            RuntimeError: If no contexts could be sampled (should not be possible).

        Notes:
            - The function uses self.rng for all random choices to ensure reproducibility when the
              generator is seeded.
            - The returned contexts are produced from a set (to deduplicate) and then converted to
              lists in arbitrary order; order of contexts and order of elements inside each
              context is not guaranteed to match input order.
        """
        min_n_contexts, max_n_contexts = self.settings["n_contexts_range"]
        min_context_size, max_context_size = self.settings["context_size_range"]

        n_measurements = len(measurements)
        max_context_size = min(max_context_size, n_measurements)
        min_context_size = min(min_context_size, n_measurements)
        n_contexts = self.rng.randint(min_n_contexts, max_n_contexts)

        contexts_set = set()
        # ensure coverage by initially sampling dynamically from the set of missing measurements
        missing = set(measurements)
        while missing:
            size = self.rng.randint(min_context_size, max_context_size)
            chosen_missing = list(missing)
            self.rng.shuffle(chosen_missing)
            context = frozenset(chosen_missing[:size])
            # update the set of missing measurements
            missing -= context
            contexts_set.add(context)
        # if we have not yet created >= n_contexts: try sample some more
        # NOTE: for now we use a lazy approach, but perhaps it might be worth calculating the number
        # of possible contexts given the context_size_range and binomial coefficients, and then use
        # some heuristic based on this to determine whether it is worth trying to sample more
        for _ in range(max(0, n_contexts - len(contexts_set))):
            # NOTE: this means that the parameter n_contexts_range is approximative at best and
            # heavily dependent on the number of measurements and the allowed context sizes
            size = self.rng.randint(min_context_size, max_context_size)
            context = frozenset(self.rng.sample(measurements, size))
            contexts_set.add(context)

        if not contexts_set:
            # should not be able to happen
            raise RuntimeError("Failed to sample any contexts")

        return [context for context in contexts_set]

    def _weighted_count_sample(self, mean, min_k, max_k):
        """Sample an integer from {min_k, ..., max_k} with an exponentially decaying weight.

        This helper draws a value k in the integer interval [min_k, max_k] using unnormalized
        weights proportional to exp(-(k - min_k) / mean). Larger mean produces a heavier tail (more
        probability on larger k), while small mean concentrates the probability on the lower bound.

        Args:
            mean (float): Tail-control parameter; must be non-negative. Small positive values bias
                toward the smallest allowed k. Internally clamped to a small epsilon to avoid
                division by zero.
            min_k (int): Lower bound of the sampling interval (inclusive).
            max_k (int): Upper bound of the sampling interval (inclusive).

        Returns:
            int: A sampled integer in [min_k, max_k]. Guaranteed to return at least min_k.

        Notes:
            - If max_k <= min_k the function returns min_k immediately.


        It is actually not a true mean, the expected value is given by (n := max_k, m := mean)
            EV = [ sum_{k=1}^{n} k*e^{-(k-1) / m} ] / [ sum_{k=1}^{n} e^{-(k-1) / m} ] =
            = [ e^{(1 - n)/m - 1/m} * (-n*e^{1/m} + e^{n/m} + n - 1) ] /
                                                        / [ (e^{1/m} - 1) (1 - e^{-n/m}) ]
        """
        # TODO: vectorize with Numpy
        if max_k <= 1:
            return 1
        # compute unnormalized weights for k = 1,...,max_k
        mean = max(1e-8, float(mean))
        # weight proportional to exp(-(k-1) / mean)
        # if mean is small, we will most likely get k = 1
        weights = [math.exp(-(k - min_k) / mean) for k in range(min_k, max_k + 1)]
        total = sum(weights)
        # normalize
        probs = [w / total for w in weights]
        r = self.rng.random()
        total = 0.0
        for k, p in enumerate(probs, start=min_k):
            total += p
            if r <= total:
                return k
        return max_k

    def generate_enabling_relations(self, measurements, outcomes):
        """Construct a consistent, acyclic set of enabling relations for each measurement.

        An enabling relation for a target measurement is a (non-empty) set of events; each event is
        a pair (measurement m, allowed outcome v). A measurement may have zero or more alternative
        enabling relations (outer list). To maintain acyclicity, only measurements that appear
        before the target in a chosen order may be used as enablers; the order is either
        lexicographic (this is the default) or a random topological order sampled by shuffling.

        Behavior and parameters (read from self.settings):
            - 'p_has_enabled': probability that a target measurement has any enabling relations at
              all. With probability (1 - p_has_enabled) the target is enabled by the empty set.
            - 'n_alternatives_mean' and 'n_alternatives_range': control how many alternative
              enabling relations are sampled per target (outer list length).
            - 'enabling_relation_size_mean' and 'enabling_relation_size_range': control the number
              of events in each enabling relation (i.e., how many distinct enabler measurements
              participate).
            - If no_lexicographic_order is True the generator shuffles measurements to produce a
              random topological order; enablers are always chosen from measurements that occur
              earlier in that order.

        Args:
            measurements (Sequence[str]): Ordered sequence of measurement names (the order is
                shuffled internally when no_lexicographic_order is True).
            outcomes (dict[str, Sequence[int]]): Mapping from measurement name to list/sequence of
                allowed outcome values (integers). Values are sampled uniformly when creating
                events.

        Returns:
            dict[str, list[list[dict]]]: A mapping from each measurement name to a list of
                alternative enabling relations. Each enabling relation is represented as a list of
                event dicts with keys:
                    - 'm' (str): enabler measurement name
                    - 'v' (int): chosen outcome value for that enabler
                If a measurement has no enabling relations the returned value is an empty list
                (meaning it is enabled by the empty set).

        Notes:
            - Duplicate enabling relations (same multiset of (m,v) pairs) are avoided.
            - If no earlier measurements exist in the imposed order, the target has no enablers.
        """
        p_has_enabled = self.settings["p_has_enabled"]
        n_alternatives_mean = self.settings["n_alternatives_mean"]
        enabling_relation_size_mean = self.settings["enabling_relation_size_mean"]
        min_alternatives, max_alternatives = self.settings["n_alternatives_range"]
        min_relation_size, max_relation_size = self.settings["enabling_relation_size_range"]
        no_lexicographic_order = self.settings["no_lexicographic_order"]

        # to ensure an acyclic causal contextuality scenario, enforce an order
        # x can enable y if x appears before y in the order
        order = list(measurements)
        if no_lexicographic_order:
            # random topological ordering
            self.rng.shuffle(order)

        # map measurement to position in order
        pos = {m: i for i, m in enumerate(order)}

        # helper to determine allowed measurements for target's enabling relations
        def allowed_enabling_measurements(target):
            target_pos = pos[target]
            # allowed if position < target position
            allowed = [m for m in measurements if pos[m] < target_pos]
            return allowed

        enabled_by = dict()

        for target in measurements:
            enabled_by[target] = []
            # decide whether target has any enabling relations at all
            if self.rng.random() >= p_has_enabled:
                continue  # enabled by default/empty set

            allowed_enablers = allowed_enabling_measurements(target)
            # if the target is first in the imposed order, then nothing can enable it
            if not allowed_enablers:
                continue

            # number of alternative enabling relations for this target measurement
            n_alts = self._weighted_count_sample(
                n_alternatives_mean, min_alternatives, max_alternatives
            )

            alt_list = []
            used_enabling_relations = set()

            for _ in range(n_alts):
                # sample enabling_relation size (at least 1)
                max_size = min(max_relation_size, len(allowed_enablers))
                min_size = min(min_relation_size, max_size)
                if max_size <= 0:
                    break
                enabling_relation_size = self._weighted_count_sample(
                    enabling_relation_size_mean, min_size, max_size
                )

                # choose distinct enabler measurements for this enabling_relation
                # if enabling_relation_size > len(allowed_enablers) it will be limited above
                enablers = self.rng.sample(allowed_enablers, enabling_relation_size)

                # for each enabling measurement, pick a value uniformly from its outcomes
                events = []
                seen_measurements = set()
                for m in enablers:
                    vals = outcomes.get(m)
                    if m in seen_measurements or not vals:
                        # skip if the measurement is already present in the enabling relation OR
                        # if the measurement does not have any outcomes (but this should not happen)
                        continue
                    v = self.rng.choice(list(vals))
                    seen_measurements.add(m)
                    events.append({"m": m, "v": int(v)})

                if not events:
                    continue

                enabling_relation = frozenset((e["m"], e["v"]) for e in events)
                # skip duplicate enabling relations.
                if enabling_relation in used_enabling_relations:
                    continue
                used_enabling_relations.add(enabling_relation)
                alt_list.append(events)

            # if alt_list is empty, then target does not have any enabling relations
            # this means that we can simply assign target's enabling relations to alt_list
            enabled_by[target] = alt_list

        return enabled_by

    def sample_measurements_and_outcomes(self):
        """Determine the list of measurements and the allowed outcomes for each measurement.

        Two modes:
            1. If self.measurement_outcomes_dict is provided, that mapping is used directly: the
               measurement list is list(measurement_outcomes_dict.keys()), and outcomes are taken
               from the mapping (no sampling).
            2. If measurement_outcomes_dict is None, the function samples:
                - n_measurements uniformly from n_measurements_range,
                - generates measurement names using _generate_measurement_names,
                - for each measurement samples a number of outcomes k uniformly from n_values_range
                  and uses the integers 0..k-1 as outcomes.

        Returns:
            tuple[list[str], dict[str, list[int]]]: A pair (measurements, outcomes) where
                measurements is an ordered list of measurement names and outcomes maps each
                measurement to a list of integer outcome values.
        """
        # TODO: add check about anti-chain and also read about it
        # TODO: allow fixed measurements, random outcomes; random measurements, fixed outcomes
        # currently, both are either are random, or both fixed if measurement_outcomes_dict != None
        min_measurements, max_measurements = self.settings["n_measurements_range"]
        min_vals, max_vals = self.settings["n_values_range"]

        if self.measurement_outcomes_dict is None:
            n_measurements = self.rng.randint(min_measurements, max_measurements)
            measurements = self._generate_measurement_names(n_measurements)
            outcomes = dict()
            for m in measurements:
                k = self.rng.randint(min_vals, max_vals)
                # outcomes are 0,...,k-1
                outcomes[m] = list(range(k))
        else:
            measurements = list(self.measurement_outcomes_dict.keys())
            outcomes = self.measurement_outcomes_dict

        return measurements, outcomes

    # TODO: expose this as standalone from the rest of the class too, so that you can essentially
    # use it as a util function to find the causally secured cover
    def _merge_requirements(self, left, right):
        """Merge two required-value maps.

        Args:
            left (dict[str, int]): Required values for some measurements.
            right (dict[str, int]): Required values for some measurements.

        Returns:
            dict[str, int] | None: The merged map, or None if there is a conflict.
        """
        merged = dict(left)
        for m, v in right.items():
            if m in merged and merged[m] != v:
                # conflicting values
                return None
            merged[m] = v
        return merged

    def _generate_local_cover(self, measurements, iterations=100):
        """Generate a local cover using, essentially, a Markov Chain Monte Carlo approach.

        We start with the cover of all singletons, and then we either add a random subset of the
        measurement set as a context, alter a measurement in one of the existing contexts, or split
        a context into two parts -- these happen with roughly equal probability (0.3, 0.3 and 0.4,
        respectively). This does not sample uniformly from the set of all local covers, but it
        should be able to reach every possible local cover. The mixing will improve with more
        iterations, but this will take longer to run (of course).
        """
        measurement_set = set(measurements)
        # start with all singletons
        current_cover = [frozenset([m]) for m in measurements]

        def cleanup(cover):
            """Ensure the cover is an anti-chain (maximal) and a cover."""
            # 1) remove non-maximal sets
            maximal = []
            sorted_cover = sorted(list(cover), key=len, reverse=True)
            for s in sorted_cover:
                if not any(s <= other for other in maximal):
                    maximal.append(s)

            # 2) ensure all measurements are covered
            current_union = set().union(*maximal)
            missing = measurement_set - current_union
            for m in missing:
                maximal.append(frozenset([m]))
            return maximal

        for _ in range(iterations):
            move_type = self.rng.random()
            new_cover = list(current_cover)

            if move_type < 0.3 or not new_cover:
                # add a random subset from power set
                n = self.rng.randint(1, len(measurements))
                s = frozenset(self.rng.sample(measurements, n))
                new_cover.append(s)

            elif move_type < 0.6:
                # mutate an existing context
                idx = self.rng.randrange(len(new_cover))
                m = self.rng.choice(measurements)
                target = set(new_cover[idx])
                if m in target and len(target) > 1:
                    target.remove(m)
                else:
                    target.add(m)
                new_cover[idx] = frozenset(target)

            else:
                # split a context (avoids getting stuck in the trivial all measurement cover)
                idx = self.rng.randrange(len(new_cover))
                target = list(new_cover[idx])
                if len(target) > 1:
                    self.rng.shuffle(target)
                    split_point = self.rng.randint(1, len(target) - 1)
                    c1 = frozenset(target[:split_point])
                    c2 = frozenset(target[split_point:])
                    new_cover.pop(idx)
                    new_cover.extend([c1, c2])

            current_cover = cleanup(new_cover)

        return sorted([sorted(list(c)) for c in current_cover])

    def sample_local_cover(self, measurements):
        # TODO: how should the number of iterations be handled?
        if len(measurements) <= len(LOCAL_COVERS):
            local_covers = LOCAL_COVERS[len(measurements) - 1]
            idx_to_measurement = {i: m for i, m in enumerate(measurements)}
            base_cover = self.rng.choice(local_covers)
            # rename all the measurements in the base cover to the measurements specified
            # NOTE: measurements are represented by indices in the statically stored cover, so we
            # can immediately use them to convert to the real measurements
            cover = [[idx_to_measurement[i] for i in context] for context in base_cover]
            return cover
        return self._generate_local_cover(measurements)

    def generate_causally_secured_cover(
        self,
        measurements,
        enabling_relations_dict,
        allow_unclean_local_covers=False,
        debug=False,
        max_partition_tries=100,
    ):
        """Generate a causally secured cover using random local covers.

        This method assumes unique causal bridges:
        each measurement has either zero or one enabling relation.

        The procedure is:

        1. Compact the enabling relations into unique bridges.
        2. Compute transitive causal closures of all measurements.
        3. Group measurements by identical enabling relation (same LHS).
        4. For each group, sample a random partition of its RHS into a local cover.
        5. Turn each local block into a candidate global context by adding its causal
           support (transitive closure).
        6. Remove non-maximal contexts so the result is an antichain.

        Args:
            measurements (Sequence[str]): Measurement names.
            enabling_relations_dict (dict[str, list[list[dict]]]): Mapping from each
                measurement to its enabling alternatives. Under the unique-bridge
                assumption, each list must have length 0 or 1.
            allow_unclean_local_covers (bool): If False, reject and resample local
                partitions that are internally inconsistent. If True, keep them and
                split incompatible blocks into singletons when needed.
            debug (bool): Print compacted relations, closures, sampled local covers,
                and merge progress.
            max_partition_tries (int): Max number of attempts to sample a clean local
                partition for each enabling group.

        Returns:
            list[list[str]]: A causally secured cover as a list of contexts.

        Raises:
            ValueError: If the enabling relations are not unique-bridge, or if the
                enabling structure is inconsistent.
        """

        def canonical_relation(relation):
            return tuple(sorted(tuple(event.values()) for event in relation))

        # 1) enforce unique bridges and compact them
        unique_bridge = dict()
        for m in measurements:
            rels = enabling_relations_dict.get(m, [])
            if len(rels) > 1:
                raise ValueError(
                    f"Measurement {m!r} has multiple enabling relations, "
                    "but this generator assumes unique causal bridges."
                )
            unique_bridge[m] = rels[0] if rels else []

        # 2) compute transitive closures of required events
        # closure_req[m] = dict(enabler_measurement -> required_value)
        closure_req = dict()
        visiting = set()

        def closure_of(m):
            if m in closure_req:
                return closure_req[m]
            if m in visiting:
                raise ValueError(f"Cyclic enabling relation detected at measurement {m!r}.")
            visiting.add(m)

            req = dict()
            for event in unique_bridge[m]:
                enabler, value = event.values()

                parent_req = closure_of(enabler)
                merged = self._merge_requirements(req, parent_req)
                if merged is None:
                    raise ValueError(
                        "Inconsistent enabling relations detected while closing "
                        f"{m!r}: the transitive closure already requires a conflicting event."
                    )
                req = merged

                if enabler in req and req[enabler] != value:
                    raise ValueError(
                        "Inconsistent enabling relations detected: "
                        f"{m!r} requires {enabler}={value}, but the transitive closure "
                        f"forces {enabler}={req[enabler]}."
                    )
                req[enabler] = value

            visiting.remove(m)
            closure_req[m] = req
            return req

        for m in measurements:
            closure_of(m)

        if debug:
            print("\n[CCS] Compact enabling relations / unique bridges:")
            for m in measurements:
                rel = canonical_relation(unique_bridge[m])
                print(f"  {m}: {rel if rel else '∅'}")

            print("\n[CCS] Transitive closures:")
            for m in measurements:
                req = closure_req[m]
                if req:
                    pretty = ", ".join(f"{k}={v}" for k, v in sorted(req.items()))
                    print(f"  τ̄({m}) = {{{pretty}}}")
                else:
                    print(f"  τ̄({m}) = ∅")

        # 3) group RHS measurements by identical lhs enabling relation
        groups = dict()
        for m in measurements:
            lhs_key = canonical_relation(unique_bridge[m])
            groups.setdefault(lhs_key, []).append(m)

        if debug:
            print("\n[CCS] Compacted enabling groups (same lhs -> rhs measurements):")
            for lhs_key, rhs in groups.items():
                lhs_str = lhs_key if lhs_key else "∅"
                print(f"  lhs={lhs_str}  => rhs={sorted(rhs)}")

        # 4) sample a random local cover for each group
        def block_is_clean(block):
            """A block is clean if all measurements in it can coexist."""
            req = dict()
            for m in block:
                merged = self._merge_requirements(req, closure_req[m])
                if merged is None:
                    return False
                req = merged
            return True

        # TODO: handle the cleanliness of the local covers in the generation in some way
        local_covers = dict()
        for lhs_key, rhs in groups.items():
            rhs = list(rhs)
            # TODO: can this happen, or would it be okay with groups being a defaultdict?
            if not rhs:
                local_covers[lhs_key] = []
                continue

            sampled = None
            # FIXME: remove the max_tries logic and force it to succeed every time in some way
            for _ in range(max_partition_tries):
                candidate = self.sample_local_cover(rhs)
                if allow_unclean_local_covers or all(block_is_clean(block) for block in candidate):
                    sampled = candidate
                    break

            if sampled is None:
                raise ValueError(
                    "Failed to sample a clean local cover for enabling group "
                    f"{lhs_key if lhs_key else '∅'} after {max_partition_tries} tries."
                )

            local_covers[lhs_key] = sampled

        if debug:
            print("\n[CCS] Sampled local covers:")
            for lhs_key, blocks in local_covers.items():
                lhs_str = lhs_key if lhs_key else "∅"
                print(f"  lhs={lhs_str}")
                for block in blocks:
                    print(f"    {sorted(block)}")

        # 5) convert each local block into a candidate global context
        candidate_contexts = []
        for lhs_key, blocks in local_covers.items():
            for block in blocks:
                req = dict()
                consistent = True
                for m in block:
                    merged = self._merge_requirements(req, closure_req[m])
                    if merged is None:
                        consistent = False
                        break
                    req = merged

                if not consistent:
                    if allow_unclean_local_covers:
                        # fall back to singleton contexts if the block is not clean
                        for m in block:
                            singleton_req = closure_req[m]
                            ctx = set([m]) | set(singleton_req.keys())
                            candidate_contexts.append(
                                {
                                    "meas": frozenset(ctx),
                                    "req": dict(singleton_req),
                                    "origin": lhs_key,
                                }
                            )
                        continue

                    raise ValueError(
                        "A sampled local block is internally inconsistent. "
                        "Set allow_unclean_local_covers=True to relax this."
                    )

                ctx = set(block) | set(req.keys())
                candidate_contexts.append(
                    {
                        "meas": frozenset(ctx),
                        "req": req,
                        "origin": lhs_key,
                    }
                )

        if debug:
            print("\n[CCS] Initial candidate contexts:")
            for ctx in candidate_contexts:
                print(f"  {sorted(ctx['meas'])}")

        # 6) remove non-maximal contexts (enforce antichain)
        candidate_contexts.sort(key=lambda c: len(c["meas"]), reverse=True)
        cover = []
        for ctx in candidate_contexts:
            meas = ctx["meas"]
            if any(meas < other["meas"] for other in cover):
                continue
            cover = [other for other in cover if not other["meas"] < meas]
            cover.append(ctx)

        # final coverage check
        covered = set()
        for ctx in cover:
            covered.update(ctx["meas"])

        missing = [m for m in measurements if m not in covered]
        if missing:
            if not allow_unclean_local_covers:
                raise ValueError(f"The generated cover does not cover all measurements: {missing}")
            for m in missing:
                ctx_meas = frozenset({m} | set(closure_req[m].keys()))
                cover.append({"meas": ctx_meas, "req": dict(closure_req[m])})

        final_cover = []
        for ctx in cover:
            final_cover.append(sorted(ctx["meas"]))

        if debug:
            print("\n[CCS] Final causally secured cover:")
            for ctx in final_cover:
                print(f"  {ctx}")

        return final_cover
