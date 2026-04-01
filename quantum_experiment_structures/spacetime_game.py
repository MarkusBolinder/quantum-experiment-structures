"""Python representation of a spacetime game."""

from collections import Counter, defaultdict
import inspect
import itertools
import json
from pathlib import Path

from quantum_experiment_structures.data.schemas import SPACETIME_GAME_SCHEMA
import jsonschema


class SpacetimeGame:
    """Python representation of a spacetime game.

    In addition to just representing the data, a number of methods are supplied which can be
    leveraged to ensure validity and/or calculate properties of the spacetime game.
    """

    def __init__(self, json_data):
        """Read, validate and initialize an instance of a spacetime game.

        Args:
            json_data: a dict-object representing well-formed JSON.

        Raises:
            NotImplementedError: If actions are not strings.
        """
        self.data = json_data

        # Extract base properties for easy access
        self.players = set(self.data["ps"])
        self.actions = set(self.data["as"])

        # Check that actions are strings
        for action in self.data["as"]:
            if not isinstance(action, str):
                raise NotImplementedError("Only string actions are currently supported.")

        # Extract information sets and build a registry of nodes
        # TODO: make the action list a set in the values to save time later with O(1) lookup
        self.info_sets = {iset["i"]: iset for iset in self.data["is"]}

        self.nodes = {}
        for iset in self.data["is"]:
            for node in iset["ns"]:
                self.nodes[node["n"]] = {
                    "node_data": node,
                    "info_set_id": iset["i"],
                    "player": iset["p"],
                }

        # build adjacency structure for forward traversal (parents -> children)
        # NOTE: this will add create adjacency lists for parents that are not listed as nodes,
        # but this will be checked by one of the checking methods anyway, so it should be fine
        self.adj = defaultdict(list)
        for node_name, info in self.nodes.items():
            for p in info["node_data"]["ps"]:
                parent, action = p.values()
                self.adj[parent].append({"c": node_name, "a": action})

    def __repr__(self):
        """Return a string representation of the spacetime game."""
        if "h" in self.data:
            return "\n".join(f"{k:6s}: {v}" for k, v in self.data["h"].items())
        return str(self.data)

    def validate(self):
        """Validate the data using the JSON Schema for spacetime games."""
        try:
            jsonschema.validate(self.data, schema=SPACETIME_GAME_SCHEMA)
        except jsonschema.ValidationError as e:
            print(f"Validation Error: {e}")
            return False
        return True

    def check_information_sets_consistency(self):
        """Verify that players and actions in information sets are the same as top-level.

        Checks the players and actions listed in all information sets and determines whether the
        union of all players and actions in the information sets are the same as the player and
        action arrays in the top-level of the data.

        Raises:
            ValueError: If the players or actions in the information sets do not match top-level.
        """
        found_players = set()
        found_actions = set()
        for iset in self.data["is"]:
            found_players.add(iset["p"])
            found_actions.update(action for action in iset["a"])

        if found_players != self.players:
            raise ValueError(
                "Union of players in information set does not match top-level player array"
                f"Expected {self.players}, but got {found_players}"
            )
        if found_actions != self.actions:
            raise ValueError(
                "Union of actions in information set does not match top-level action array"
                f"Expected {self.actions}, but got {found_actions}"
            )
        return True

    def check_node_graph_integrity(self):
        """Verify the causal links between nodes in the spacetime game.

        Iterates through all nodes to ensure that any referenced parent or child nodes
        are present in the graph, and that the actions connecting them are registered
        in the corresponding information set's action list. Also checks for self-references.

        Raises:
            ValueError: If a node references an unknown parent, child, action, or itself.
        """
        for name, node_info in self.nodes.items():
            node = node_info["node_data"]
            info_set = self.info_sets[node_info["info_set_id"]]
            # check parents
            for p in node["ps"]:
                parent, action = p.values()
                parent_node = self.nodes.get(parent)
                if parent_node is not None:
                    parent_info_set_id = parent_node["info_set_id"]
                if (
                    parent == name
                    or parent is None
                    or action not in self.info_sets[parent_info_set_id]["a"]
                ):
                    raise ValueError(
                        f"Parental problems for node '{name}' from parent '{parent}' with "
                        f"action {action}. Parent in nodes: {parent_node is None}. "
                        "(Expected True.) Action in parent node's information set: "
                        f"{action in info_set['a']}. (Expected True.)"
                    )
        return True

    def check_no_cycles(self):
        """Ensure there are no cycles in the node graph.

        Returns:
            bool: True if no cycles are detected.

        Raises:
            ValueError: If a cycle is detected.
        """
        visited = set()
        stack = set()

        def has_cycle(node):
            visited.add(node)
            stack.add(node)

            for child in self.adj[node]:
                child_name = child["c"]
                if child_name not in visited:
                    if has_cycle(child_name):
                        return True
                elif child_name in stack:
                    return True

            stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Cycle detected in node graph involving node '{node}'.")
        return True

    # TODO: check that if a measurement is present in the history, then all its enabling
    # measurements should be present and the converse, that if a group of measurements that enable
    # something is present, then the something is also present (totality).
    def check_histories_consistency(self):
        """Verify that assignments and utilities in histories reference valid entities.

        Checks all histories to ensure:
        1. Actions are playable in the specified information sets.
        2. Utility (u) lists all players in the game.

        Raises:
            ValueError: If history logic is broken or players are missing from payoffs.
        """
        # TODO: check totality of histories too.
        for history in self.data.get("z", []):
            # check that actions are playable in the information set
            info_set_counter = Counter()
            for assignment in history["h"]:
                iset_id = assignment["i"]
                info_set_counter.update([iset_id])
                action = assignment["a"]

                if iset_id not in self.info_sets:
                    raise ValueError(
                        f"History '{history['z']}' references unknown information set '{iset_id}'."
                    )

                if action not in self.info_sets[iset_id]["a"]:
                    raise ValueError(
                        f"Action '{action}' is not playable in information set '{iset_id}' "
                        f"for history '{history['z']}'."
                    )
            # make sure every information set is only listed once in the history
            for k, v in info_set_counter.items():
                if v > 1:
                    raise ValueError(
                        f"Information set '{k}' assigned more than one action in history {history}."
                    )
            if "s" in history:
                assigned_info_sets = set(info_set_counter.keys())
                history_info_sets = set(history["s"])
                if assigned_info_sets != history_info_sets:
                    raise ValueError(
                        "Information sets activated by assignments does not match the information "
                        "sets the history lists as activated. Assigned information sets: "
                        f"{assigned_info_sets}; {history['s']=}."
                    )

            # check completeness of utility
            players_in_utility = set(payoff["p"] for payoff in history["u"])
            if players_in_utility != self.players:
                missing = self.players - players_in_utility
                raise ValueError(
                    f"Utility for history '{history['z']}' is missing players: {missing}."
                )
        return True

    def check_strategies_consistency(self):
        """Verify strategy validity, player matching, and uniqueness.

        Checks:
        1. Player associated with strategy group matches player of the info sets used.
        2. No duplicate strategies (as sets of assignments) exist for a player.

        Raises:
            ValueError: If strategy/player mismatch or duplicate strategies found.
        """
        # TODO: check totality of strategies too. I.e. that all possible strategies are included.
        for strategy_group in self.data.get("s", []):
            player = strategy_group["p"]
            if player not in self.players:
                raise ValueError(f"Strategy lists an unknown player '{player}'.")

            seen_strategies = set()
            for strategy in strategy_group["s"]:
                info_set_counter = Counter()
                # check player consistency with info set
                for assignment in strategy:
                    iset_id, action = assignment.values()
                    info_set_counter.update([iset_id])
                    iset = self.info_sets.get(iset_id)
                    if iset is None:
                        raise ValueError(
                            f"Strategy for '{player}' references unknown info set '{iset_id}'."
                        )

                    if iset["p"] != player:
                        raise ValueError(
                            f"Strategy for player '{player}' contains info set '{iset_id}' "
                            f"belonging to player '{iset['p']}'."
                        )

                    if action not in iset["a"]:
                        raise ValueError(
                            f"Action '{action}' not playable in information set '{iset_id}'."
                        )
                # make sure every information set is only listed once in the strategy
                for k, v in info_set_counter.items():
                    if v > 1:
                        raise ValueError(
                            f"Information set '{k}' assigned more than one action in strategy "
                            f"{strategy}. {v} assignments found."
                        )

                # check for duplicate strategies, e.g. [[A,B],[B,A]]
                strategy_set = frozenset(tuple(a.values()) for a in strategy)
                if strategy_set in seen_strategies:
                    raise ValueError(f"Duplicate strategy found for player '{player}': {strategy}")
                seen_strategies.add(strategy_set)
        return True

    def check_number_of_strategies(self):
        """Verify that the amount of strategies for each player is correct.

        Returns:
            bool: False if there are more than one strategy groups associated with any one player or
                the union of the players with stratgies is not the set of all players.
        """
        if "s" not in self.data:
            return True
        strategies_per_player = Counter(strategy_group["p"] for strategy_group in self.data["s"])
        players_with_strategies = set(strategies_per_player.keys())
        # only one strategy group per player
        if any(n > 1 for n in strategies_per_player.values()):
            return False
        # all players should have strategies
        if players_with_strategies != self.players:
            return False
        return True

    def add_played_information_sets(self):
        """Populate each history with the information sets activated in that history."""
        for history in self.data.get("z", []):
            # skip if the field already exists
            if "s" in history:
                continue
            history["s"] = [assignment["i"] for assignment in history["h"]]

    def add_histories(self):
        """Add missing complete histories by traversing the DAG.

        Use DFS to recursively build up all possible complete histories of the spacetime game. Only
        adds the history to the data if it is not already present -- this is based on the
        information set and action tuples that define a history.
        """
        # FIXME: these does not seem to produce correct histories when there are enabling relations
        # with more than one event, e.g. with {(X,0),(Z,0)} enables T, then the histories that
        # contain T, only contain one of X or Z, not both.
        if "z" not in self.data:
            self.data["z"] = []

        roots = [n for n, info in self.nodes.items() if not info["node_data"]["ps"]]

        def get_content(h):
            return frozenset(tuple(item.values()) for item in h)

        existing = set(get_content(z["h"]) for z in self.data["z"])
        found = []

        # dfs over the dag to find all paths
        def find_paths(node_name, current_h):
            info = self.nodes[node_name]
            iset_id = info["info_set_id"]
            for action in self.info_sets[iset_id]["a"]:
                new_h = current_h + [{"i": iset_id, "a": action}]
                children = [c["c"] for c in self.adj[node_name] if c["a"] == action]
                if not children:
                    # found a leaf in the tree: append the history that got here
                    found.append(new_h)
                else:
                    for child in children:
                        find_paths(child, new_h)

        for root in roots:
            find_paths(root, [])
        for history in found:
            # ignore histories that are already present
            # TODO: optimize this, because we are still calculating the history
            if get_content(history) not in existing:
                isets_in_history = [assignment["i"] for assignment in history]
                history_id = "z_" + "".join(assignment["a"] for assignment in history)
                self.data["z"].append(
                    {
                        "z": history_id,
                        "h": history,
                        "s": isets_in_history,
                        # creating new utility lists avoids referencing the same object again
                        "u": [{"p": p, "v": 0} for p in self.players],
                    }
                )

    def add_strategies(self):
        """Add all missing full (non-reduced) strategies for every player.

        Populates the data with predefined actions for each player in each information set,
        corresponding to all their possible strategies. This method does not create the reduced
        strategies, but instead assigns actions to all information sets -- even though they may not
        be played/reached in the spacetime game given the previous actions taken.
        """
        # TODO: check that this method produces correct results
        if "s" not in self.data:
            self.data["s"] = []

        # map players to all information sets they are in
        player_isets = {
            p: [i for i, data in self.info_sets.items() if data["p"] == p] for p in self.players
        }

        for player, isets in player_isets.items():
            # find first strategy that is associated with 'player'
            group = next(
                (
                    strategy_group
                    for strategy_group in self.data["s"]
                    if strategy_group["p"] == player
                ),
                None,
            )
            if not group:
                group = {"p": player, "s": []}
                self.data["s"].append(group)

            # TODO: can the check for existing strategies be more memory efficient
            existing = set(frozenset(tuple(a.values()) for a in s) for s in group["s"])
            action_lists = [self.info_sets[i]["a"] for i in isets]
            for combo in itertools.product(*action_lists):
                strategy = [{"i": i, "a": a} for i, a in zip(isets, combo)]
                if frozenset(tuple(a.values()) for a in strategy) not in existing:
                    group["s"].append(strategy)

    def add_human_readable(self):
        """Add a human readable representation of the spacetime game to the data.

        This method constructs strings for the components of the spacetime game
        tuple (N, R, P, A, rho, chi, sigma, I, Z, u) and stores them in self.data["h"].

        There are no checks to ensure that the human readable format correctly describes
        the game. However, the schema enforces (light) constraints on the format of the
        human readable part.
        """
        # 1. P and A: sets of players and actions
        ps_representation = "{" + ", ".join(sorted(self.players)) + "}"
        as_representation = "{" + ", ".join(sorted(self.actions)) + "}"

        # 2. N: set of decision nodes
        all_node_names = sorted(self.nodes.keys())
        ns_representation = "{" + ", ".join(all_node_names) + "}"

        # 3. R and σ: edges and edge action labeling
        # σ(N,M) represents the action label on the directed edge from N to M
        sigma_labels = []
        for node_name in all_node_names:
            for child in self.adj[node_name]:
                sigma_labels.append(f"σ({node_name}, {child['c']}) = {child['a']}")
        es_representation = ", ".join(sigma_labels) if sigma_labels else "∅"

        # 4. I, ρ, χ: information sets, player labeling, and action labeling
        # each information set i in I must be compatible with ρ and χ (same player and action set)
        is_list = []
        for i_id in sorted(self.info_sets.keys()):
            iset = self.info_sets[i_id]
            node_set = "{" + ", ".join(sorted(n["n"] for n in iset["ns"])) + "}"
            player = iset["p"]
            action_set = "{" + ", ".join(sorted(iset["a"])) + "}"
            is_list.append(f"{i_id} = {node_set} [ρ:{player}, χ:{action_set}]")
        is_representation = ", ".join(is_list)

        # 5. Z: complete histories, represented as sets of (information set, action) assignments
        z_list = []
        for history in self.data.get("z", []):
            assignments = "{" + ", ".join(f"({a['i']}, {a['a']})" for a in history["h"]) + "}"
            z_list.append(f"{history['z']} = {assignments}")
        z_representation = ", ".join(z_list) if z_list else "∅"

        # 6. u: utility functions -- mapping outcomes to payoffs for each player
        u_list = []
        for history in self.data.get("z", []):
            z_id = history["z"]
            for payoff in history.get("u", []):
                u_list.append(f"u_{payoff['p']}({z_id}) = {payoff['v']}")
        u_representation = ", ".join(u_list) if u_list else "∅"

        # 7. s: strategies, represented as sets of histories for each player
        s_list = []
        for strategy_group in self.data.get("s", []):
            player = strategy_group["p"]
            strategies = []
            for strategy in strategy_group["s"]:
                assignments = "{" + ", ".join(f"({a['i']}, {a['a']})" for a in strategy) + "}"
                strategies.append(assignments)
            s_list.append(f"S_{player} = {{{', '.join(strategies)}}}")
        s_representation = ", ".join(s_list) if s_list else "∅"

        self.data["h"] = {
            "ns": ns_representation,
            "es": es_representation,
            "ps": ps_representation,
            "as": as_representation,
            "is": is_representation,
            "z": z_representation,
            "u": u_representation,
            "s": s_representation,
        }

    def all_checks(self):
        """Perform all checks."""
        for name, member in inspect.getmembers(self):
            if inspect.ismethod(member) and name.startswith("check"):
                ok = member()
                if not ok:
                    raise ValueError(f"Inconsistency detected: {name} failed.")
        return True

    def all_adds(self):
        """Add everything that can be added based on the base CCS data."""
        for name, member in inspect.getmembers(self):
            if inspect.ismethod(member) and name.startswith("add"):
                member()

    def to_json(self, filename, indent=None):
        """Flush data to a JSON file."""
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".json")
        with path.open("w") as f:
            json.dump(self.data, f, indent=indent)

    def append_to_json_lines(self, filename):
        """Append the spacetime game data to a JSON Lines file."""
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".jsonl")
        with path.open("a") as f:
            json.dump(self.data, f)
            f.write("\n")

    def everything(self):
        # first validate against schema
        if not self.validate():
            raise jsonschema.ValidationError("The data is not valid against the schema.")
        # then add missing fields
        # FIXME: if histories/strategies get added after the add_human_readable method is called,
        # then they will not be properly added to the human readable object
        self.all_adds()
        # then check that everything is correct
        self.all_checks()
        return True
