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
        for node_name, node_info in self.nodes.items():
            for p in node_info["node_data"]["ps"]:
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
            # check parents
            for p in node["ps"]:
                parent, action = p.values()
                parent_node = self.nodes.get(parent)
                if parent_node is not None:
                    parent_info_set = self.info_sets[parent_node["info_set_id"]]
                else:
                    parent_info_set = {"a": []}
                if parent == name or parent_node is None or action not in parent_info_set["a"]:
                    raise ValueError(
                        f"Parental problems for node '{name}' from parent '{parent}' with "
                        f"action {action}. Parent in nodes: {parent_node is None}. "
                        "(Expected True.) Action in parent node's information set: "
                        f"{action in parent_info_set['a']}. (Expected True.)"
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

    def check_totality_and_cototality(self):
        """Check that totality and co-totality holds for all histories.

        Totality means that every measurement accessible from the current causal history must be
        assigned an outcome. Co-totality is the inverse of this: if a measurement is assigned an
        outcome in a history, then all the measurements in its causal support must also be assigned
        an outcome in the same history.

        Raises:
            ValueError if any history does not satisfy the totality and co-totality conditions.
        """
        for history in self.data.get("z", []):
            assignments = history["h"]
            context_label = f"history '{history['z']}'"
            # map information set ids to the chosen actions in this specific set
            mapping = {a["i"]: a["a"] for a in assignments}
            active_isets = set(mapping.keys())

            # 1) co-totality: if a measurement is present, it must be enabled by its causal past
            for iset_id in active_isets:
                # information set is supported if at least one of its nodes has all parents
                # satisfied by the current history/strategy assignments.
                is_supported = False
                for node_entry in self.info_sets[iset_id]["ns"]:
                    node_name = node_entry["n"]
                    parents = self.nodes[node_name]["node_data"]["ps"]

                    # all([]) is True => root nodes are enabled
                    if all(
                        p["p"] in self.nodes
                        and self.nodes[p["p"]]["info_set_id"] in mapping
                        and mapping[self.nodes[p["p"]]["info_set_id"]] == p["a"]
                        for p in parents
                    ):
                        is_supported = True
                        break

                if not is_supported:
                    raise ValueError(
                        f"Co-totality violation in {context_label}: information set '{iset_id}' "
                        "is present, but none of its nodes are enabled by the causal past "
                        "defined in the assignments."
                    )

            # 2) totality: if a measurement is enabled by the current history, it must be present
            for node_name, node_info in self.nodes.items():
                parents = node_info["node_data"]["ps"]

                # check if this specific node is enabled by the assignments
                if all(
                    p["p"] in self.nodes
                    and self.nodes[p["p"]]["info_set_id"] in mapping
                    and mapping[self.nodes[p["p"]]["info_set_id"]] == p["a"]
                    for p in parents
                ):
                    target_iset = node_info["info_set_id"]
                    if target_iset not in active_isets:
                        raise ValueError(
                            f"Totality violation in {context_label}: measurement '{target_iset}' "
                            f"(node '{node_name}') is enabled but has no assigned outcome."
                        )
        return True

    def check_histories_consistency(self):
        """Verify that assignments and utilities in histories reference valid entities.

        Checks all histories to ensure:
        1. Actions are playable in the specified information sets.
        2. Utility (u) lists all players in the game.

        Raises:
            ValueError: If history logic is broken or players are missing from payoffs.
        """
        # TODO: check that all possible histories are included
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
        # TODO: check that all possible strategies are included
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

    def check_reduced_strategies_consistency(self):
        """Verify reduced strategy validity, reachability, and uniqueness.

        Checks:
            1. Basic structure: player matching and valid action choices.
            2. Reachability: ⟂ is only used for non-activated sets; real actions
               are only used for activated sets.
            3. Uniqueness: No duplicate reduced strategies for a player.

        Raises:
            ValueError if any of the above checks are violated.
        """
        if "rs" not in self.data:
            return True

        player_to_isets = defaultdict(set)
        for iset, data in self.info_sets.items():
            player_to_isets[data["p"]].add(iset)

        for strategy_group in self.data["rs"]:
            player = strategy_group["p"]
            if player not in self.players:
                raise ValueError(f"Reduced strategy group lists unknown player '{player}'.")

            player_isets = player_to_isets[player]
            seen_strategies = set()

            for strategy in strategy_group["s"]:
                # 1) convert strategy to a lookup map and check basic validity
                strategy_map = dict()
                for assignment in strategy:
                    iset_id, action = assignment.values()

                    if iset_id not in player_isets:
                        raise ValueError(
                            f"Player '{player}' strategy contains foreign info set '{iset_id}'."
                        )

                    if action != "⟂" and action not in self.info_sets[iset_id]["a"]:
                        raise ValueError(f"Invalid action '{action}' in info set '{iset_id}'.")

                    strategy_map[iset_id] = action

                # 2) get all activated information sets, given the strategy
                # TODO: this call is expensive, should probably optimize in some way
                activated = self._get_activated_information_sets_for_player(player, strategy_map)

                # 3) verify reachability/bottom consistency
                for iset_id in player_isets:
                    action = strategy_map.get(iset_id)
                    is_active = iset_id in activated

                    if is_active and action == "⟂":
                        raise ValueError(
                            f"Information set '{iset_id}' is reachable "
                            f"but assigned '⟂' in strategy: {strategy}"
                        )
                    if not is_active and action != "⟂":
                        raise ValueError(
                            f"Information set '{iset_id}' is not reachable "
                            f"but assigned real action '{action}' in strategy: {strategy}"
                        )

                # 4) check for uniqueness
                strategy_tuple = tuple(sorted(tuple(item.values()) for item in strategy))
                if strategy_tuple in seen_strategies:
                    raise ValueError(
                        f"Duplicate reduced strategy found for player '{player}': {strategy}"
                    )
                seen_strategies.add(strategy_tuple)

        return True

    def _get_activated_information_sets_for_player(self, player, strategy_map):
        """Return the set of info_set_ids activated by this player's strategy."""
        activated = set()
        # start with nodes that have no parents
        reachable_nodes = set(
            name for name, node_info in self.nodes.items() if not node_info["node_data"]["ps"]
        )

        # the DAG form of the game means we can propagate reachability
        # using a simple fixed-point iteration or topological approach
        seen = set()
        while True:
            new_reachable = False
            for name in list(reachable_nodes):
                if name in seen:
                    continue
                seen.add(name)

                iset_id = self.nodes[name]["info_set_id"]
                activated.add(iset_id)

                # find children nodes enabled by these possible actions
                for child_name, child_info in self.nodes.items():
                    if child_name in reachable_nodes:
                        continue

                    # a child node is reachable if parent requirements are met
                    parents = child_info["node_data"]["ps"]
                    if not parents:
                        continue

                    child_reachable = True
                    for p in parents:
                        parent, parent_action = p.values()
                        p_iset = self.nodes[parent]["info_set_id"]

                        # if the parent node is not reachable, or the required action
                        # is not among the 'possible' actions, we cannot reach this child
                        if parent not in reachable_nodes or (
                            self.info_sets[p_iset]["p"] == player
                            and strategy_map.get(p_iset) != parent_action
                        ):
                            child_reachable = False
                            break
                        # NOTE: for other players, we already assume any of their
                        # actions are possible if their node is reachable

                    if child_reachable:
                        reachable_nodes.add(child_name)
                        new_reachable = True

            if not new_reachable:
                break

        return activated

    def add_played_information_sets(self):
        """Populate each history with the information sets activated in that history."""
        for history in self.data.get("z", []):
            # skip if the field already exists
            if "s" in history:
                continue
            history["s"] = [assignment["i"] for assignment in history["h"]]

    def add_histories(self):
        """Add missing complete histories by traversing the DAG.

        Uses an inductive expansion based on the enabling relation. A history is
        built by identifying all accessible (enabled) information sets and branching
        on their possible outcomes until the history is total and co-total (i.e. complete).
        """
        if "z" not in self.data:
            self.data["z"] = []

        def get_content(h_list):
            return frozenset(tuple(item.values()) for item in h_list)

        existing_contents = set(get_content(z["h"]) for z in self.data["z"])
        all_iset_ids = self.info_sets.keys()

        def is_node_enabled(node_name, current_h_dict):
            parents = self.nodes[node_name]["node_data"]["ps"]
            if not parents:
                return True

            # group parent requirements by their information set
            # - all different information sets must be satisfied
            # - if multiple parents are in the same information set, only one must be satisfied
            iset_to_required_actions = defaultdict(set)
            for p in parents:
                p_iset = self.nodes[p["p"]]["info_set_id"]
                iset_to_required_actions[p_iset].add(p["a"])

            for p_iset, allowed_actions in iset_to_required_actions.items():
                if p_iset not in current_h_dict or current_h_dict[p_iset] not in allowed_actions:
                    return False
            return True

        def is_iset_enabled(iset_id, current_h_dict):
            # an information set is enabled if at least one of its nodes is enabled
            for node_entry in self.info_sets[iset_id]["ns"]:
                if is_node_enabled(node_entry["n"], current_h_dict):
                    return True
            return False

        def expand_history(current_h_dict):
            # find all information sets not yet in history that are now enabled
            # TODO: optimize, since we only take the first element below
            candidates = [
                i
                for i in all_iset_ids
                if i not in current_h_dict and is_iset_enabled(i, current_h_dict)
            ]

            if not candidates:
                # history is total (maximal)
                h_list = [{"i": i, "a": a} for i, a in sorted(current_h_dict.items())]
                content = get_content(h_list)
                if content not in existing_contents:
                    history_id = "z_" + "".join(
                        str(current_h_dict[i]) for i in sorted(current_h_dict.keys())
                    )
                    self.data["z"].append(
                        {
                            "z": history_id,
                            "h": h_list,
                            "s": sorted(list(current_h_dict.keys())),
                            "u": [{"p": p, "v": 0} for p in self.players],
                        }
                    )
                    existing_contents.add(content)
                return

            # to avoid permutations of the same history, we pick the first available enabled
            # information set and branch on its actions -- totality ensures other candidates will
            # be picked up in subsequent recursion levels
            target_iset = candidates[0]
            for action in self.info_sets[target_iset]["a"]:
                new_h = current_h_dict.copy()
                new_h[target_iset] = action
                expand_history(new_h)

        # start recursion from an empty history
        expand_history(dict())

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

    def add_reduced_strategies(self):
        """Add all reduced strategies for every player to the game data.

        A reduced strategy assigns outcomes only to activated information sets.
        An information set is activated if there exists a sequence of outcomes
        for other players that reaches it, given the current player's strategy.
        """
        if "rs" not in self.data:
            self.data["rs"] = []

        player_to_isets = defaultdict(set)
        for iset, data in self.info_sets.items():
            player_to_isets[data["p"]].add(iset)

        player_to_strategy = {strategy["p"]: strategy for strategy in self.data["rs"]}
        missing = self.players - set(player_to_strategy)
        for player in missing:
            strategy_skeleton = {"p": player, "s": []}
            player_to_strategy[player] = strategy_skeleton
            self.data["rs"].append(strategy_skeleton)

        for player in self.players:
            player_isets = player_to_isets[player]
            group = player_to_strategy[player]

            # TODO: check all the existing strategies and add the hashable representations to the
            # set so that we do not add duplicate strategies
            existing_contents = set()

            def expand(current_map):
                # 1) identify what is currently activated based on choices made so far
                activated = self._get_activated_information_sets_for_player(player, current_map)
                player_activated = activated.intersection(player_isets)

                # 2) find info sets we still need to decide
                to_decide = [i for i in player_activated if i not in current_map]

                if not to_decide:
                    # base case: no more reachable nodes to decide for
                    final_strategy = []
                    for i in player_isets:
                        # assign action if activated, else bottom
                        # TODO: should probably use None instead of '⟂' and update schema
                        action = current_map.get(i, "⟂")
                        final_strategy.append({"i": i, "a": action})

                    content = frozenset(tuple(assignment.values()) for assignment in final_strategy)
                    if content not in existing_contents:
                        group["s"].append(final_strategy)
                        existing_contents.add(content)
                    return

                # 3) branch on the next available activated info set
                target = to_decide[0]
                for action in self.info_sets[target]["a"]:
                    next_map = current_map.copy()
                    next_map[target] = action
                    expand(next_map)

            expand(dict())

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
        """Add everything that can be added based on the base CCS data.

        For correctness, certain methods need to be run before others, in particular, we need to add
        strategies and histories before adding human readable, and we need to add histories before
        we can add the field of which information sets are played in the history.
        """
        methods_to_add = {
            name: member
            for name, member in inspect.getmembers(self)
            if inspect.ismethod(member) and name.startswith("add")
        }
        del methods_to_add["add_histories"]
        del methods_to_add["add_human_readable"]
        self.add_histories()
        for method in methods_to_add.values():
            method()
        # calling this last ensures all information is available for the method
        self.add_human_readable()

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
        self.all_adds()
        # then check that everything is correct
        self.all_checks()
        return True


class AlternatingSpacetimeGame(SpacetimeGame):
    """Subclass of SpacetimeGame enforcing alternating game properties.

    An alternating spacetime game G satisfies the following properties:
        2-PLAYERS
            It has two players (we call them Alfred, who is nature, and Bob, who is the observer).
        BIPARTITE
            The graph (N ,R) always connects a node played by Alfred to a node played
            by Bob, or a node played by Bob to a node played by Alfred. In other words,
            the graph structure of the spacetime game is a bipartite graph if we ignore
            the direction of the edges.
        EVEN
            All root nodes are played by Bob. All leaf nodes are played by Alfred.
        BOB-S
            All information sets played by Bob are singletons. It means Bob is fully
            informed about decisions in his causal past, even in situations in which he
            can carry out the same experiment under different circumstances.
        BOB-A
            At any of Bob’s nodes, all available actions are used on at least one edge.
                ∀t ∈ N_B ,∀C ∈ χ(t),∃n ∈ N_A,σ(t,n)= C
        BA1
            Each node played by Alfred has exactly one parent node:
                ∀n ∈ N_A,∃!t ∈ N_B ,t ⌣ n
        BA2
            Given a node N played by Bob, two nodes (played by Alfred) connected
            to N with the same label (it is a measurement context) must be in different
            information sets (these are measurement settings). In other words, distinct
            nodes for the same measurement in the same context would be superfluous.
                ∀t ∈ N_B ,∀n,m ∈ successors(t),σ(t,n)= σ(t,m) ⇒ ι(n) ̸= ι(m)
        BA3
            Given a node played by Bob, two different outgoing labels cannot point to
            the exact same information sets.
                ∀t ∈ N_B ,∀C_1,C_2 ∈ χ(t),{ι(n)|σ(t,n)= C_1} = {ι(n)|σ(t,n)= C_2} ⇒ C_1 = C_2
        AB1
            All nodes in the same information set played by Alfred15 have the same
            outgoing edges: same labels, same destination nodes. In other words, the
            causal future of a measurement does not depend on the context in which it
            was carried out.
                ∀x ∈ I_A,∀n,m ∈ x,successors(n)= successors(m)
                ∀x ∈ I_A,∀n,m ∈ x,∀u ∈ successors(x),σ(n,u)= σ(m,u)
        AB2
            Two distinct nodes played by Bob cannot have the same causal bridge.
                ∀t,u ∈ N_B,
                (predecessors(t)=predecessors(u) ∧ ∀n ∈ predecessors(t),σ(n,t)= σ(n,u)) ⇒ t = u.

        Definition from:
            Fourny, G. Spacetime Games Subsume Causal Contextuality Scenarios.
            Int J Theor Phys 65, 95 (2026). https://doi.org/10.1007/s10773-026-06295-4
    """

    def __init__(self, json_data):
        super().__init__(json_data)
        self.bob_player = None
        self.alfred_player = None
        self._identify_players()

    def _identify_players(self):
        """Identify which player is Bob (roots) and which is Alfred (leaves)."""
        if len(self.players) != 2:
            return  # check_2_players will catch this

        for node_info in self.nodes.values():
            # root: no parents
            if not node_info["node_data"]["ps"]:
                self.bob_player = node_info["player"]
                break
        # NOTE: this code assumes two players
        self.alfred_player = list(self.players - set([self.bob_player]))[0]

    def check_2_players(self):
        """Check that the game has exactly two players."""
        return len(self.players) == 2

    def check_bipartite(self):
        """Verify that the spacetime game DAG is bipartite with respect to the players.

        All Alfred nodes connect to Bob nodes and vice versa.
        """
        for name, node_info in self.nodes.items():
            current_player = node_info["player"]
            for edge in self.adj.get(name, []):
                child_player = self.nodes[edge["c"]]["player"]
                if current_player == child_player:
                    return False
        return True

    def check_roots_and_leaves(self):
        """Ensure all roots are Bob; all leaves are Alfred."""
        for name, node_info in self.nodes.items():
            # check roots
            if not node_info["node_data"]["ps"] and node_info["player"] != self.bob_player:
                return False
            # check leaves
            if name not in self.adj and node_info["player"] != self.alfred_player:
                return False
        return True

    def check_singleton_bob_info_sets(self):
        """Check that all Bob's information sets are singletons."""
        for iset in self.info_sets.values():
            if iset["p"] == self.bob_player and len(iset["ns"]) != 1:
                return False
        return True

    def check_bob_a(self):
        """Check that all Bob actions available are played (label of some edge)."""
        for name, node_info in self.nodes.items():
            if node_info["player"] == self.bob_player:
                iset = self.info_sets[node_info["info_set_id"]]
                available_actions = set(iset["a"])
                used_actions = set(edge["a"] for edge in self.adj.get(name, []))
                if available_actions != used_actions:
                    return False
        return True

    def check_ba1(self):
        """Ensure each Alfred node has exactly one parent node."""
        for node_info in self.nodes.values():
            if node_info["player"] == self.alfred_player:
                if len(node_info["node_data"]["ps"]) != 1:
                    return False
        return True

    def check_ba2(self):
        """Verify property BA2 of the alternating game category."""
        for name, node_info in self.nodes.items():
            if node_info["player"] == self.bob_player:
                # map action label to set of info set ids
                action_to_isets = defaultdict(list)
                for edge in self.adj.get(name, []):
                    child_iset = self.nodes[edge["c"]]["info_set_id"]
                    action_to_isets[edge["a"]].append(child_iset)

                for iset_list in action_to_isets.values():
                    if len(iset_list) != len(set(iset_list)):
                        return False
        return True

    def check_ba3(self):
        """Ensure different outgoing labels from Bob must point to different information sets."""
        for name, node_info in self.nodes.items():
            if node_info["player"] == self.bob_player:
                # map action label to frozen set of child info set ids
                label_to_isets = defaultdict(set)
                for edge in self.adj.get(name, []):
                    action = edge["a"]
                    child_iset = self.nodes[edge["c"]]["info_set_id"]
                    label_to_isets[action].add(child_iset)
                label_to_isets_set = {
                    label: frozenset(isets) for label, isets in label_to_isets.items()
                }

                # check that distinct labels point to distinct sets
                seen_sets = set()
                for label_set in label_to_isets_set.values():
                    if label_set in seen_sets:
                        return False
                    seen_sets.add(label_set)
        return True

    def check_ab1(self):
        """Check that nodes in the same Alfred info set have identical outgoing edges."""
        for iset in self.info_sets.values():
            if iset["p"] == self.alfred_player:
                nodes_in_set = [n["n"] for n in iset["ns"]]
                if not nodes_in_set:
                    continue

                # Use the first node as the reference
                ref_node = nodes_in_set[0]
                ref_edges = set(tuple(edge.values()) for edge in self.adj.get(ref_node, []))

                for other_node in nodes_in_set[1:]:
                    other_edges = set(tuple(edge.values()) for edge in self.adj.get(other_node, []))
                    if ref_edges != other_edges:
                        return False
        return True

    def check_ab2(self):
        """Verify that the game has unique causal bridges."""
        bob_nodes = [
            name for name, node_info in self.nodes.items() if node_info["player"] == self.bob_player
        ]
        bridges = set()
        for node in bob_nodes:
            # a bridge is the set of (parent_node, action)
            bridge = frozenset(
                tuple(parent.values()) for parent in self.nodes[node]["node_data"]["ps"]
            )
            if bridge in bridges:
                return False
            bridges.add(bridge)
        return True

    def check_even_height(self):
        """Check that every Alfred node has even height and every Bob node has odd height."""
        leaves = set(self.nodes) - set(node for node, data in self.adj.items() if data)
        seen = {leaf: 0 for leaf in leaves}

        def get_height(node):
            if node in seen:
                # NOTE: this will cover the leaves too, since they have already been added
                return seen[node]
            res = 1 + max(get_height(edge["c"]) for edge in self.adj[node])
            seen[node] = res
            return res

        for name, node_info in self.nodes.items():
            parity = 0 if node_info["player"] == self.alfred_player else 1
            if get_height(name) % 2 != parity:
                return False
        return True
