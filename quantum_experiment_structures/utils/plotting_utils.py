"""Collection of helpful functions for plotting."""

from collections import defaultdict
import json

import graphviz


def plot_process_matrix(pmf_data, filename="pmf", format="png", view=True):
    """Visualize the Process Matrix Framework topology.

    Args:
        pmf_data: dictionary or JSON string of the PMF instance.
        filename: output filename.
        format: image format (png, pdf, svg).
        view: whether to open the file immediately.
    """
    if isinstance(pmf_data, str):
        data = json.loads(pmf_data)
    else:
        data = pmf_data

    pmf = data.get("ProcessMatrixFramework", {})
    labs = pmf.get("Labs", [])
    wires = pmf.get("Wires", [])

    dot = graphviz.Digraph(
        name=filename,
        comment="PMF Visualization",
        graph_attr={
            "rankdir": "LR",  # left to right
            "nodesep": "0.7",
            "ranksep": "1.2",
            "splines": "true",
            "fontname": "Arial",
            "fontsize": "48",
        },
        edge_attr={"fontsize": "32", "penwidth": "1.5"},
    )

    # 1) add labs as structured rectangles
    for lab in labs:
        idx = str(lab["Index"])
        name = lab.get("Name", f"Lab {idx}")
        in_q = lab.get("NumberOfInQubits", 0)
        out_q = lab.get("NumberOfOutQubits", 0)

        fill_color = "#ecf0f1"  # default grey
        if name.lower() == "start":
            fill_color = "#d5f5e3"  # greenish
        elif name.lower() == "end":
            fill_color = "#f5b7b1"  # reddish
        elif any(n in name.lower() for n in ["alice", "bob", "chris"]):
            fill_color = "#d6eaf8"  # blueish

        # embed HTML to format the "lab box"
        # this lists the lab name, qubit counts, and measurement summary
        label = f"""<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="2" BGCOLOR="{fill_color}">
                        <FONT POINT-SIZE="36"><B>{name}</B> (ID: {idx})</FONT>
                    </TD>
                </TR>
                <TR>
                    <TD><FONT POINT-SIZE="32">In Qubits: {in_q}</FONT></TD>
                    <TD><FONT POINT-SIZE="32">Out Qubits: {out_q}</FONT></TD>
                </TR>
        """

        # measurement metadata (CPMap outcomes)
        for m in lab["Measurements"]:
            axis = m["MeasurementAxisIndex"]
            outcomes = [str(cp["MeasurementOutcomeIndex"]) for cp in m["CPMaps"]]
            label += (
                '<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="30">'
                f"Axis {axis}: outcomes {{{','.join(outcomes)}}}</FONT></TD></TR>"
            )

        label += "</TABLE>>"
        dot.node(idx, label=label, shape="none")

    # 2) add wires as edges: label edges with specific local port indices
    for wire in wires:
        from_lab = str(wire["From"]["LabIdx"])
        from_port = wire["From"]["OutQubitLocalIdx"]
        to_lab = str(wire["To"]["LabIdx"])
        to_port = wire["To"]["InQubitLocalIdx"]

        edge_label = f"out:{from_port} \u2192 in:{to_port}"

        dot.edge(from_lab, to_lab, label=edge_label)

    dot.render(filename, format=format, cleanup=True, view=view)
    return dot


def plot_extensive_game(
    game_dict, filename="extensive_game", format="png", view=True, same_size=False
):
    """Represent an extensive form game tree visually.

    Args:
        game_dict: The dictionary representing the extensive form game (recursive schema).
        filename: name of file to which the graph will be written.
        format: file format.
        view: boolean indicating whether to immediately open the created file.
        same_size: boolean indicating whether all the nodes should have the same size.
    """

    # 1) collect all labels for same_size heuristic
    all_potential_labels = []

    def collect_labels(node):
        kind = node.get("kind")
        if kind == "choice":
            iset_id = node.get("information-set", "N/A")
            player = node.get("player", "N/A")
            # schema does not have explicit node names, so use a generic placeholder
            all_potential_labels.append(f"Node\n({iset_id})\n[P{player}]")
            for child in node.get("children", node.get("Children", [])):
                collect_labels(child)
        elif kind == "outcome":
            payoffs = node.get("payoffs", [])
            if isinstance(payoffs, list):
                label = "Payoffs:\n" + "\n".join(str(p) for p in payoffs)
            else:
                label = f"Payoff: {payoffs}"
            all_potential_labels.append(label)

    collect_labels(game_dict)

    if same_size:
        max_chars = 0
        max_lines = 0
        for label in all_potential_labels:
            lines = label.split("\n")
            max_lines = max(max_lines, len(lines))
            for line in lines:
                max_chars = max(max_chars, len(line))

        width = max_chars * 0.25
        height = max_lines * 0.5
        node_size_val = max(width, height)
        node_size = str(node_size_val)
        leaf_size = (str(width), str(height))
    else:
        node_size = "2.5"
        leaf_size = ("1.5", "1.0")

    fixed_size = str(same_size).lower()

    # 2) initialize DOT graph
    dot = graphviz.Digraph(
        name=filename,
        comment="GEFII Game Tree",
        graph_attr={
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "nodesep": "0.8",
            "ranksep": "1.0",
        },
        node_attr={
            "fontsize": "32",
            "shape": "circle",
            "fixedsize": fixed_size,
            "width": node_size,
            "height": node_size,
        },
        edge_attr={
            "fontsize": "48",
            "fontcolor": "darkgreen",
        },
    )

    # 3) recursive building of the tree
    node_counter = [0]
    iset_members = defaultdict(list)

    def build_tree(node, parent_id=None, edge_label=None):
        current_id = f"n{node_counter[0]}"
        node_counter[0] += 1

        kind = node["kind"]

        if kind == "choice":
            iset_id = node.get("information-set", "N/A")
            player = node.get("player", "N/A")
            display_label = f"({iset_id})\n[P{player}]"

            dot.node(current_id, display_label)
            iset_members[iset_id].append(current_id)

            # NOTE: the schema defines the property 'Children' with a capital 'C',
            # but other properties are not defined with a capital letter
            children = node.get("Children", [])
            for i, child in enumerate(children):
                build_tree(child, parent_id=current_id, edge_label=str(i))

        elif kind == "outcome":
            payoffs = node.get("payoffs", [])
            if isinstance(payoffs, list):
                payoff_label = "Payoffs:\n" + "\n".join(str(p) for p in payoffs)
            else:
                payoff_label = f"Payoff: {payoffs}"

            dot.node(
                current_id,
                payoff_label,
                shape="box",
                style="dotted",
                width=leaf_size[0],
                height=leaf_size[1],
                fixedsize=fixed_size,
                fontsize="24",
            )

        if parent_id:
            dot.edge(parent_id, current_id, label=edge_label)

    build_tree(game_dict)

    # 4) connect information sets with dashed lines
    for iset_id, nodes in iset_members.items():
        if len(nodes) > 1:
            # create a subgraph manually to force the same horizontal rank
            s = graphviz.Digraph(name=f"rank_iset_{iset_id}")
            s.attr(rank="same")
            for node_id in nodes:
                s.node(node_id)
            dot.subgraph(s)

            for i in range(len(nodes) - 1):
                dot.edge(
                    nodes[i],
                    nodes[i + 1],
                    style="dashed",
                    color="blue",
                    arrowhead="none",
                    constraint="false",
                    penwidth="2.0",
                )

    dot.render(filename, format=format, cleanup=True, view=view)
    return dot


def plot_spacetime_game(game, filename="spacetime_game", format="png", view=True, same_size=False):
    """Produce a visual representation of a spacetime game DAG.

    Args:
        game: the qes.SpacetimeGame instance.
        filename: name of file to which the graph will be written.
        format: file format.
        view: boolean indicating whether to immediately open the created file.
        same_size: boolean indicating whether all the nodes in the created DAG should have the same
            size. If True, the minimum necessary size will be heuristically estimated based on the
            longest labels present in the spacetime game. If False, dynamic sizing will be used.

    Returns:
        the graphviz.Digraph object that is created.
    """
    if same_size:
        # pre-calculate all potential labels to find the maximum dimensions needed
        all_potential_labels = []
        for node_name, node_info in game.nodes.items():
            iset_id = node_info["info_set_id"]
            player = game.info_sets[iset_id].get("p", "N/A")
            all_potential_labels.append(f"{node_name}\n({iset_id})\n[{player}]")

        for history in game.data.get("z", []):
            u_list = [f"{u['p']}:{u['v']}" for u in history["u"]]
            all_potential_labels.append("Payoffs:\n" + "\n".join(u_list))

        max_chars = 0
        max_lines = 0
        for label in all_potential_labels:
            lines = label.split("\n")
            max_lines = max(max_lines, len(lines))
            for line in lines:
                max_chars = max(max_chars, len(line))

        # heuristic for 32pt font: ~0.25in per char width, ~0.5in per line height
        # We take the max of width and height to keep the nodes circular/square
        width = max_chars * 0.25
        height = max_lines * 0.5
        node_size_val = max(width, height)
        node_size = str(node_size_val)
        leaf_size = (str(width), str(height))
    else:
        node_size = "2.5"
        leaf_size = ("1.5", "1.0")
    fixed_size = str(same_size).lower()

    dot = graphviz.Digraph(
        name=filename,
        comment="Spacetime Game DAG",
        graph_attr={
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "nodesep": "0.8",
            "ranksep": "1.0",
        },
        node_attr={
            "fontsize": "32",
            "shape": "circle",
            "fixedsize": fixed_size,
            "width": node_size,
            "height": node_size,
        },
        edge_attr={
            "fontsize": "48",
            "fontcolor": "darkgreen",
        },
    )

    # '{', '}', '|', etc. are special characters with dot, so prefix 'n' to the names
    node_to_id = {name: f"n{i}" for i, name in enumerate(game.nodes.keys())}

    # 1) create decision nodes
    for node_name, node_info in game.nodes.items():
        safe_id = node_to_id[node_name]
        iset_id = node_info["info_set_id"]
        player = game.info_sets[iset_id].get("p", "N/A")

        display_label = f"{node_name}\n({iset_id})\n[{player}]"
        dot.node(safe_id, display_label)

    # 2) create edges and terminal payoff nodes
    for node_name, node_info in game.nodes.items():
        safe_id = node_to_id[node_name]
        iset_id = node_info["info_set_id"]
        actions = game.info_sets[iset_id].get("a", [])

        # track which actions from this node actually lead to children
        out_edges = game.adj.get(node_name, [])
        actions_with_children = set(edge["a"] for edge in out_edges)

        for action in actions:
            if action in actions_with_children:
                # standard causal transition
                for edge in out_edges:
                    if edge["a"] == action:
                        dot.edge(safe_id, node_to_id[edge["c"]], label=str(action))
            else:
                # terminal branch: lead to a payoff node
                leaf_id = f"leaf_{safe_id}_{action}"

                # search for a history 'z' that contains this (iset, action) pair to find payoff
                payoff_label = "End"
                for history in game.data.get("z", []):
                    if any(h["i"] == iset_id and h["a"] == action for h in history["h"]):
                        u_list = [f"{u['p']}:{u['v']}" for u in history["u"]]
                        payoff_label = "Payoffs:\n" + "\n".join(u_list)
                        break

                # render the terminal node as a smaller box or empty shape
                dot.node(
                    leaf_id,
                    payoff_label,
                    shape="box",
                    style="dotted",
                    width=leaf_size[0],
                    height=leaf_size[1],
                    fixedsize=fixed_size,
                    fontsize="24",
                )
                dot.edge(safe_id, leaf_id, label=str(action))

    # 3) handle information sets (add dashed lines)
    iset_members = defaultdict(list)
    for node_name, node_info in game.nodes.items():
        iset_id = node_info["info_set_id"]
        iset_members[iset_id].append(node_name)

    for iset_id, nodes in iset_members.items():
        n = len(nodes)
        if n <= 1:
            continue
        for i in range(n - 1):
            dot.edge(
                node_to_id[nodes[i]],
                node_to_id[nodes[i + 1]],
                style="dashed",
                color="blue",
                arrowhead="none",
                constraint="false",
                penwidth="2.0",
            )

    dot.render(filename, format=format, cleanup=True, view=view)
    return dot
