#!/usr/bin/env python3
"""
Leader Election Visualization + LaTeX Figure Generator

Visualizes the results of the leader election task.
Each agent outputs either "Yes" (leader) or "No" (follower).
Success = exactly one agent says "Yes" and all others say "No".

Run examples:

# Single file (interactive)
python -m multi_agent.topology.visualization.leader_main results/ontology/leader_results_*.json --interactive --layout auto

# Batch (generate images + LaTeX)
python -m multi_agent.topology.visualization.leader_main results/ontology/leader_*.json --latex --layout auto
"""

import argparse
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
from .loader import load_result_file
from .layouts import create_layout_positions


# ------------------- Layout Selection -------------------
def pick_layout(result_data, user_layout: str) -> str:
    """Choose layout based on graph type if user selects 'auto'."""
    if user_layout != "auto":
        return user_layout

    topo = result_data.get("graph_generator", "").lower()
    if topo == "ws":
        return "circular"
    elif topo in ["ba", "scale-free"]:
        return "spring"
    elif topo in ["dt", "delaunay"]:
        return "kamada"
    elif topo in ["sequential", "crewai-sequential"]:
        return "spectral"
    elif topo in ["hierarchical", "crewai-hierarchical"]:
        return "kamada"
    else:
        return "spring"


# ------------------- Visualization -------------------
def create_leader_network(result_data, layout: str = "spring") -> go.Figure:
    """
    Visualize leader election results.
    🟩 Green: valid elected leader (exactly one "Yes")
    🔴 Red: multiple or invalid leaders (more than one "Yes")
    ⚪ Gray: followers ("No")
    🟧 Orange: if some nodes answered with invalid responses
    """
    # --- Build graph ---
    G = nx.Graph()
    nodes = result_data.get("graph", {}).get("nodes", [])
    id_to_name = {n["id"]: n["name"] for n in nodes}
    for i in range(len(nodes)):
        G.add_node(i, name=id_to_name.get(i, f"Agent {i+1}"))

    for e in result_data.get("graph", {}).get("links", []):
        G.add_edge(e["source"], e["target"])

    answers = result_data.get("answers", [])
    yes_nodes = [i for i, a in enumerate(answers) if str(a).strip().lower() == "yes"]
    no_nodes = [i for i, a in enumerate(answers) if str(a).strip().lower() == "no"]
    invalid_nodes = [i for i, a in enumerate(answers) if str(a).strip().lower() not in ["yes", "no"]]

    # --- Determine success/failure ---
    valid = len(invalid_nodes) == 0
    one_leader = len(yes_nodes) == 1
    success = valid and one_leader
    leader_nodes = yes_nodes.copy()

    # --- Node colors ---
    colors = []
    if not valid:
        invalid_set = set(invalid_nodes)
    else:
        invalid_set = set()

    for n in G.nodes():
        if n in invalid_set:
            colors.append("#FFA500")  # orange = invalid answer
        elif n in leader_nodes:
            if len(leader_nodes) == 1:
                colors.append("#2E8B57")  # green = correct single leader
            else:
                colors.append("#DC143C")  # red = multiple leaders
        else:
            colors.append("#C0C0C0")  # gray = follower

    # --- Layout positions ---
    pos = create_layout_positions(G, layout)

    # --- Draw edges ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#B0B0B0", width=1),
        opacity=0.3,
        name="Network Edges",
        hoverinfo="none"
    ))

    # --- Nodes ---
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    hover_text = [f"{G.nodes[n]['name']} → {answers[n] if n < len(answers) else 'None'}"
                for n in G.nodes()]

    # --- Dynamic node sizing ---
    sizes = []
    for n in G.nodes():
        if n in leader_nodes and len(leader_nodes) == 1:
            sizes.append(28)  # 🟩 big green leader
        elif n in leader_nodes and len(leader_nodes) > 1:
            sizes.append(20)  # 🔴 slightly larger if multiple leaders
        elif n in invalid_set:
            sizes.append(18)  # 🟧 moderate size for invalids
        else:
            sizes.append(12)  # ⚪ regular followers

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=1.2, color="black")),
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))


    # --- Layout ---
    fig.update_layout(
        title=f"Leader Election - {result_data.get('graph_generator','Unknown').upper()} Topology<br>"
              f"Score: {result_data.get('score',0):.2f} | Success: {success}",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=False
    )

    return fig


# ------------------- LaTeX Figure Generator -------------------
def generate_latex_figure_group(image_dict, caption=None):
    if caption is None:
        caption = (
            "Leader election results across different network sizes ($n=4$ to $100$). "
            "Each subfigure shows the final decisions of agents. "
            "Green nodes represent the correctly elected leader, red nodes indicate multiple leaders, "
            "orange nodes mark invalid responses, and gray nodes represent followers."
        )

    latex = []
    latex.append("\\begin{figure*}[t]")
    latex.append("    \\centering")
    row_order = ["ws", "ba", "dt", "sequential", "hierarchical", "all"]
    label_map = {"ws": "WS", "ba": "BA", "dt": "DT",
                 "sequential": "Seq.", "hierarchical": "Hier.", "all": "All"}
    sub_fig_size = {4: "0.12", 8: "0.16", 16: "0.19", 50: "0.22", 100: "0.26"}

    for topo in row_order:
        if topo not in image_dict:
            continue
        latex.append(f"    % --- {topo.upper()} ---")
        for n, path in sorted(image_dict[topo], key=lambda x: x[0]):
            fig_path = str(path).replace("results/", "figures/").replace("\\", "/")
            fig_path = str(fig_path).replace("ontology/", "topology/")
            width = sub_fig_size.get(n, "0.20")
            label = label_map.get(topo, topo.upper())
            latex.append(
                f"    \\subfloat[{label} ($n={n}$)]{{"
                f"\\includegraphics[trim=90 90 80 90,clip,width={width}\\textwidth]{{{fig_path}}}}}"
            )
        latex.append("    \\\\")
        latex.append("    \\vspace{2mm}")

    latex.append(f"    \\caption{{{caption}}}")
    latex.append("    \\label{{fig:leader-results}}")
    latex.append("\\end{figure*}")
    return "\n".join(latex)


# ------------------- Main Entry -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize leader election experiment results and generate LaTeX block"
    )
    parser.add_argument("files", nargs="+", help="Leader election result JSON file(s) to visualize")
    parser.add_argument(
        "--layout",
        choices=["spring", "circular", "kamada", "spectral", "community", "auto"],
        default="spring",
        help="Network layout algorithm"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Plotly visualization")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX snippet for all figures")

    args = parser.parse_args()

    generated = {}  # {graph_type: [(n, path)]}

    for filepath in args.files:
        result_data = load_result_file(filepath)
        layout = pick_layout(result_data, args.layout)
        fig = create_leader_network(result_data, layout=layout)


        p = Path(filepath)
        framework = result_data.get("framework", "unknown").lower()

        # 🔄 For Concordia framework, aggregate all under one category
        if framework == "concordia":
            graph_type = "all"
        else:
            graph_type = result_data.get("graph_generator", "unknown").lower()

        num_nodes = result_data.get("num_nodes", 0)
        out_dir = p.parent


        suffix = "_leader"
        output_path = out_dir / f"{p.stem}_{framework}{suffix}.png"
        fig.write_image(str(output_path))
        print(f"✅ Saved {output_path}")

        generated.setdefault(graph_type, []).append((num_nodes, output_path))

    # 📄 Optionally generate LaTeX snippet for all results
    if args.latex:
        latex_code = generate_latex_figure_group(generated)
        tex_file = Path("results/ontology/leader_enhanced_figures.tex")
        tex_file.write_text(latex_code)
        print(f"📄 LaTeX code written to {tex_file}")


if __name__ == "__main__":
    main()
