#!/usr/bin/env python3
"""
Vertex Cover Visualization + LaTeX Figure Generator

Visualizes the results of the Vertex Cover task.

Each agent outputs either "Yes" (coordinator / in cover)
or "No" (regular node). 

🟩 Green = valid minimal vertex cover (good coverage and minimality)
🟦 Blue  = "Yes" nodes in valid cover
🔴 Red   = coverage failure (uncovered edge)
🟧 Orange = invalid answers
⚪ Gray   = non-cover nodes ("No")

Run examples:

# Single file (interactive)
python -m multi_agent.topology.visualization.vertexcover_main results/ontology/vertex_cover_results_*.json --interactive --layout auto

# Batch (generate images + LaTeX)
python -m multi_agent.topology.visualization.vertexcover_main results/ontology/vertex_cover_*.json --latex --layout auto
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
def create_vertexcover_network(result_data, layout: str = "spring") -> go.Figure:
    """
    Visualize vertex cover results:
    🟩 Green: minimal valid vertex cover (Yes nodes form full coverage)
    🟦 Blue: valid "Yes" node in cover
    🔴 Red: uncovered edge (coverage failure)
    🟧 Orange: invalid answers
    ⚪ Gray: non-cover nodes ("No")
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

    # --- Compute coverage and minimality ---
    uncovered_edges = [(u, v) for u, v in G.edges()
                       if not ((u in yes_nodes) or (v in yes_nodes))]
    coverage = len(uncovered_edges) == 0
    valid = len(invalid_nodes) == 0

    # minimality: for each "Yes", if setting it to "No" breaks full coverage
    minimal_count = 0
    for u in yes_nodes:
        _yes = set(yes_nodes)
        _yes.discard(u)
        still_cover = all(((x in _yes) or (y in _yes)) for x, y in G.edges())
        if not still_cover:
            minimal_count += 1

    minimal = (minimal_count == len(yes_nodes)) if yes_nodes else False
    success = valid and coverage and minimal

    # --- Node colors ---
    colors = []
    for n in G.nodes():
        if n in invalid_nodes:
            colors.append("#FFA500")  # orange = invalid answer
        elif n in yes_nodes:
            if success:
                colors.append("#2E8B57")  # green if perfect
            else:
                colors.append("#1E90FF")  # blue for cover nodes
        else:
            colors.append("#CFCFCF")  # gray = non-cover

    # --- Node sizes ---
    sizes = []
    for n in G.nodes():
        if n in yes_nodes and success:
            sizes.append(28)  # big green (valid minimal cover)
        elif n in yes_nodes:
            sizes.append(20)
        elif n in invalid_nodes:
            sizes.append(18)
        else:
            sizes.append(12)

    # --- Layout positions ---
    pos = create_layout_positions(G, layout)

    # --- Edge visualization ---
    edge_x_cov, edge_y_cov = [], []
    edge_x_fail, edge_y_fail = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if (u, v) in uncovered_edges:
            edge_x_fail += [x0, x1, None]
            edge_y_fail += [y0, y1, None]
        else:
            edge_x_cov += [x0, x1, None]
            edge_y_cov += [y0, y1, None]

    fig = go.Figure()

    # covered edges = gray
    fig.add_trace(go.Scatter(
        x=edge_x_cov, y=edge_y_cov,
        mode="lines",
        line=dict(color="#B0B0B0", width=1.5),
        opacity=0.3,
        name="Covered Edges",
        hoverinfo="none"
    ))

    # uncovered edges = red
    fig.add_trace(go.Scatter(
        x=edge_x_fail, y=edge_y_fail,
        mode="lines",
        line=dict(color="#DC143C", width=2),
        opacity=0.8,
        name="Uncovered Edges",
        hoverinfo="none"
    ))

    # --- Nodes ---
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    hover_text = [f"{G.nodes[n]['name']} → {answers[n] if n < len(answers) else 'None'}"
                  for n in G.nodes()]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=1.2, color="black")),
        text=[("C" if n in yes_nodes else "") for n in G.nodes()],
        textfont=dict(size=12, color="white", family="Arial Black"),
        textposition="middle center",
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))

    # --- Layout ---
    fig.update_layout(
        title=f"Vertex Cover - {result_data.get('graph_generator','Unknown').upper()} Topology<br>"
              f"Score: {result_data.get('score',0):.2f} | Success: {success}",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.8, y=1.25)
    )
    return fig


# ------------------- LaTeX Figure Generator -------------------
def generate_latex_figure_group(image_dict, caption=None):
    if caption is None:
        caption = (
            "Vertex cover results across different network sizes ($n=4$ to $100$). "
            "Each subfigure shows the agents' final decisions. "
            "Green nodes represent a minimal valid cover, blue nodes indicate members of a non-minimal cover, "
            "red edges mark uncovered pairs, orange nodes represent invalid responses, and gray nodes are non-cover agents."
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
    latex.append("    \\label{{fig:vertexcover-results}}")
    latex.append("\\end{figure*}")
    return "\n".join(latex)


# ------------------- Main Entry -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize vertex cover experiment results and generate LaTeX block"
    )
    parser.add_argument("files", nargs="+", help="Vertex cover result JSON file(s) to visualize")
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
        fig = create_vertexcover_network(result_data, layout=layout)

        p = Path(filepath)
        framework = result_data.get("framework", "unknown").lower()

        # 🔄 For Concordia framework, aggregate all under one category
        if framework == "concordia":
            graph_type = "all"
        else:
            graph_type = result_data.get("graph_generator", "unknown").lower()

        num_nodes = result_data.get("num_nodes", 0)
        out_dir = p.parent

        suffix = "_vertexcover"
        output_path = out_dir / f"{p.stem}_{framework}{suffix}.png"
        fig.write_image(str(output_path))
        print(f"✅ Saved {output_path}")

        generated.setdefault(graph_type, []).append((num_nodes, output_path))

    # 📄 Optionally generate LaTeX snippet for all results
    if args.latex:
        latex_code = generate_latex_figure_group(generated)
        tex_file = Path("results/ontology/vertexcover_enhanced_figures.tex")
        tex_file.write_text(latex_code)
        print(f"📄 LaTeX code written to {tex_file}")


if __name__ == "__main__":
    main()
