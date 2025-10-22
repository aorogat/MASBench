#!/usr/bin/env python3
"""
Matching Network Visualization

Visualize the results of the matching task.

Run examples:

# Single file
python -m multi_agent.topology.visualization.matching_main results/ontology/matching_results_20250917_223003_rounds8_gpt-4o-mini_nodes4.json --interactive --layout auto

# Batch
python -m multi_agent.topology.visualization.matching_main results/ontology/matching_*.json --interactive --layout auto

# for images and latex
python -m multi_agent.topology.visualization.matching_main results/ontology/matching_*.json --latex --layout auto

# for images and latex (use original way to visualize (same way used to compute score))
python -m multi_agent.topology.visualization.matching_main results/ontology/matching_*.json --latex --original --layout auto


"""

#!/usr/bin/env python3
"""
Matching Network Visualization + LaTeX Figure Generator

Generates visualizations for matching experiments and produces a LaTeX file
grouping all figures by topology (e.g., WS, BA, DT, Seq., Hierarchical)
similar to the coloring experiments figure layout.
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


# ------------------- Main Visualization -------------------
def create_matching_network_original_code(result_data, layout: str = "spring") -> go.Figure:
    """
    Visualization aligned with Matching.get_score logic. This function follow the same way agentsnet compute the score.
    Each agent locally according to two consistency criteria: (i)~an agent must select one of its immediate neighbors in the underlying topology, and (ii)~the selected neighbor must reciprocate the choice. Any violation of these rules---for example, choosing a non-neighbor or a non-reciprocating agent---is counted as an inconsistency. Additionally, if two neighboring agents both select ``None,'' this pair is penalized to discourage idle behavior when matching opportunities exist. The final score is computed as 
    $\text{Score} = 1 - \frac{\text{InconsistentNodes}}{|V|}$,
    where $|V|$ is the number of agents in the graph. Thus, the score reflects the fraction of locally consistent agents rather than the number of globally disjoint pairs. 

    """
    G = nx.Graph()
    nodes = result_data.get("graph", {}).get("nodes", [])
    id_to_name = {n["id"]: n["name"] for n in nodes}
    name_to_id = {v: k for k, v in id_to_name.items()}

    for i in range(len(nodes)):
        G.add_node(i, name=id_to_name.get(i, f"Agent {i+1}"))

    for e in result_data.get("graph", {}).get("links", []):
        G.add_edge(e["source"], e["target"])

    # --- Answers and mappings ---
    answers = result_data.get("answers", [])
    node_names = [id_to_name[i] for i in G.nodes]
    name_to_match = {node_names[i]: answers[i] for i in range(len(node_names))}

    # --- Local consistency check (same as get_score) ---
    inconsistent_nodes = set()
    inconsistent_edges = set()

    for node in G.nodes:
        matching_name = answers[node] if node < len(answers) else "None"
        if matching_name != "None":
            # Invalid neighbor
            neighbor_names = [node_names[u] for u in G.neighbors(node)]
            if matching_name not in neighbor_names:
                inconsistent_nodes.add(node)
            # Non-reciprocal match
            elif name_to_match.get(matching_name, None) != node_names[node]:
                inconsistent_nodes.add(node)
                # Add visible edge for non-mutual connection
                if matching_name in name_to_id:
                    inconsistent_edges.add(tuple(sorted((node, name_to_id[matching_name]))))
        else:
            # Both sides 'None'
            for v in G.neighbors(node):
                if answers[v] == "None":
                    inconsistent_nodes.add(node)
                    inconsistent_nodes.add(v)
                    inconsistent_edges.add(tuple(sorted((node, v))))
                    break

    # --- Layout ---
    pos = create_layout_positions(G, layout)

    # --- Edge categories ---
    gray_x, gray_y = [], []
    orange_x, orange_y = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge = tuple(sorted((u, v)))
        if edge in inconsistent_edges:
            orange_x += [x0, x1, None]
            orange_y += [y0, y1, None]
        else:
            gray_x += [x0, x1, None]
            gray_y += [y0, y1, None]

    # --- Draw edges ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gray_x, y=gray_y,
        mode="lines",
        line=dict(color="#B0B0B0", width=1),
        opacity=0.3,
        name="Other Edges",
        hoverinfo="none"
    ))

    fig.add_trace(go.Scatter(
        x=orange_x, y=orange_y,
        mode="lines",
        line=dict(color="#FFA500", width=2, dash="dot"),
        opacity=0.9,
        name="Local Inconsistencies",
        hoverinfo="none"
    ))

    # --- Node colors ---
    node_colors = []
    for n in G.nodes:
        if n in inconsistent_nodes:
            node_colors.append("#DC143C")  # red = inconsistent
        else:
            node_colors.append("#2E8B57")  # green = locally consistent

    # --- Coordinates ---
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    hover_text = [f"{G.nodes[n]['name']} → {answers[n] if n < len(answers) else 'None'}"
                  for n in G.nodes()]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        marker=dict(size=12, color=node_colors, line=dict(width=1, color="black")),
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))

    # --- Layout ---
    fig.update_layout(
        title=f"Matching Task - {result_data.get('graph_generator','Unknown').upper()} Topology<br>"
              f"Score: {result_data.get('score',0):.2f} | "
              f"Success: {result_data.get('successful', False)}",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.8, y=1.25)
    )

    return fig

def create_matching_network(result_data, layout: str = "spring") -> go.Figure:
    """
    Visualize matching task:
    - 🟩 Green: mutual pairs (always green even if reused)
    - 🔴 Red: conflicting or reused selections (outsiders targeting taken agents)
    - 🟧 Orange: one-sided valid selections
    - ⚪ Gray: topology-only connections

    The enhanced visualization extends this local logic to explicitly mark \emph{conflict edges} and \emph{consistent pairs}. A green edge connects mutually consistent agents whose choices satisfy the above criteria; an orange dotted edge indicates a local inconsistency (e.g., one-sided or invalid neighbor selection), while red nodes highlight agents involved in any violation. Gray edges correspond to structural links that did not participate in the matching process. Consequently, the new visualization mirrors the local consistency mechanism used in the scoring function, while the earlier (strict) visualization instead emphasized global exclusivity---treating any reuse of an agent in multiple pairs as a conflict and reducing the score accordingly.
    """
    # --- Build graph ---
    G = nx.Graph()
    nodes = result_data.get("graph", {}).get("nodes", [])
    id_to_name = {n["id"]: n["name"] for n in nodes}
    name_to_id = {v: k for k, v in id_to_name.items()}
    for i in range(len(nodes)):
        G.add_node(i, name=id_to_name.get(i, f"Agent {i+1}"))

    if "links" in result_data.get("graph", {}):
        for e in result_data["graph"]["links"]:
            G.add_edge(e["source"], e["target"])

    # --- Directed choices ---
    answers = result_data.get("answers", [])
    choices = {i: name_to_id[a] for i, a in enumerate(answers)
               if a and a.lower() != "none" and a in name_to_id}

    # --- Mutual pairs (green) ---
    mutual_pairs = {tuple(sorted((a, b))) for a, b in choices.items()
                    if b in choices and choices[b] == a}
    protected_nodes = {n for e in mutual_pairs for n in e}

    # --- All chosen edges ---
    all_edges = {tuple(sorted((a, b))) for a, b in choices.items()}

    # --- Usage count ---
    usage = {}
    for a, b in all_edges:
        usage[a] = usage.get(a, 0) + 1
        usage[b] = usage.get(b, 0) + 1

    # --- Conflicts & one-sided ---
    matched_edges = mutual_pairs.copy()
    conflict_edges, one_sided_edges = set(), set()
    for a, b in all_edges:
        edge = tuple(sorted((a, b)))
        if edge in matched_edges:
            continue
        if usage[a] > 1 or usage[b] > 1 or b in protected_nodes:
            conflict_edges.add(edge)
        else:
            one_sided_edges.add(edge)

    # --- Layout positions ---
    pos = create_layout_positions(G, layout)

    # --- Prepare coordinates ---
    coords = {"match": ([], []), "conflict": ([], []),
              "one": ([], []), "neutral": ([], [])}

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge = tuple(sorted((u, v)))
        if edge in matched_edges:
            g = "match"
        elif edge in conflict_edges:
            g = "conflict"
        elif edge in one_sided_edges:
            g = "one"
        else:
            g = "neutral"
        xs, ys = coords[g]
        xs += [x0, x1, None]
        ys += [y0, y1, None]

    # --- Build Plotly Figure ---
    fig = go.Figure()

    def add_edges(xs, ys, color, name, dash=None, op=0.9):
        if not xs:
            return
        line = dict(color=color, width=2)
        if dash:
            line["dash"] = dash
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=line, opacity=op, name=name, hoverinfo="none"
        ))

    add_edges(*coords["neutral"], "#B0B0B0", "Other Edges", op=0.25)
    add_edges(*coords["match"], "#2E8B57", "Matched Pairs")
    add_edges(*coords["conflict"], "#DC143C", "Conflicts")
    add_edges(*coords["one"], "#FFA500", "One-Sided Choices", dash="dot")

    # --- Node colors ---
    matched_nodes = {n for e in matched_edges for n in e}
    conflict_nodes = {n for e in conflict_edges for n in e}
    one_nodes = {n for e in one_sided_edges for n in e}

    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    colors = []
    for n in G.nodes():
        if n in matched_nodes:
            colors.append("#2E8B57")
        elif n in conflict_nodes:
            colors.append("#DC143C")
        elif n in one_nodes:
            colors.append("#FFA500")
        else:
            colors.append("#C0C0C0")

    hover = [f"{G.nodes[n]['name']} → {answers[n] if n < len(answers) else 'None'}"
             for n in G.nodes()]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=1, color="black")),
        textposition="top center",
        hovertext=hover, hoverinfo="text",
        showlegend=False
    ))

    # --- Layout Settings ---
    fig.update_layout(
        title=f"Matching Task - {result_data.get('graph_generator','Unknown').upper()} Topology<br>"
              f"Score: {result_data.get('score',0):.2f} | Success: {result_data.get('successful', False)}",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.8, y=1.25)
    )
    return fig


# ------------------- Generate LaTeX Figure Block -------------------
def generate_latex_figure_group(image_dict, caption="Matching experiment outcomes across different network sizes ($n=4$ to $100$). Each subfigure shows the final pair assignments of agents. Valid mutual pairs are shown in green, conflicts in red, and one-sided selections in orange."):
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
            # Ensure consistent figure path (replace 'results/' → 'figures/')
            fig_path = str(path).replace("results/", "figures/").replace("\\", "/")
            fig_path = str(fig_path).replace("ontology/", "topology/").replace("\\", "/")
            # Safely pick sub-figure width
            width = sub_fig_size.get(n, "0.20")  # fallback width
            label = label_map.get(topo, topo.upper())
            latex.append(
                f"    \\subfloat[{label} ($n={n}$)]{{"
                f"\\includegraphics[trim=100 100 90 100,clip,width={width}\\textwidth]{{{fig_path}}}}}"
            )
        latex.append("    \\\\")
        latex.append("    \\vspace{2mm}")


    latex.append(f"    \\caption{{{caption}}}")
    latex.append("    \\label{fig:matching-results}")
    latex.append("\\end{figure*}")
    return "\n".join(latex)


# ------------------- Main Entry -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize matching experiment results and generate LaTeX block"
    )
    parser.add_argument("files", nargs="+", help="Matching result JSON file(s) to visualize")
    parser.add_argument(
        "--layout",
        choices=["spring", "circular", "kamada", "spectral", "community", "auto"],
        default="spring",
        help="Network layout algorithm"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Plotly visualization")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX snippet for all figures")

    # 🆕 Added flag for using the original (strict) matching visualization
    parser.add_argument(
        "--original",
        action="store_true",
        help="Use the original matching visualization logic (strict conflict interpretation)"
    )

    args = parser.parse_args()

    generated = {}  # {graph_type: [(n, path)]}

    for filepath in args.files:
        result_data = load_result_file(filepath)
        layout = pick_layout(result_data, args.layout)

        # 🧩 Select which visualization logic to use
        if args.original:
            fig = create_matching_network_original_code(result_data, layout=layout)
        else:
            fig = create_matching_network(result_data, layout=layout)

        p = Path(filepath)
        framework = result_data.get("framework", "unknown").lower()

        # 🔄 For Concordia framework, aggregate all under one category
        if framework == "concordia":
            graph_type = "all"
        else:
            graph_type = result_data.get("graph_generator", "unknown").lower()

        num_nodes = result_data.get("num_nodes", 0)
        out_dir = p.parent

        # 🧾 File naming: add "_matching_original" when using the old method
        suffix = "_matching_original" if args.original else "_matching"
        output_path = out_dir / f"{p.stem}_{framework}{suffix}.png"

        fig.write_image(str(output_path))
        print(f"✅ Saved {output_path}")

        generated.setdefault(graph_type, []).append((num_nodes, output_path))

    # 📄 Optionally generate LaTeX snippet for all results
    if args.latex:
        latex_code = generate_latex_figure_group(generated)
        tex_file = Path("results/ontology/matching_enhanced_figures"+suffix+".tex")
        tex_file.write_text(latex_code)
        print(f"📄 LaTeX code written to {tex_file}")



if __name__ == "__main__":
    main()
