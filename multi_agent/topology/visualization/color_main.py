#!/usr/bin/env python3
"""
Coloring Network Visualization

This module is the entry point for visualizing coloring experiment results.

Run with:

# Run on one file
python -m multi_agent.topology.visualization.color_main results/ontology/coloring_results_20250917_222432_rounds8_gpt-4o-mini_nodes4.json --interactive --layout auto

# Run on all coloring files (each JSON → HTML with same name)
python -m multi_agent.topology.visualization.color_main results/ontology/coloring_*.json --interactive --layout auto
"""

import argparse
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
from plotly.colors import qualitative

from .loader import load_result_file
from .layouts import create_layout_positions
import math

def spiral_layout(G, scale=1.0, spacing=0.5):
    """Place nodes in a spiral layout."""
    pos = {}
    n = len(G.nodes())
    for i, node in enumerate(G.nodes()):
        angle = i * spacing
        radius = scale * (i / n)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pos[node] = (x, y)
    return pos


def pick_layout(result_data, user_layout: str) -> str:
    """Choose layout based on graph type if user selects 'auto'."""
    if user_layout != "auto":
        return user_layout

    topo = result_data.get("graph_generator", "").lower()
    if topo == "ws":  # Watts–Strogatz small-world
        return "community"
    elif topo in ["ba", "scale-free"]:  # Barabási–Albert scale-free
        return "community"
    elif topo in ["dt", "delaunay"]:  # geometric graphs
        return "kamada"
    elif topo in ["sequential", "crewai-sequential"]:
        # linear chain looks better in spectral layout
        return "spiral"
    elif topo in ["hierarchical", "crewai-hierarchical"]:
        # tree structure → use kamada or spring for readability
        return "kamada"
    else:
        return "spring"  # safe default



def create_coloring_network(result_data, layout: str = "spring") -> go.Figure:
    """
    Create an interactive Plotly network visualization for coloring tasks.
    - Each group = node color
    - Conflicts (neighbors with same color) = red edges
    - Valid agreements (neighbors with different colors) = green edges
    """
    # Build graph
    G = nx.Graph()
    num_nodes = result_data.get("num_nodes", 0)
    for i in range(num_nodes):
        G.add_node(i, name=f"Agent {i+1}")

    if "graph" in result_data:
        for edge in result_data["graph"].get("links", []):
            G.add_edge(edge["source"], edge["target"])

    # Answers (groups)
    answers = result_data.get("answers", [])
    cleaned_answers = ["Unassigned" if a is None or a == "" else str(a) for a in answers]

    node_names = [G.nodes[n]["name"] for n in G.nodes()]

    # Unique groups → color palette
    unique_groups = sorted(set(cleaned_answers))
    palette = qualitative.Set3 + qualitative.Set1 + qualitative.Set2
    group_to_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    node_colors = [group_to_color.get(ans, "#cccccc") for ans in cleaned_answers]

    # Layout
    pos = create_layout_positions(G, layout)

    # Edges
    conflict_x, conflict_y, agree_x, agree_y = [], [], [], []
    for u, v in G.edges():
        u_ans, v_ans = cleaned_answers[u], cleaned_answers[v]
        x0, y0, x1, y1 = pos[u][0], pos[u][1], pos[v][0], pos[v][1]
        if u_ans == v_ans:
            conflict_x += [x0, x1, None]
            conflict_y += [y0, y1, None]
        else:
            agree_x += [x0, x1, None]
            agree_y += [y0, y1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agree_x, y=agree_y,
        mode="lines",
        line=dict(color="#2E8B57", width=1),
        opacity=0.6,
        name="Valid (different groups)",
        hoverinfo="none"
    ))

    fig.add_trace(go.Scatter(
        x=conflict_x, y=conflict_y,
        mode="lines",
        line=dict(color="#DC143C", width=1),
        opacity=0.9,
        name="Conflict (same group)",
        hoverinfo="none"
    ))

    # Nodes
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    hover_text = [f"{name}: {ans}" for name, ans in zip(node_names, cleaned_answers)]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        marker=dict(size=10, color=node_colors, line=dict(width=1, color="black")),
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        title=f"Coloring Task - {result_data.get('graph_generator','Unknown').upper()} Topology<br>"
              f"Score: {result_data.get('score',0):.2f} | "
              f"Success: {result_data.get('successful', False)}",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=True
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize coloring experiment results")
    parser.add_argument("files", nargs="+", help="Coloring result JSON file(s) to visualize")
    parser.add_argument(
        "--layout",
        choices=["spring", "circular", "kamada", "spectral", "community", "auto"],
        default="spring",
        help="Network layout algorithm (or 'auto' to choose based on topology)",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Plotly visualization")

    args = parser.parse_args()

    for filepath in args.files:
        result_data = load_result_file(filepath)
        layout = pick_layout(result_data, args.layout)

        fig = create_coloring_network(result_data, layout=layout)

        if args.interactive:
            output_path = Path(filepath).with_suffix(".html")
            fig.write_html(output_path)
            print(f"✅ Saved interactive visualization to {output_path} (layout={layout})")
        else:
            output_path = Path(filepath).with_suffix(".png")
            fig.write_image(str(output_path))
            print(f"✅ Saved static visualization to {output_path} (layout={layout})")


if __name__ == "__main__":
    main()
