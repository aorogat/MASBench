#!/usr/bin/env python3
"""
Consensus Network Visualization

This module is the entry point for visualizing consensus experiment results.

Run with:

# Run on one file
python -m multi_agent.topology.visualization.consensus_main results/ontology/consensus_results_20250917_230934_rounds3_gpt-4o-mini_nodes4.json --interactive --layout auto

# Run on all consensus files (each JSON → HTML with same name)
python -m multi_agent.topology.visualization.consensus_main results/ontology/consensus_*.json --interactive --layout auto
"""

import argparse
from pathlib import Path

from .loader import load_result_file
from .static_plot import visualize_network
from .interactive_plot import create_interactive_network
from .analysis import analyze_topology_properties

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
        return "circular"
    elif topo in ["ba", "scale-free"]:  # Barabási–Albert scale-free
        return "spring"
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


def main():
    parser = argparse.ArgumentParser(description="Visualize consensus experiment results")
    parser.add_argument("files", nargs="+", help="Consensus result JSON file(s) to visualize")
    parser.add_argument(
        "--layout",
        choices=["spring", "circular", "kamada", "spectral", "community", "auto"],
        default="spring",
        help="Network layout algorithm (or 'auto' to choose based on topology)",
    )
    parser.add_argument("--analyze", action="store_true", help="Print network topology analysis")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Plotly visualization")

    args = parser.parse_args()

    for filepath in args.files:
        result_data = load_result_file(filepath)
        layout = pick_layout(result_data, args.layout)

        if args.analyze:
            props = analyze_topology_properties(result_data)
            print(f"\n=== Analysis for {Path(filepath).name} ===")
            for k, v in props.items():
                print(f"{k}: {v}")

        if args.interactive:
            fig = create_interactive_network(result_data, layout=layout)

            # Always save with same name, .html extension
            output_path = Path(filepath).with_suffix(".html")
            fig.write_html(output_path)
            print(f"✅ Saved interactive visualization to {output_path} (layout={layout})")
        else:
            # Default to PNG with same name
            output_path = Path(filepath).with_suffix(".png")
            fig = visualize_network(result_data, layout=layout, save_path=output_path, show_plot=False)
            print(f"✅ Saved static plot to {output_path} (layout={layout})")


if __name__ == "__main__":
    main()
