"""
Graph layout utilities using NetworkX.
"""

import math
import networkx as nx
from typing import Dict
from networkx.algorithms import community


def create_layout_positions(G: nx.Graph, layout_type: str = "spring", group_assignments=None) -> Dict:
    """Generate node positions using a chosen NetworkX layout."""
    if layout_type == "spring":
        return nx.spring_layout(G, seed=42, k=2)
    elif layout_type == "circular":
        return nx.circular_layout(G)
    elif layout_type == "kamada":
        return nx.kamada_kawai_layout(G)
    elif layout_type == "spectral":
        return nx.spectral_layout(G)
    elif layout_type == "community":
        return community_layout(G)
    else:
        return nx.spring_layout(G, seed=42)


def community_layout(G: nx.Graph, offset: float = 4.0, k_inside: float = 0.8) -> Dict:
    """
    Layout nodes clustered by detected communities.
    - offset: controls distance between communities
    - k_inside: controls spacing inside each community's spring layout
    """
    from networkx.algorithms import community

    comms = list(community.greedy_modularity_communities(G))
    pos = {}

    if not comms:
        return pos

    # Arrange communities in approximately square grid
    per_row = max(1, math.ceil(math.sqrt(len(comms))))

    for i, comm in enumerate(comms):
        subgraph = G.subgraph(comm)

        # spring_layout with higher k -> more spread inside community
        sub_pos = nx.spring_layout(subgraph, seed=42, k=k_inside)

        # Center sublayout around origin to avoid skew when applying offsets
        xs, ys = zip(*sub_pos.values())
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        centered = {
            node: (coords[0] - centroid_x, coords[1] - centroid_y)
            for node, coords in sub_pos.items()
        }

        # Scale clusters based on their size so dense clusters remain readable
        scale = 1 + math.log(len(comm) + 1, 2)

        # Place communities in grid
        x_shift = (i % per_row) * offset
        y_shift = (i // per_row) * offset
        for node, coords in centered.items():
            pos[node] = (
                coords[0] * scale + x_shift,
                coords[1] * scale + y_shift,
            )

    return pos
