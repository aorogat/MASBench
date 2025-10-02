"""
Graph layout utilities using NetworkX.
"""

import networkx as nx
from typing import Dict
from networkx.algorithms import community


import math
import networkx as nx
from typing import Dict

def spiral_layout(G: nx.Graph, scale: float = 1.0, spacing: float = 0.5) -> Dict:
    """Place nodes in a spiral layout (progressive outward circle)."""
    pos = {}
    n = len(G.nodes())
    for i, node in enumerate(G.nodes()):
        angle = i * spacing
        radius = scale * (i / n)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pos[node] = (x, y)
    return pos

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
    elif layout_type == "spiral":
        return spiral_layout(G)
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

    for i, comm in enumerate(comms):
        subgraph = G.subgraph(comm)

        # spring_layout with higher k -> more spread inside community
        sub_pos = nx.spring_layout(subgraph, seed=42, k=k_inside)

        # Place communities in grid
        x_shift = (i % 3) * offset   # 3 groups per row
        y_shift = (i // 3) * offset
        for node, coords in sub_pos.items():
            pos[node] = (coords[0] + x_shift, coords[1] + y_shift)

    return pos
