"""Draw a configured project topology and print its router-to-router links."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from presto.config import load_config
from presto.seeding import make_rng
from presto.sim.topology import load_topology


def main() -> None:
    config = load_config("configs/scenario/steady.yaml")
    graph = load_topology(config, make_rng(0, "topology"))

    positions = nx.spring_layout(graph, seed=7)
    figure, axis = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(graph, positions, ax=axis, width=2, edge_color="#607D9B")
    nx.draw_networkx_nodes(
        graph, positions, ax=axis, node_size=900, node_color="#1F6E8C",
        edgecolors="white", linewidths=2,
    )
    nx.draw_networkx_labels(
        graph, positions, ax=axis, labels={n: str(n) for n in graph.nodes},
        font_color="white", font_size=11, font_weight="bold",
    )
    axis.set_title(f"{graph.number_of_nodes()} routers, {graph.number_of_edges()} links")
    axis.axis("off")
    figure.tight_layout()

    output = Path("results/figures/nsfnet_topology.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Topology: {config['topology']['name']}")
    print(f"Routers: {graph.number_of_nodes()} | Links: {graph.number_of_edges()}")
    print("Links (router pairs):")
    for u, v, data in sorted(graph.edges(data=True), key=lambda row: row[2]["link_id"]):
        print(f"  link {data['link_id']:02d}: router {u} -- router {v}")
    print(f"Diagram saved to: {output}")


if __name__ == "__main__":
    main()
