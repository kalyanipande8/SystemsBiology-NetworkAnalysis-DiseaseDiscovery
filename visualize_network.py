"""Generate a static figure of the PPI network colored by ground-truth
disease module, for the README. Uses matplotlib + networkx's spring layout."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    G = nx.read_edgelist(DATA_DIR / "ppi_network.edgelist")
    with open(DATA_DIR / "ground_truth_modules.json") as f:
        modules = json.load(f)

    gene_to_module = {}
    for i, (phenotype, genes) in enumerate(modules.items()):
        for g in genes:
            gene_to_module[g] = i

    colors = [gene_to_module.get(n, -1) for n in G.nodes()]
    sizes = [40 if gene_to_module.get(n, -1) != -1 else 8 for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=0.25)

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                            cmap="tab20", linewidths=0)
    ax.set_title("Synthetic PPI network: 600 nodes, 12 seeded disease modules "
                 "(gray = background/unassigned)")
    ax.axis("off")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / "network_modules.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'network_modules.png'}")


if __name__ == "__main__":
    main()
