"""
Network-medicine pathway/disease-module discovery.

Real algorithm: Random Walk with Restart (RWR), a standard network-propagation
/ "network diffusion" method used in network medicine (e.g. Kohler et al.
2008 "Walking the interactome for prioritization of candidate disease genes";
Cowen et al. 2017 "Network propagation" review) to identify disease modules
from a seed set of perturbed genes.

For each phenotype, the seed set is the UNION of perturbed genes across all
patients diagnosed with that phenotype (this is the only information the
algorithm is given -- it does NOT see the ground-truth module). RWR then
diffuses that signal across the real network topology and the top-scoring
nodes are called the discovered disease module / pathway for that phenotype.

We then benchmark discovered modules against the injected ground-truth
modules (Jaccard overlap) purely for validation/reporting -- the discovery
algorithm itself never uses ground truth.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RESTART_PROB = 0.5     # r: probability of teleporting back to seed set each step
MAX_ITER = 200
TOL = 1e-8
TOP_FRACTION = 0.06    # fraction of network nodes retained as the discovered module


def load_network():
    G = nx.read_edgelist(DATA_DIR / "ppi_network.edgelist")
    return G


def load_patients():
    with open(DATA_DIR / "patients.json") as f:
        return json.load(f)


def random_walk_restart(G, nodelist, adj_norm, seed_genes, restart_prob=RESTART_PROB):
    """Run RWR from a seed set over a pre-column-normalized adjacency matrix.
    p_{t+1} = (1-r) * W @ p_t + r * p0
    Returns a dict {gene: steady-state visitation probability}.
    """
    n = len(nodelist)
    idx = {g: i for i, g in enumerate(nodelist)}
    p0 = np.zeros(n)
    valid_seeds = [g for g in seed_genes if g in idx]
    if not valid_seeds:
        return {g: 0.0 for g in nodelist}
    for g in valid_seeds:
        p0[idx[g]] = 1.0 / len(valid_seeds)

    p = p0.copy()
    for _ in range(MAX_ITER):
        p_next = (1 - restart_prob) * (adj_norm @ p) + restart_prob * p0
        if np.linalg.norm(p_next - p, 1) < TOL:
            p = p_next
            break
        p = p_next
    return {g: p[idx[g]] for g in nodelist}


def build_column_normalized_adjacency(G, nodelist):
    A = nx.to_numpy_array(G, nodelist=nodelist)
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    W = A / col_sums  # column-normalize so each column sums to 1 (transition matrix)
    return W


def discover_pathways(G, patients, top_fraction=TOP_FRACTION):
    """Run RWR per phenotype (seeded from that phenotype's pooled patient
    perturbation profiles) and return discovered modules."""
    nodelist = list(G.nodes())
    W = build_column_normalized_adjacency(G, nodelist)

    phenotypes = sorted(set(p["true_phenotype"] for p in patients))
    discovered = {}
    scores_by_phenotype = {}
    n_top = max(5, int(len(nodelist) * top_fraction))

    for phenotype in phenotypes:
        seed_genes = set()
        for p in patients:
            if p["true_phenotype"] == phenotype:
                seed_genes.update(p["perturbed_genes"])
        scores = random_walk_restart(G, nodelist, W, seed_genes)
        scores_by_phenotype[phenotype] = scores
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        module = [g for g, _ in ranked[:n_top]]
        discovered[phenotype] = module

    return discovered, scores_by_phenotype


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def evaluate_against_ground_truth(discovered):
    with open(DATA_DIR / "ground_truth_modules.json") as f:
        ground_truth = json.load(f)

    results = {}
    for phenotype, module in discovered.items():
        gt = ground_truth[phenotype]
        results[phenotype] = {
            "jaccard": jaccard(module, gt),
            "overlap": len(set(module) & set(gt)),
            "discovered_size": len(module),
            "ground_truth_size": len(gt),
        }
    return results


def main():
    G = load_network()
    patients = load_patients()
    discovered, _ = discover_pathways(G, patients)
    eval_results = evaluate_against_ground_truth(discovered)

    print(f"Discovered {len(discovered)} disease-associated pathways/modules via RWR diffusion")
    mean_jaccard = np.mean([r["jaccard"] for r in eval_results.values()])
    print(f"Mean Jaccard overlap vs ground-truth modules: {mean_jaccard:.3f}")
    for phenotype, r in eval_results.items():
        print(f"  {phenotype:38s} jaccard={r['jaccard']:.2f}  "
              f"overlap={r['overlap']}/{r['ground_truth_size']}")

    with open(DATA_DIR / "discovered_pathways.json", "w") as f:
        json.dump(discovered, f, indent=2)

    return discovered, eval_results


if __name__ == "__main__":
    main()
