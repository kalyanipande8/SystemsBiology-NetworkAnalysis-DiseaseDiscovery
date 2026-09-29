"""
Tests for the network-medicine pipeline. Run with: pytest -q

Covers:
- Network structure sanity (node/edge counts, scale-free-ish degree spread).
- Pathway discovery count lands in a reasonable band around the target (~12).
- Injected sanity check: two patients with IDENTICAL perturbation profiles
  must be identified as maximally (or near-maximally) similar to each other.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import simulate_network_data as sim
import pathway_discovery as pd
import patient_similarity as ps

DATA_DIR = ROOT / "data"


@pytest.fixture(scope="module")
def built_pipeline():
    """Ensure pipeline artifacts exist (build them if this is a fresh checkout)."""
    if not (DATA_DIR / "patients.json").exists():
        sim.main()
    if not (DATA_DIR / "discovered_pathways.json").exists():
        pd.main()
    return True


def test_network_structure(built_pipeline):
    G = pd.load_network()
    assert G.number_of_nodes() == sim.N_NODES
    # Barabasi-Albert with m=2 gives at least (n-m)*m edges (before densification adds more)
    min_expected_edges = (sim.N_NODES - sim.BA_M) * sim.BA_M
    assert G.number_of_edges() >= min_expected_edges

    degrees = [d for _, d in G.degree()]
    # scale-free-ish: max degree should be well above the mean (heavy tail),
    # and a majority of nodes should have low degree.
    mean_deg = np.mean(degrees)
    max_deg = max(degrees)
    assert max_deg > 5 * mean_deg
    low_degree_frac = sum(1 for d in degrees if d <= mean_deg) / len(degrees)
    assert low_degree_frac > 0.55


def test_ground_truth_modules(built_pipeline):
    with open(DATA_DIR / "ground_truth_modules.json") as f:
        modules = json.load(f)
    assert len(modules) == sim.N_PHENOTYPES
    for phenotype, genes in modules.items():
        assert sim.MODULE_SIZE_RANGE[0] <= len(genes) <= sim.MODULE_SIZE_RANGE[1] + 2
    # modules should be (largely) disjoint by construction
    all_genes = [g for genes in modules.values() for g in genes]
    assert len(all_genes) == len(set(all_genes))


def test_pathway_discovery_count_in_reasonable_band(built_pipeline):
    with open(DATA_DIR / "discovered_pathways.json") as f:
        discovered = json.load(f)
    # target ~12 pathways (one per phenotype) -- allow a small band
    assert 10 <= len(discovered) <= 14

    eval_results = pd.evaluate_against_ground_truth(discovered)
    mean_jaccard = np.mean([r["jaccard"] for r in eval_results.values()])
    # diffusion-based discovery should recover a genuine, well-above-chance
    # fraction of each true module (chance-level Jaccard here is << 0.05)
    assert mean_jaccard > 0.3


def test_identical_patients_are_maximally_similar():
    """Injected sanity check: construct two patients with identical
    perturbation profiles; they must come out as (near-)maximally similar
    and rank as each other's top match."""
    G = pd.load_network()
    sim_ab, top_match = ps.sanity_check_identical_patients(G)
    assert sim_ab == pytest.approx(1.0, abs=1e-6)
    assert top_match == "TWIN_B"


def test_similarity_matrix_symmetric_and_bounded(built_pipeline):
    G = pd.load_network()
    with open(DATA_DIR / "patients.json") as f:
        patients = json.load(f)
    # subsample for speed
    subset = patients[:20]
    profiles, _ = ps.compute_diffusion_profiles(G, subset)
    ids, sim_matrix = ps.build_similarity_matrix(subset, profiles)

    assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-8)
    assert sim_matrix.max() <= 1.0 + 1e-6
    assert sim_matrix.min() >= -1e-6
    # diagonal should be self-similarity == 1
    assert np.allclose(np.diag(sim_matrix), 1.0, atol=1e-6)


def test_same_phenotype_patients_more_similar_than_different(built_pipeline):
    """Basic biological-plausibility check: patients sharing a true phenotype
    should on average be more similar than patients from different phenotypes."""
    G = pd.load_network()
    with open(DATA_DIR / "patients.json") as f:
        patients = json.load(f)
    subset = [p for p in patients if p["true_phenotype"] in
              (patients[0]["true_phenotype"], patients[-1]["true_phenotype"])]
    profiles, _ = ps.compute_diffusion_profiles(G, subset)
    ids, sim_matrix = ps.build_similarity_matrix(subset, profiles)

    pheno = {p["patient_id"]: p["true_phenotype"] for p in subset}
    same, diff = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = sim_matrix[i, j]
            if pheno[ids[i]] == pheno[ids[j]]:
                same.append(s)
            else:
                diff.append(s)
    assert np.mean(same) > np.mean(diff)
