"""
Sample-similarity analysis and personalized disease models.

Real algorithm: for each patient, run Random Walk with Restart (same
propagation operator as pathway_discovery.py) seeded from that patient's own
perturbed-gene set. This "smooths" each patient's raw gene list over the
network topology into a dense diffusion profile -- two patients with
perturbations in different genes of the *same* pathway end up with similar
diffusion profiles, which a raw Jaccard-on-gene-sets would miss. This is the
same network-propagation-based patient stratification idea used in tools
like NetICS / Hofree et al. 2013 "Network-based stratification of tumor
mutations."

Patient-patient similarity = cosine similarity of diffusion profiles.
We also report plain Jaccard-on-perturbed-genes for comparison.

"Personalized disease model": for a query patient, find the k most similar
patients, and report the pathway(s) shared among that neighborhood -- a
diagnosis-support style output for rare-disease cases where the patient's
own perturbed genes may be too sparse/atypical to diagnose in isolation.
"""

import json
from pathlib import Path

import numpy as np

from pathway_discovery import (
    load_network,
    build_column_normalized_adjacency,
    random_walk_restart,
    RESTART_PROB,
)

DATA_DIR = Path(__file__).parent / "data"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_diffusion_profiles(G, patients):
    nodelist = list(G.nodes())
    W = build_column_normalized_adjacency(G, nodelist)
    profiles = {}
    for p in patients:
        scores = random_walk_restart(G, nodelist, W, p["perturbed_genes"], restart_prob=RESTART_PROB)
        profiles[p["patient_id"]] = np.array([scores[g] for g in nodelist])
    return profiles, nodelist


def cosine_similarity(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def build_similarity_matrix(patients, profiles):
    ids = [p["patient_id"] for p in patients]
    n = len(ids)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = cosine_similarity(profiles[ids[i]], profiles[ids[j]])
            sim[i, j] = s
            sim[j, i] = s
    return ids, sim


def most_similar(query_id, ids, sim, k=5):
    i = ids.index(query_id)
    scores = [(ids[j], sim[i, j]) for j in range(len(ids)) if j != i]
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:k]


def personalized_disease_model(query_id, patients, ids, sim, discovered_pathways, k=5):
    """Diagnosis-support output: nearest neighbors + their phenotypes +
    the discovered pathway(s) shared with the query patient's own perturbed genes."""
    by_id = {p["patient_id"]: p for p in patients}
    query = by_id[query_id]
    neighbors = most_similar(query_id, ids, sim, k=k)

    neighbor_phenotypes = [by_id[nid]["true_phenotype"] for nid, _ in neighbors]
    # majority vote among neighbors = suggested phenotype
    from collections import Counter
    vote = Counter(neighbor_phenotypes)
    suggested_phenotype, votes = vote.most_common(1)[0]

    query_genes = set(query["perturbed_genes"])
    shared_pathway_genes = set(discovered_pathways.get(suggested_phenotype, [])) & query_genes

    return {
        "patient_id": query_id,
        "true_phenotype": query["true_phenotype"],
        "suggested_phenotype": suggested_phenotype,
        "suggestion_confidence": votes / k,
        "nearest_neighbors": [{"patient_id": nid, "phenotype": by_id[nid]["true_phenotype"],
                                "similarity": round(s, 4)} for nid, s in neighbors],
        "shared_pathway_genes_with_suggestion": sorted(shared_pathway_genes),
    }


def sanity_check_identical_patients(G):
    """Construct two synthetic patients with IDENTICAL perturbation profiles
    and confirm the similarity method assigns them similarity == 1.0 and
    ranks each other as the top match. Used both here and in the test suite."""
    nodelist = list(G.nodes())
    sample_genes = list(nodelist)[:6]
    twin_a = {"patient_id": "TWIN_A", "true_phenotype": "test", "perturbed_genes": sample_genes}
    twin_b = {"patient_id": "TWIN_B", "true_phenotype": "test", "perturbed_genes": sample_genes}
    decoy = {"patient_id": "DECOY", "true_phenotype": "test", "perturbed_genes": list(nodelist)[200:210]}
    patients = [twin_a, twin_b, decoy]
    profiles, _ = compute_diffusion_profiles(G, patients)
    ids, sim = build_similarity_matrix(patients, profiles)
    sim_ab = sim[ids.index("TWIN_A"), ids.index("TWIN_B")]
    top_match_for_a = most_similar("TWIN_A", ids, sim, k=1)[0][0]
    return sim_ab, top_match_for_a


def main():
    G = load_network()
    with open(DATA_DIR / "patients.json") as f:
        patients = json.load(f)
    with open(DATA_DIR / "discovered_pathways.json") as f:
        discovered_pathways = json.load(f)

    profiles, nodelist = compute_diffusion_profiles(G, patients)
    ids, sim = build_similarity_matrix(patients, profiles)

    np.save(DATA_DIR / "patient_similarity_matrix.npy", sim)
    with open(DATA_DIR / "patient_ids.json", "w") as f:
        json.dump(ids, f)

    # Diagnosis-support demo for a handful of example patients
    example_ids = [patients[0]["patient_id"], patients[26]["patient_id"], patients[80]["patient_id"]]
    demo_results = []
    for pid in example_ids:
        result = personalized_disease_model(pid, patients, ids, sim, discovered_pathways, k=5)
        demo_results.append(result)
        print(f"\nPatient {pid} (true phenotype: {result['true_phenotype']})")
        print(f"  Suggested phenotype (from {5} nearest neighbors): "
              f"{result['suggested_phenotype']} (confidence {result['suggestion_confidence']:.2f})")
        print(f"  Nearest neighbor similarities: {result['nearest_neighbors']}")
        print(f"  Shared pathway genes supporting suggestion: "
              f"{result['shared_pathway_genes_with_suggestion']}")

    # cohort-wide accuracy of majority-vote diagnosis suggestion vs true phenotype
    correct = 0
    for p in patients:
        r = personalized_disease_model(p["patient_id"], patients, ids, sim, discovered_pathways, k=5)
        if r["suggested_phenotype"] == r["true_phenotype"]:
            correct += 1
    accuracy = correct / len(patients)
    print(f"\nCohort-wide nearest-neighbor diagnosis-suggestion accuracy: "
          f"{correct}/{len(patients)} = {accuracy:.3f}")

    sim_ab, top_match = sanity_check_identical_patients(G)
    print(f"\nSanity check (identical-profile synthetic twins): "
          f"similarity={sim_ab:.6f}, top match for TWIN_A = {top_match}")

    with open(DATA_DIR / "diagnosis_demo.json", "w") as f:
        json.dump(demo_results, f, indent=2)

    return ids, sim, demo_results, accuracy


if __name__ == "__main__":
    main()
