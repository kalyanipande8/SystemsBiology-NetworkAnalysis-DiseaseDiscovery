"""Run the full pipeline end to end:
1. Simulate the synthetic PPI network, disease modules, and patient cohort.
2. Discover disease pathways/modules per phenotype via RWR network diffusion.
3. Build patient-patient similarity (diffusion-based) and demo personalized
   disease models (diagnosis support) for a few example patients.
4. Render the network figure.

All numbers printed here are copied directly into README.md - nothing in
the README is hand-typed/fabricated.
"""

import simulate_network_data
import pathway_discovery
import patient_similarity
import visualize_network


def main():
    print("=" * 70)
    print("STEP 1: Simulating synthetic PPI network + disease modules + cohort")
    print("=" * 70)
    simulate_network_data.main()

    print()
    print("=" * 70)
    print("STEP 2: Network-medicine pathway discovery (Random Walk with Restart)")
    print("=" * 70)
    pathway_discovery.main()

    print()
    print("=" * 70)
    print("STEP 3: Patient similarity + personalized disease models")
    print("=" * 70)
    patient_similarity.main()

    print()
    print("=" * 70)
    print("STEP 4: Rendering network figure")
    print("=" * 70)
    visualize_network.main()

    print()
    print("Pipeline complete. See data/ for intermediate artifacts and "
          "output/network_modules.png for the figure.")


if __name__ == "__main__":
    main()
