# mpc-triangle-count-exact-approx-simulations
These are simulations of the algorithms found in Sections 4 and 5 of https://arxiv.org/abs/2002.08299. We simulate running these algorithms in the MPC model and test our algorithms using graphs from the Stanford Large Network Dataset Collection (https://snap.stanford.edu/data/). 

exact_deg_counting.py: Simulates our exact algorithm provided in Section 5 of our paper. Returns the number of MPC rounds necessary to exactly count the number of triangles versus the amount of space per machine. Compares against the baseline algorithm of removing vertices up to degree equal to the degeneracy of the graph.

estimation_simulation.py: Simulates our approximation algorithm provided in Section 4 of our paper. Returns the approximation factor of our algorithm versus the amount of space per machine. Compares against our implementation of the partition algorithm.
