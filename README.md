[![Paper](https://img.shields.io/badge/paper-arXiv%3A2502.14835-B31B1B.svg)](https://arxiv.org/abs/2502.14835)

# Adaptive Syndrome Extraction

[Noah Berthusen](https://noahberthusen.github.io), [Shi Jie Samuel Tan](https://shi-jie-samuel-tan.github.io/), Eric Huang, Daniel Gottesman

### Abstract
Device error rates on current quantum computers have improved enough to where demonstrations of error correction below break-even are now possible. Still, the circuits required for quantum error correction introduce significant overhead and sometimes inject more errors than they correct. In this work, we introduce adaptive syndrome extraction as a scheme to improve code performance and reduce the quantum error correction cycle time by measuring only the stabilizer generators that are likely to provide useful syndrome information. We provide a concrete example of the scheme through the [[4,2,2]] code concatenated with a hypergraph product code and a syndrome extraction cycle that uses quantum error detection to modify the syndrome extraction circuits in real time. Compared to non-concatenated codes and non-adaptive syndrome extraction, we find that the adaptive scheme achieves over an order of magnitude lower logical error rates while requiring fewer CNOT gates and physical qubits. Furthermore, we show how to achieve fault-tolerant universal logical computation with [[4,2,2]]-concatenated hypergraph product codes.

### Description
This repository includes information, code, and data to generate the figures in the paper. To install the required Python packages use `pip install -r requirements/dev`.

### Figures
All the codes used to create the figures in the paper are found in the `/figures/figure_scripts` folder. They are all written in Python, and use the matplotlib library. Files used to draw certain figures in the paper can be found in the `/figures/figure_svgs` folder.
- `plot_circuit_expander.py` Generates Fig. 5.
- `plot_circuit_lacross.py` Generates Fig. 6.
- `plot_overhead.py` Generates Fig. 7.
- `plot_circuit_threshold.py` Generates Fig. 9.
- `plot_single_shot.py` Generates Fig. 10.


### Simulations

#### Circuit-level simulations
Code used to perform the circuit-level simulations of the bivariate bicycle codes can be found in the main `/adaptive_qec/` directory. Listed below are the files and their functions:
- `code_distance.g` [GAP](https://www.gap-system.org/) program which calculates the distance of a CSS code using the [QDistRnd](https://github.com/QEC-pages/QDistRnd) package. The files referenced, `QX.mtx` and `QZ.mtx` are the X and Z parity check matrices of the code in the [Matrix Market](https://networkrepository.com/mtx-matrix-market-format.html) file format.
- `memory_circuit.py` Main driver file for the circuit-level simulations as defined in Section VI of the paper. The user can input whether the syndrome extraction is adaptive--if the chosen QECC is a [[4,2,2]]-concatenated code. Circuit-level simulations powered by [Stim](https://github.com/quantumlib/Stim) are then performed for a user-defined number of rounds.
- `edge_coloring.py` Calculates the optimal edge coloring of a bipartite graph. Used in constructing the circuits and applying accurate idling error. This comes from Chris Pattison's `exp_ldpc` [library](https://github.com/qldpc/exp_ldpc).
- `circuit_utils.py` Generates the noisy circuits for the [[4,2,2,]]-concatenated and non-concatenated codes.
- `result_memory.py` Handles the format and saving of the simulation results.
- `classical_code.py` Handles the reading and writing of classical error correcting codes.
- `quantum_code.py` Handles the reading and writing of quantum error correcting codes.