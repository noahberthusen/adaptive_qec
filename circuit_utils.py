import numpy as np
from edge_coloring import edge_color_bipartite
import stim
import networkx as nx
from classical_code import *
from quantum_code import *
import matplotlib.pyplot as plt
from itertools import chain

def _create_bipartite_graph_from_parity_matrix(H, checks):
    G = nx.Graph()
    m, n = H.shape
    G.add_nodes_from(range(n), bipartite=1)
    G.add_nodes_from(checks, bipartite=0)

    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                G.add_edge(j, checks[i])
    return G


def generate_hgp_circuit(H, checks, stab_type, p):
    G = _create_bipartite_graph_from_parity_matrix(H, checks)
    coloring = edge_color_bipartite(G)

    c = stim.Circuit()

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p/10)

    for r in coloring:
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p)
        c.append("DEPOLARIZE1", data_qbts, p/10)

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p/10)
    return c

def generate_422_hgp_circuit(H, checks, stab_type, p):
    # should pass only concatenated HGP generators in checks
    cumsum = np.cumsum(H, axis=1)
    occurrence_type = cumsum % 2
    c = stim.Circuit()

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p/10)

    # measures first qubit of each Iceberg logical
    mask = (occurrence_type == 0)
    mask_H = H * ~mask
    G = _create_bipartite_graph_from_parity_matrix(mask_H, checks)
    coloring = edge_color_bipartite(G)
    for i, r in enumerate(coloring):
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p)
        c.append("DEPOLARIZE1", data_qbts, p/10)

    # measures second qubit of each Iceberg logical
    mask = (occurrence_type == 1)
    mask_H = H * ~mask
    G = _create_bipartite_graph_from_parity_matrix(mask_H, checks)
    coloring = edge_color_bipartite(G)

    for i, r in enumerate(coloring):
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p)
        c.append("DEPOLARIZE1", data_qbts, p/10)

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p/10)

    return c

def generate_422_circuit(H, x_checks, z_checks, p):
    # only pass in Iceberg generators
    # I think this only works for the Z basis
    data_qbts = np.arange(H.shape[1]) # should be divisible by 4
    c = stim.Circuit()
    c.append("H", x_checks)
    c.append("DEPOLARIZE1", x_checks, p/10)

    targets = list(chain.from_iterable(zip(x_checks, data_qbts[data_qbts % 4 == 0])))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 0], p/10)
    targets = list(chain.from_iterable(zip(data_qbts[data_qbts % 4 == 0], z_checks)))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 0], p/10)
    targets = list(chain.from_iterable(zip(data_qbts[data_qbts % 4 == 1], z_checks)))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 1], p/10)
    targets = list(chain.from_iterable(zip(x_checks, data_qbts[data_qbts % 4 == 1])))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 1], p/10)


    targets = list(chain.from_iterable(zip(x_checks, data_qbts[data_qbts % 4 == 2])))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 2], p/10)
    targets = list(chain.from_iterable(zip(data_qbts[data_qbts % 4 == 2], z_checks)))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 2], p/10)
    targets = list(chain.from_iterable(zip(data_qbts[data_qbts % 4 == 3], z_checks)))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 3], p/10)
    targets = list(chain.from_iterable(zip(x_checks, data_qbts[data_qbts % 4 == 3])))
    c.append("CNOT", targets)
    c.append("DEPOLARIZE2", targets, p)
    c.append("DEPOLARIZE1", data_qbts[data_qbts % 4 != 3], p/10)

    c.append("H", x_checks)
    c.append("DEPOLARIZE1", x_checks, p/10)

    return c


if __name__ == "__main__":
    hgp_qcode = read_qcode("./codes/qcodes/surface/HGP_13_1/HGP_13_1.qcode")
    qcode = hgp_qcode

    hgp_Hx, hgp_Hz, hgp_Lx, hgp_Lz, _ = hgp_qcode.to_numpy()
    Hx, Hz, Lx, Lz, mapping = qcode.to_numpy()

    cn = qcode.n
    cmx = qcode.xm
    cmz = qcode.zm
    data_qbts = np.arange(cn)
    x_checks = np.arange(cn,cn+cmx)
    z_checks = np.arange(cn+cmx,cn+cmx+cmz)

    c = generate_422_circuit(Hx[:qcode.qedxm], x_checks[:qcode.qedxm], z_checks[:qcode.qedxm], 0.001)

    G = _create_bipartite_graph_from_parity_matrix(Hx[qcode.qedxm:], x_checks[qcode.qedxm:])
    coloring = edge_color_bipartite(G)
    print(len(coloring))
    # c = generate_422_hgp_circuit(Hx[qcode.qedxm:], x_checks[qcode.qedxm:], True, 0.001)

    # with open("tmp.txt", "w+") as f:
    #     f.write(str(c))