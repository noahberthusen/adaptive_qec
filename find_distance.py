from classical_code import read_code
import numpy as np
import os
import galois

codes = [
    "8_6_3_4",
    "12_9_3_4",
    "16_12_3_4",
    "20_15_3_4",
    "24_18_3_4",
    "28_21_3_4",
    "32_24_3_4",
    "40_30_3_4",
    "60_45_3_4"

]
distances = []

def par2gen(H):
    GF = galois.GF(2)
    gfH = GF(H)
    gfH_rank = np.linalg.matrix_rank(gfH)

    rref_H = gfH.row_reduce()

    swaps = []
    col_H = rref_H.copy()
    for i in range(gfH_rank):
        inds = np.where(col_H[i])[0]
        pivot = inds[0]
        col_H[:,[i,pivot]] = col_H[:,[pivot,i]]
        swaps.append((i,pivot))

    col_H = col_H[:gfH_rank]
    col_G = GF(np.hstack([col_H[:,gfH_rank:].T, np.eye(H.shape[1]-gfH_rank, dtype=int)]))

    G = col_G.copy()
    for swap in swaps[::-1]:
        G[:,[swap[1],swap[0]]] = G[:,[swap[0],swap[1]]]

    if (np.any(G @ rref_H[:gfH_rank].T) or np.any(col_G @ col_H.T)):
        print("FAILED")
        return
    return (np.array(G, dtype=int), np.array(col_G, dtype=int))


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

num_iters = 1000
GF = galois.GF(2)
for i, code in enumerate(codes):
    file = os.path.join(path, f'./codes/ccodes/{code}.code')
    ccode = read_code(file)
    d = ccode.n
    n = ccode.n

    H = np.zeros((ccode.m, ccode.n), dtype=int)
    for j, row in enumerate(ccode.check_nbhd):
        for r in row:
            H[j][r] = 1
    k = n - np.linalg.matrix_rank(GF(H))

    hx1 = np.kron(H, np.eye(H.shape[1], dtype=bool))
    hx2 = np.kron(np.eye(H.shape[0], dtype=bool), H.T)
    Hx = np.hstack([hx1, hx2])

    hz1 = np.kron(np.eye(H.shape[1], dtype=bool), H)
    hz2 = np.kron(H.T, np.eye(H.shape[0], dtype=bool))
    Hz = np.hstack([hz1, hz2])

    G, col_G = par2gen(H)

    for j in range(1, 2**k):
        num = bin(j)[2:].zfill(k)
        w = np.array([int(b) for b in num])
        encoded_w = w@G % 2

        # if (np.any(H@encoded_w)):
        #     print('Not a codeword')
        if (np.count_nonzero(encoded_w) < d):
            d = np.count_nonzero(encoded_w)
    qedn = 4
    qedk = 2
    distances.append(d)
    print(f"[[{n},{k},{d}]],\t[[{n**2+(n-k)**2},{k**2},{d}]],\t[[{qedn*(n**2+(n-k)**2)},{qedk*(k**2)},{2*d}]]")

# print(distances)