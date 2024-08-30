from classical_code import *
import numpy as np
import os

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

n = 60
dv = 3 # w_c. Every bit is in this many checks
dc = 4 # w_r. Every check has this many bits in it
m = (n*dv)//dc
k = n - m

vs = np.array([[j for i in range(dv)] for j in range(n)]).flatten()
cs = np.array([[j for i in range(dc)] for j in range(m)]).flatten()

H = np.zeros((m, n), dtype=bool)

while (vs.size and cs.size):
    # choose random 'stub' from each array
    double_edge = True
    while(double_edge):
        v_ind = np.random.randint(0, len(vs))
        c_ind = np.random.randint(0, len(cs))

        if (H[cs[c_ind]][vs[v_ind]] != 1):
            double_edge = False
            H[cs[c_ind]][vs[v_ind]] = 1
            vs = np.delete(vs, v_ind)
            cs =np.delete(cs, c_ind)

# H = np.loadtxt('./ldpc_codes/16_4_3.txt', dtype=np.int8)

f_name = os.path.join(path, f'./codes/{n}_{m}_{dv}_{dc}.code')

bit_nbhd = []
for bit in range(n):
    checks = np.where(H[:, bit])[0]
    bit_nbhd.append(checks)

check_nbhd = []
for check in range(m):
    bits = np.where(H[check])[0]
    check_nbhd.append(bits)

ccode = ClassicalCode(n, m, dv, dc, bit_nbhd, check_nbhd)
write_code(f_name, ccode)