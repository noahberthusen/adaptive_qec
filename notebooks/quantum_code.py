import numpy as np
from bposd.css import css_code
from classical_code import *

class QuantumCode:
    def __init__(self, n, k, xm, zm, qedxm, qedzm, Hx, Hz, Lx, Lz):
        self.n = n
        self.k = k
        self.xm = xm
        self.zm = zm
        self.qedxm = qedxm # expected that first qedxm gens are for qed
        self.qedzm = qedzm # ''                  qedzm
        self.Hx = Hx
        self.Hz = Hz
        self.Lx = Lx
        self.Lz = Lz

    def to_numpy(self):
        Hx = np.zeros(shape=(self.xm, self.n), dtype=int)
        for i, xgen in enumerate(self.Hx):
            for qbt in xgen:
                Hx[i][qbt] = 1
        Hz = np.zeros(shape=(self.zm, self.n), dtype=int)
        for i, zgen in enumerate(self.Hz):
            for qbt in zgen:
                Hz[i][qbt] = 1
        Lx = np.zeros(shape=(self.k, self.n), dtype=int)
        for i, xlog in enumerate(self.Lx):
            for qbt in xlog:
                Lx[i][qbt] = 1
        Lz = np.zeros(shape=(self.k, self.n), dtype=int)
        for i, zlog in enumerate(self.Lz):
            for qbt in zlog:
                Lz[i][qbt] = 1

        return Hx, Hz, Lx, Lz


def read_qcode(f_name):
    with open(f_name, 'r') as f:
        n = int(f.readline().split(',')[1])
        k = int(f.readline().split(',')[1])
        xm = int(f.readline().split(',')[1])
        zm = int(f.readline().split(',')[1])
        qedxm = int(f.readline().split(',')[1])
        qedzm = int(f.readline().split(',')[1])
        f.readline()

        Hx = []
        for i in range(xm):
            supp = [int(c) for c in f.readline().strip(',\n').split(',')]
            Hx.append(supp)
        f.readline()
        Hz = []
        for i in range(zm):
            supp = [int(c) for c in f.readline().strip(',\n').split(',')]
            Hz.append(supp)
        f.readline()

        Lx = []
        for i in range(k):
            supp = [int(c) for c in f.readline().strip(',\n').split(',')]
            Lx.append(supp)
        f.readline()
        Lz = []
        for i in range(k):
            supp = [int(c) for c in f.readline().strip(',\n').split(',')]
            Lz.append(supp)

    return QuantumCode(n, k, xm, zm, qedxm, qedzm, Hx, Hz, Lx, Lz)


def write_qcode(f_name, qcode: QuantumCode):
    with open(f_name, 'w') as f:
        f.write(f'n,{qcode.n}\n')
        f.write(f'k,{qcode.k}\n')
        f.write(f'xm,{qcode.xm}\n')
        f.write(f'zm,{qcode.zm}\n')
        f.write(f'qedxm,{qcode.qedxm}\n')
        f.write(f'qedzm,{qcode.qedzm}\n')

        f.write('Hx\n')
        for xgen in qcode.Hx:
            for qbt in xgen:
                f.write(str(qbt))
                f.write(',')
            f.write('\n')
        f.write(f'Hz\n')
        for zgen in qcode.Hz:
            for qbt in zgen:
                f.write(str(qbt))
                f.write(',')
            f.write('\n')

        f.write(f'Lx\n')
        for xlog in qcode.Lx:
            for qbt in xlog:
                f.write(str(qbt))
                f.write(',')
            f.write('\n')
        f.write(f'Lz\n')
        for zlog in qcode.Lz:
            for qbt in zlog:
                f.write(str(qbt))
                f.write(',')
            f.write('\n')


def format_qcode(Hx, Hz):
    Hx_inds = [np.where(Hx[i])[0] for i in range(Hx.shape[0])]
    Hz_inds = [np.where(Hz[i])[0] for i in range(Hz.shape[0])]

    qcode = css_code(Hx, Hz)
    xL, zL = qcode.compute_logicals()
    k = xL.shape[0]

    xL_inds = [np.where(x)[0] for x in xL]
    zL_inds = [np.where(z)[0] for z in zL]

    return k, Hx_inds, Hz_inds, xL_inds, zL_inds


def hgp(ccode: ClassicalCode, fpath):
    H = ccode.to_numpy()
    dim0, dim1 = H.shape

    I1 = np.eye(dim1, dtype=int)
    I0 = np.eye(dim0, dtype=int)

    hx1 = np.kron(H, I1)
    hx2 = np.kron(I0, H.T)
    HGPHx = np.hstack([hx1, hx2])

    hz1 = np.kron(I1, H)
    hz2 = np.kron(H.T, I0)
    HGPHz = np.hstack([hz1, hz2])

    m, n = HGPHx.shape
    Hx_inds = [np.where(HGPHx[i])[0] for i in range(HGPHx.shape[0])]
    Hz_inds = [np.where(HGPHz[i])[0] for i in range(HGPHz.shape[0])]

    k, Hx_inds, Hz_inds, xL_inds, zL_inds = format_qcode(HGPHx, HGPHz)

    qcode = QuantumCode(n, k, HGPHx.shape[0], HGPHx.shape[0],
                        0, 0,
                        Hx_inds, Hz_inds, xL_inds, zL_inds)
    write_qcode(fpath+ f"/HGP_{n}_{k}.qcode", qcode)


def concatenate_iceberg(qcode: QuantumCode, ibn, fpath):
    def iceberglogicals(n):
        icebergX = np.zeros(shape=(n-2,2), dtype=int)
        icebergZ = np.zeros(shape=(n-2,2), dtype=int)

        for i in range(n-2):
            icebergX[i] = np.array([0,i+1])
            icebergZ[i] = np.array([i+1,n-1])

        return icebergX, icebergZ

    iceberg642Xlogicals = np.array([
        [1,2],
        [0,1],
        [4,5],
        [3,4],
    ])

    iceberg642Zlogicals = np.array([
        [0,1],
        [1,2],
        [3,4],
        [4,5]
    ])


    ibk = ibn-2
    icebergX = np.ones(ibn, dtype=int)
    # icebergZ = np.ones(ibn, dtype=int)

    if (ibn == 6):
        icebergXlogicals, icebergZlogicals = iceberg642Xlogicals, iceberg642Zlogicals
    else:
        icebergXlogicals, icebergZlogicals = iceberglogicals(ibn)

    Hx, Hz, Lx, Lz = qcode.to_numpy()
    concatenatedStabilizersQED = np.kron(np.eye(Hx.shape[1]//ibk, dtype=int), icebergX) # ibk | Hx.shape[1] required

    concatenatedStabilizersXQEC = np.zeros(shape=(Hx.shape[0], concatenatedStabilizersQED.shape[1]), dtype=int)
    concatenatedStabilizersZQEC = np.zeros(shape=(Hz.shape[0], concatenatedStabilizersQED.shape[1]), dtype=int)

    for i, r in enumerate(Hx):
        for x in np.where(r)[0]:
            concatenatedStabilizersXQEC[i][icebergXlogicals[x%ibk]+(ibn*(x//ibk))] ^= 1

    for i, r in enumerate(Hz):
        for z in np.where(r)[0]:
            concatenatedStabilizersZQEC[i][icebergZlogicals[z%ibk]+(ibn*(z//ibk))] ^= 1

    concatenatedHx = np.vstack([concatenatedStabilizersXQEC, concatenatedStabilizersQED][::-1])
    concatenatedHz = np.vstack([concatenatedStabilizersZQEC, concatenatedStabilizersQED][::-1])

    concatenatedxL = np.zeros(shape=(Lx.shape[0], concatenatedStabilizersQED.shape[1]), dtype=int)
    concatenatedzL = np.zeros(shape=(Lz.shape[0], concatenatedStabilizersQED.shape[1]), dtype=int)

    for i, r in enumerate(Lx):
        for x in np.where(r)[0]:
            concatenatedxL[i][icebergXlogicals[x%ibk]+(ibn*(x//ibk))] ^= 1

    for i, r in enumerate(Lz):
        for z in np.where(r)[0]:
            concatenatedzL[i][icebergZlogicals[z%ibk]+(ibn*(z//ibk))] ^= 1

    xL_inds = [np.where(x)[0] for x in concatenatedxL]
    zL_inds = [np.where(z)[0] for z in concatenatedzL]

    m, n = concatenatedHx.shape
    k = len(xL_inds)

    Hx_inds = [np.where(concatenatedHx[i])[0] for i in range(concatenatedHx.shape[0])]
    Hz_inds = [np.where(concatenatedHz[i])[0] for i in range(concatenatedHz.shape[0])]

    qcode = QuantumCode(n, k, concatenatedHx.shape[0], concatenatedHz.shape[0],
                        concatenatedStabilizersQED.shape[0], concatenatedStabilizersQED.shape[0],
                        Hx_inds, Hz_inds, xL_inds, zL_inds)
    write_qcode(fpath + f"/HGP_C{ibn}{ibk}2_{n}_{k}.qcode", qcode)