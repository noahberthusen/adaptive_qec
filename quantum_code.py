import numpy as np
from utils import get_logicals

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

    zL = get_logicals(Hx, Hz, False)
    xL = get_logicals(Hx, Hz, True)
    k = xL.shape[0]

    xL_inds = [np.where(x)[0] for x in xL]
    zL_inds = [np.where(z)[0] for z in zL]

    return k, Hx_inds, Hz_inds, xL_inds, zL_inds