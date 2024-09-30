import numpy as np
import matplotlib.pyplot as plt
from classical_code import *
from utils import get_logicals
from ldpc import bposd_decoder, bp_decoder
from result import Result, save_new_res
import string, random, argparse


def main(args):
    num_shots = int(args.n)
    qubit_error_rate = float(args.q)
    meas_error_rate = float(args.m)

    alphabet = string.ascii_letters + string.digits
    uuid = ''.join(random.choices(alphabet, k=8))


    ccode = read_code("./codes/16_12_3_4.code")
    H = np.zeros((ccode.m, ccode.n), dtype=int)
    for i in range(ccode.m):
        for j in range(ccode.n):
            if (j in ccode.check_nbhd[i]):
                H[i][j] = 1
    dim0, dim1 = H.shape

    I1 = np.eye(dim1, dtype=int)
    I0 = np.eye(dim0, dtype=int)

    dE21 = np.kron(H, I0)
    dE22 = np.kron(I1, H.T)
    dE2 = np.vstack([dE21, dE22])
    Hz = dE2.T

    dE11 = np.kron(I0, H.T)
    dE12 = np.kron(H, I1)
    dE1 = np.hstack([dE11, dE12])
    Hx = dE1
    m, n = Hx.shape

    zL = get_logicals(Hx, Hz, False)
    xL = get_logicals(Hx, Hz, True)
    k = len(xL)

    xL_inds = [np.where(x)[0] for x in xL]
    zL_inds = [np.where(z)[0] for z in zL]

    bposd_dec = bposd_decoder(
        Hx, # the parity check matrix
        error_rate=qubit_error_rate,
        # channel_probs=channel_probs, #assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=100, #pcm.shape[1], #the maximum number of iterations for BP)
        bp_method="ms",
        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        # osd_order=min(pcm.shape[0],10) #the osd search depth
    )


    rs = []
    for ii in range(1,num_shots+1):
        num_rounds = 0
        residual_error = np.zeros(Hx.shape[1], dtype=int)

        while True:
            num_rounds += 1
            new_qubit_error = np.random.choice([0, 1], size=Hx.shape[1], p=[1-qubit_error_rate, qubit_error_rate])
            new_synd_error = np.random.choice([0, 1], size=Hx.shape[0], p=[1-meas_error_rate, meas_error_rate])
            curr_qubit_error = residual_error ^ new_qubit_error
            curr_synd = ((Hx @ curr_qubit_error) % 2) ^ new_synd_error

            guessed_error = bposd_dec.decode(curr_synd)
            residual_error = curr_qubit_error ^ guessed_error
            obs = [np.count_nonzero(residual_error[l]) % 2 for l in zL_inds]

            if np.any(obs):
                break

        res = Result(0, n, k, qubit_error_rate, meas_error_rate, 1, num_rounds, 0)
        rs.append(res)

        if (ii%100==0):
            save_new_res('./tmp.res', rs)
            rs = []



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1e4, help="Number of shots")
    parser.add_argument('-q', default=0.001, help="Qubit error rate")
    parser.add_argument('-m', default=0.001, help="Measurement error rate")

    args = parser.parse_args()

    main(args)