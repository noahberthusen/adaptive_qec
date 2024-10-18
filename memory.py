import numpy as np
from classical_code import *
from quantum_code import *
from result_memory import save_new_res, Result
from ldpc import bposd_decoder, bp_decoder
import string, random, argparse
from pathlib import Path

def main(args):
    qcode_fname = args.q
    concat = int(args.c)
    adaptive = int(args.a)
    num_shots = int(args.n)
    qubit_error_rate = float(args.e)
    meas_error_rate = float(args.m)
    num_rounds = int(args.r)

    qcode = read_qcode(qcode_fname)
    concat = 1 if qcode.qedxm and concat else 0
    adaptive = 1 if qcode.qedxm and adaptive else 0
    Hx, Hz, Lx, Lz = qcode.to_numpy()

    res_f_name = f"./results/{Path(qcode_fname).name}.res"



    overlapping_x_generators = np.empty(qcode.qedxm, dtype=object)
    for i in range(qcode.qedxm):
        tmp = np.array([], dtype=int)
        for j in range(qcode.qedxm, qcode.xm):
            if np.any(Hx[i] & Hx[j]): tmp = np.append(tmp, j)
        overlapping_x_generators[i] = tmp

    overlapping_z_generators = np.empty(qcode.qedzm, dtype=object)
    for i in range(qcode.qedzm):
        tmp = np.array([], dtype=int)
        for j in range(qcode.qedzm):
            if np.any(Hz[i] & Hz[j]): tmp = np.append(tmp, j)
        overlapping_z_generators[i] = tmp

    def get_overlapping(measurements, gen_type=False, not_overlapping=False):
        overlapping_generators = overlapping_x_generators if gen_type else overlapping_z_generators
        gens_to_measure = set()
        for g in np.where(measurements)[0]:
            gens_to_measure |= set(overlapping_generators[g])

        if not_overlapping:
            return np.array(list(set(np.arange(qcode.qedxm,qcode.xm)) ^ gens_to_measure))
        else:
            return np.array(list(gens_to_measure))

    # SHOULD ADD SYNDROME BIT ERRORS TO PCM? e.g. arXiv:2004.11199
    if (concat == 1):
        bp_qed_dec = bp_decoder(
            Hx[:qcode.qedxm],
            error_rate=qubit_error_rate,
            bp_method="msl",
            max_iter=100, #pcm.shape[1],
            ms_scaling_factor=0,
        )

    bposd_qed_qec_dec = bposd_decoder(
        Hx[qcode.qedxm:],
        error_rate=qubit_error_rate,
        bp_method="msl",
        max_iter=Hx.shape[1],
        ms_scaling_factor=0,
        osd_method="osd0",
        # osd_order=40
    )

    bposd_qec_dec = bposd_decoder(
        Hx[qcode.qedxm:],
        error_rate=qubit_error_rate,
        bp_method="msl",
        max_iter=Hx.shape[1],
        ms_scaling_factor=0,
        osd_method="osd0",
        # osd_order=40
    )


    rs = []
    for ii in range(1,num_shots+1):
        curr_qubit_error = np.zeros(Hx.shape[1], dtype=int)

        for jj in range(num_rounds):
            new_qubit_error = np.random.choice([0, 1], size=Hx.shape[1], p=[1-qubit_error_rate, qubit_error_rate])
            new_synd_error = np.random.choice([0, 1], size=Hx.shape[0], p=[1-meas_error_rate, meas_error_rate])
            curr_qubit_error ^= new_qubit_error
            curr_synd = ((Hx @ curr_qubit_error) % 2) ^ new_synd_error

            updated_synd = curr_synd.copy()
            non_overlapping_gens = get_overlapping(curr_synd[:qcode.qedxm], True, True)
            if (len(non_overlapping_gens)):
                updated_synd[non_overlapping_gens] = 0

            # QEC
            if (concat == 0):
                if (adaptive == 1):
                    guessed_error = bposd_qec_dec.decode(updated_synd[qcode.qedxm:])
                else:
                    guessed_error = bposd_qec_dec.decode(curr_synd[qcode.qedxm:])
            elif (concat == 1):
                # QED + QEC
                _ = bp_qed_dec.decode(curr_synd[:qcode.qedxm])

                ######################## # THIS MIGHT NEED TO CHANGE SLIGHTLY, SOFT INFORMATION DECODING
                # new_channel_probs = np.exp(-bp_qed_dec.log_prob_ratios)
                new_channel_probs = 1 / (np.exp(bp_qed_dec.log_prob_ratios) + 1)
                new_channel_probs = new_channel_probs / np.sum(new_channel_probs) / (jj+1)# DIVIDED BY NUM_ROUNDS IMPROVES IT FOR SOME REASON !!!!!!!!!!!!
                bposd_qed_qec_dec.update_channel_probs(new_channel_probs)
                ########################

                if (adaptive == 1):
                    guessed_error = bposd_qed_qec_dec.decode(updated_synd[qcode.qedxm:])
                else:
                    guessed_error = bposd_qed_qec_dec.decode(curr_synd[qcode.qedxm:])


            curr_qubit_error ^= guessed_error

        # curr_synd = ((Hx @ curr_qubit_error) % 2)
        # guessed_error = bposd_qec_dec.decode(curr_synd[qcode.qedxm:])
        # curr_qubit_error ^= guessed_error
        # curr_synd = ((Hx @ curr_qubit_error) % 2)

        obs = (Lx @ curr_qubit_error) % 2
        success = not np.any(obs) # and not np.any(curr_synd)
        res = Result(concat, adaptive, qcode.n, qcode.k, num_rounds, qubit_error_rate, meas_error_rate, 1, int(success))
        rs.append(res)

        if (ii%1000==0):
            save_new_res(res_f_name, rs)
            rs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', default="./codes/qcodes/HGP_C642_150_4.qcode", help="Code to simulate")
    # parser.add_argument('-q', default="./codes/qcodes/HGP_400_16.qcode", help="Code to simulate")
    # parser.add_argument('-q', default="./codes/qcodes/HGP_100_4.qcode", help="Code to simulate")

    parser.add_argument('-c', default=0, help="Concatenated decoding?")
    parser.add_argument('-a', default=1, help="QED+QEC protocol?")
    parser.add_argument('-n', default=1e5, help="Number of shots")
    parser.add_argument('-e', default=1e-2, help="Qubit error rate")
    parser.add_argument('-m', default=1e-2, help="Measurement error rate")
    parser.add_argument('-r', default=3, help="Number of rounds")

    args = parser.parse_args()

    main(args)