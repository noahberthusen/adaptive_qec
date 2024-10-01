import numpy as np
import matplotlib.pyplot as plt
from classical_code import *
from quantum_code import *
from ldpc import bposd_decoder, bp_decoder
from result_lifetime import Result, save_new_res
import string, random, argparse


def main(args):
    qcode_fname = args.q
    concat = int(args.c)
    adaptive = int(args.a)
    num_shots = int(args.n)
    qubit_error_rate = float(args.e)
    meas_error_rate = float(args.m)

    # alphabet = string.ascii_letters + string.digits
    # uuid = ''.join(random.choices(alphabet, k=8))

    qcode = read_qcode(qcode_fname)
    concat = 1 if qcode.qedxm and concat else 0
    Hx, Hz, Lx, Lz = qcode.to_numpy()

    if (adaptive == 1):
        overlapping_x_generators = np.empty(100, dtype=object)
        for i in range(qcode.qedxm):
            tmp = np.array([], dtype=int)
            for j in range(qcode.qedxm, qcode.xm):
                if np.any(Hx[i] & Hx[j]): tmp = np.append(tmp, j)
            overlapping_x_generators[i] = tmp

        overlapping_z_generators = np.empty(100, dtype=object)
        # for i in range(concatenatedStabilizersQED.shape[0]):
        #     tmp = np.array([], dtype=int)
        #     for j in range(concatenatedStabilizersZQEC.shape[0]):
        #         if np.any(concatenatedStabilizersQED[i] & concatenatedStabilizersZQEC[j]): tmp = np.append(tmp, j+concatenatedStabilizersQED.shape[0])
        #     overlapping_z_generators[i] = tmp

    def get_overlapping(measurements, gen_type=False, not_overlapping=False):
        overlapping_generators = overlapping_x_generators if gen_type else overlapping_z_generators
        gens_to_measure = set()
        for g in np.where(measurements)[0]:
            gens_to_measure |= set(overlapping_generators[g])

        if not_overlapping:
            return np.array(list(set(np.arange(100,196)) ^ gens_to_measure))
        else:
            return np.array(list(gens_to_measure))



    bp_qed_dec = bp_decoder(
        Hx[:100], # the parity check matrix
        error_rate=qubit_error_rate,
        # channel_probs=new_channel_probs, #assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=100, #pcm.shape[1], #the maximum number of iterations for BP)
        bp_method="ps",
        # osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    )

    bp_qed_qec_dec = bposd_decoder(
        Hx,
        error_rate=qubit_error_rate,
        bp_method="ps",
        max_iter=100,
        # ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        # osd_order=40 #the osd search depth
    )

    bposd_qec_dec = bposd_decoder(
        Hx,
        error_rate=qubit_error_rate,
        bp_method="ps",
        max_iter=100,
        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        # osd_order=40 #the osd search depth
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


            # QEC
            if (concat == 0 and adaptive == 0):
                guessed_error = bposd_qec_dec.decode(curr_synd)
            elif (concat == 1):
                # QED + QEC
                initial_guess = bp_qed_dec.decode(curr_synd[:100])

                ########################
                # new_channel_probs = np.exp(-bp_qed_dec.log_prob_ratios) # THIS MIGHT NEED TO CHANGE SLIGHTLY
                # new_channel_probs = new_channel_probs / np.sum(new_channel_probs)
                new_channel_probs = 1 / (np.exp(bp_qed_dec.log_prob_ratios) + 1)
                # new_channel_probs = new_channel_probs / np.sum(new_channel_probs)
                # new_channel_probs[400:] = meas_error_rate
                bp_qed_qec_dec.update_channel_probs(new_channel_probs)
                ########################

                updated_synd = curr_synd.copy()
                if (adaptive == 1):
                    updated_synd[get_overlapping(curr_synd[:100], True, True)] = 0

                guessed_error = bp_qed_qec_dec.decode(updated_synd)


            residual_error = curr_qubit_error ^ guessed_error
            obs = [np.count_nonzero(residual_error[l]) % 2 for l in qcode.Lz]

            if np.any(obs):
                break

        res = Result(concat, adaptive, qcode.n, qcode.k, qubit_error_rate, meas_error_rate, 1, num_rounds, 0)
        rs.append(res)

        if (ii%100==0):
            save_new_res('./tmp.res', rs)
            rs = []



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', default="./codes/qcodes/HGP_C422_400_8.qcode", help="Code to simulate")
    parser.add_argument('-c', default=1, help="Concatenated decoding?")
    parser.add_argument('-a', default=1, help="QED+QEC protocol?")
    parser.add_argument('-n', default=1e3, help="Number of shots")
    parser.add_argument('-e', default=0.001, help="Qubit error rate")
    parser.add_argument('-m', default=0.001, help="Measurement error rate")

    args = parser.parse_args()

    main(args)