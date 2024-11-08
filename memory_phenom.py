import numpy as np
from classical_code import *
from quantum_code import *
from ldpc import BpDecoder, BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
import argparse
from result_memory import save_new_res, Result
from pathlib import Path
from filelock import Timeout, FileLock
import os


def main(args):
    qcode_fname = args.q
    concat = int(args.c)
    adapt = int(args.a)
    soft = int(args.s)
    num_shots = int(args.n)
    qubit_error_rate = float(args.e)
    meas_error_rate = float(args.m)
    num_rounds = int(args.r)

    qcode = read_qcode(qcode_fname)
    concat = 1 if qcode.qedxm and concat else 0
    adapt = 1 if qcode.qedxm and adapt else 0
    soft = 1 if qcode.qedxm and soft else 0


    filepath = Path(qcode_fname)
    hgp_qcode = read_qcode(f"{os.path.join(filepath.parent, filepath.parent.name)}.qcode")
    hgp_Hx, hgp_Hz, hgp_Lx, hgp_Lz, _ = hgp_qcode.to_numpy()
    Hx, Hz, Lx, Lz, mapping = qcode.to_numpy()

    stab_type = True
    tanner = True if meas_error_rate else False

    H = Hx if stab_type else Hz
    hgp_H = hgp_Hx if stab_type else hgp_Hz

    qec_aug_dec_H = np.hstack([hgp_H, np.eye(hgp_H.shape[0], dtype=int)])
    qec_aug_channel_probs = [qubit_error_rate]*hgp_H.shape[1] + [meas_error_rate]*(hgp_H.shape[0])

    qec_dec_H = hgp_H
    qec_channel_probs = [qubit_error_rate]*hgp_H.shape[1]
    L = Lx if stab_type else Lz


    res_f_name = f"./results/{Path(qcode_fname).name}_laptop.res"
    res_f_name_lock = f"./results/locks/{Path(qcode_fname).name}.res.lock"
    lock = FileLock(res_f_name_lock, timeout=10)

    ###################################################

    overlapping_x_generators = np.empty(qcode.qedxm, dtype=object)
    for i in range(qcode.qedxm):
        tmp = np.array([], dtype=int)
        for j in range(qcode.qedxm,qcode.xm):
            if np.any(Hx[i] & Hx[j]): tmp = np.append(tmp, j)
        overlapping_x_generators[i] = tmp

    overlapping_z_generators = np.empty(qcode.qedxm, dtype=object)
    for i in range(qcode.qedzm):
        tmp = np.array([], dtype=int)
        for j in range(qcode.qedzm,qcode.zm):
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

    def iceberglogicals(n):
        icebergX = np.zeros(shape=(n-2,2), dtype=int)
        icebergZ = np.zeros(shape=(n-2,2), dtype=int)

        for i in range(n-2):
            icebergX[i] = np.array([0,i+1])
            icebergZ[i] = np.array([i+1,n-1])

        return icebergX, icebergZ
    icebergX, icebergZ = iceberglogicals(4)


    qec_aug_dec = BpDecoder(
        qec_aug_dec_H,
        channel_probs=qec_aug_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial"
        # osd_method="osd0",
        # osd_order=4 #the osd search depth
    )

    qec_dec = BpLsdDecoder(
        qec_dec_H,
        channel_probs=qec_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial",
        lsd_method="lsd_cs",
        lsd_order=4 #the osd search depth
    )

    def decode(curr_qubit_error, curr_synd_error, augment):
        curr_synd = ((H @ curr_qubit_error) % 2) ^ curr_synd_error
        if adapt: curr_synd[get_overlapping(curr_synd[:qcode.qedxm], stab_type, not_overlapping=True)] = 0

        if not concat:
            # QEC only
            if augment:
                curr_qubit_error ^= qec_aug_dec.decode(curr_synd[qcode.qedzm:])[:qec_dec_H.shape[1]]
            else:
                curr_qubit_error ^= qec_dec.decode(curr_synd[qcode.qedzm:])
        elif concat:
            # QED + QEC
            curr_qed_synd = curr_synd[:qcode.qedxm]
            curr_hgp_synd = curr_synd[qcode.qedxm:]

            block_correction = np.array([0,0,0,1], dtype=int) if stab_type else np.array([1,0,0,0], dtype=int)
            corrections = np.concatenate([block_correction if x == 1 else np.zeros(4, dtype=int) for x in curr_qed_synd])
            curr_qubit_error ^= corrections

            #######################
            if soft:
                new_channel_probs = qubit_error_rate**2 * np.ones(hgp_H.shape[1])
                new_channel_probs[mapping[curr_qed_synd == 1].flatten()] = 0.5
                if augment:
                    new_channel_probs = np.concatenate([new_channel_probs, [meas_error_rate]*hgp_H.shape[0]])
                    qec_aug_dec.update_channel_probs(new_channel_probs)
                else:
                    qec_dec.update_channel_probs(new_channel_probs)
            ########################

            if augment:
                logical_correction = qec_aug_dec.decode(curr_hgp_synd)[:hgp_H.shape[1]]
            else:
                logical_correction = qec_dec.decode(curr_hgp_synd)[:hgp_H.shape[1]]

            physical_correction = np.zeros(Hx.shape[1], dtype=int)

            for c in np.where(logical_correction)[0]:
                    iceberg_block = np.where(mapping == c)[0][0]
                    iceberg_log = np.where(mapping == c)[1][0]
                    if stab_type:
                            physical_correction[icebergZ[iceberg_log]+(4*iceberg_block)] ^= 1
                    else:
                            physical_correction[icebergX[iceberg_log]+(4*iceberg_block)] ^= 1

            curr_qubit_error ^= physical_correction
        return curr_qubit_error.astype(int)

    rs = []
    for ii in range(1,num_shots+1):
        curr_qubit_error = np.zeros(H.shape[1], dtype=int)

        for jj in range(num_rounds):
            new_qubit_error = np.random.choice([0, 1], size=H.shape[1], p=[1-qubit_error_rate, qubit_error_rate])
            new_synd_error = np.random.choice([0, 1], size=H.shape[0], p=[1-meas_error_rate, meas_error_rate])
            curr_qubit_error ^= new_qubit_error

            curr_qubit_error = decode(curr_qubit_error, new_synd_error, tanner)

        curr_qubit_error = decode(curr_qubit_error, np.zeros(H.shape[0], dtype=int), 0)

        obs = (L @ curr_qubit_error) % 2

        success = not np.any(obs)
        res = Result(concat, adapt, soft, qcode.n, qcode.k, num_rounds, qubit_error_rate, meas_error_rate, 0, 1, int(success))
        rs.append(res)

        if (ii % 1000 == 0):
            with lock:
                save_new_res(res_f_name, rs)
            rs = []



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', default="./codes/qcodes/HGP_100_4/HGP_C422_200_4.qcode", help="Code to simulate")
    # parser.add_argument('-q', default="./codes/qcodes/HGP_400_16/HGP_C422_800_16.qcode", help="Code to simulate")

    # parser.add_argument('-q', default="./codes/qcodes/HGP_100_4/HGP_100_4.qcode", help="Code to simulate")


    parser.add_argument('-c', default=1, help="Concatenated decoding?")
    parser.add_argument('-a', default=0, help="QED+QEC protocol?")
    parser.add_argument('-s', default=1, help="Soft information?")

    parser.add_argument('-n', default=1e4, help="Number of shots")
    parser.add_argument('-e', default=0.01, help="Qubit error rate")
    parser.add_argument('-m', default=0.01, help="Measurement error rate")
    parser.add_argument('-r', default=10, help="Number of rounds")

    args = parser.parse_args()

    main(args)