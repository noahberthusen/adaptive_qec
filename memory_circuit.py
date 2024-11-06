import numpy as np
from classical_code import *
from quantum_code import *
import stim
from scipy.sparse import lil_matrix
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
    adaptive = int(args.a)
    num_shots = int(args.n)
    qubit_error_rate = float(args.e)
    meas_error_rate = float(args.m)
    num_rounds = int(args.r)
    stab_type = True

    qcode = read_qcode(qcode_fname)
    concat = 1 if qcode.qedxm and concat else 0

    filepath = Path(qcode_fname)
    hgp_qcode = read_qcode(f"{os.path.join(filepath.parent, filepath.parent.name)}.qcode")
    hgp_Hx, hgp_Hz, hgp_Lx, hgp_Lz, _ = hgp_qcode.to_numpy()
    Hx, Hz, Lx, Lz, mapping = qcode.to_numpy()


    stab_type = False
    tanner = True if meas_error_rate else False

    H = Hx if stab_type else Hz
    hgp_H = hgp_Hx if stab_type else hgp_Hz

    # qed_dec_H = H[:qcode.qedxm]
    # qed_channel_probs = [qubit_error_rate]*H.shape[1]

    qec_aug_dec_H = np.hstack([hgp_H, np.eye(hgp_H.shape[0], dtype=int)])
    qec_aug_channel_probs = [qubit_error_rate]*hgp_H.shape[1] + [meas_error_rate]*(hgp_H.shape[0])

    qec_dec_H = hgp_H
    qec_channel_probs = [qubit_error_rate]*hgp_H.shape[1]
    L = Lx if stab_type else Lz


    res_f_name = f"./results/{Path(qcode_fname).name}.res"
    res_f_name_lock = f"./results/locks/{Path(qcode_fname).name}.res.lock"
    lock = FileLock(res_f_name_lock, timeout=10)


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
        osd_method="lsd_cs",
        osd_order=4 #the osd search depth
    )

    def decode(curr_synd, augment, concat):
        guessed_error = np.zeros(H.shape[1], dtype=int)

        if (concat == 0):
            # QEC only
            if augment:
                guessed_error ^= qec_aug_dec.decode(curr_synd[qcode.qedzm:])[:hgp_H.shape[1]]
            else:
                guessed_error ^= qec_dec.decode(curr_synd[qcode.qedzm:])
        elif (concat == 1):
            # QED + QEC
            curr_qed_synd = curr_synd[:qcode.qedxm]
            curr_hgp_synd = curr_synd[qcode.qedxm:]

            block_correction = np.array([0,0,0,1], dtype=int) if stab_type else np.array([1,0,0,0], dtype=int)
            corrections = np.concatenate([block_correction if x == 1 else np.zeros(4, dtype=int) for x in curr_qed_synd])
            guessed_error ^= corrections

            #######################
            new_channel_probs = 0.0003 * np.ones(hgp_H.shape[1])
            new_channel_probs[mapping[curr_qed_synd == 1].flatten()] = 0.25
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

            guessed_error ^= physical_correction
        return guessed_error


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


    cn = qcode.n
    cmx = qcode.xm
    cmz = qcode.zm
    data_qbts = np.arange(cn)
    x_checks = np.arange(cn,cn+cmx)
    z_checks = np.arange(cn+cmx,cn+cmx+cmz)
    all_qbts = np.arange(cn+cmx+cmz)

    precomputed_x_checks_circuits = np.empty(cmx, dtype=object)
    for x in np.arange(qcode.xm):
        c = stim.Circuit()
        gen_qbts = data_qbts[np.where(Hx[x])[0]]
        for qbt in gen_qbts:
            path_qbts = [x_checks[x], qbt]
            c.append("CNOT", path_qbts)
            c.append("DEPOLARIZE2", path_qbts, qubit_error_rate)
        precomputed_x_checks_circuits[x] = c

    def prepare_x_checks(checks):
        c = stim.Circuit()
        if len(checks) == 0: return c
        c.append("H", [x_checks[x_check] for x_check in checks])
        c.append("DEPOLARIZE1", [x_checks[x_check] for x_check in checks], qubit_error_rate)
        for x in checks:
            c += precomputed_x_checks_circuits[x]
        c.append("H", [x_checks[x_check] for x_check in checks])
        c.append("DEPOLARIZE1", [x_checks[x_check] for x_check in checks], qubit_error_rate)
        return c

    precomputed_z_checks_circuits = np.empty(cmz, dtype=object)
    for z in np.arange(qcode.zm):
        c = stim.Circuit()
        gen_qbts = data_qbts[np.where(Hz[z])[0]]
        for qbt in gen_qbts:
            path_qbts = [qbt, z_checks[z]]
            c.append("CNOT", path_qbts)
            c.append("DEPOLARIZE2", path_qbts, qubit_error_rate)
        precomputed_z_checks_circuits[z] = c

    def prepare_z_checks(checks):
        c = stim.Circuit()
        if len(checks) == 0: return c
        for z in checks:
            c += precomputed_z_checks_circuits[z]
        return c


    class Simulation:
        def __init__(self, num_rounds, stab_type, concat=True, adaptive=True):
            self.num_rounds = num_rounds
            self.stab_type = stab_type
            self.curr_round = 1
            self.concat = concat
            self.adaptive = adaptive

            self.z_check_history = np.ones(cmz, dtype=int)
            self.x_check_history = np.ones(cmx, dtype=int)
            self.z_syndrome_history = np.zeros(shape=(num_rounds+3, cmz), dtype=int)
            self.x_syndrome_history = np.zeros(shape=(num_rounds+3, cmx), dtype=int)
            self.z_observables = np.zeros(qcode.k, dtype=int)
            self.x_observables = np.zeros(qcode.k, dtype=int)

            self.c = stim.Circuit()
            self.s = stim.TableauSimulator()

            if self.stab_type:
                self.c.append("H", [qbt for qbt in data_qbts])
                self.c += prepare_z_checks(np.arange(cmz)).without_noise()
                # self.c.append("X_ERROR", z_checks, 1e-4)
                for i, z_check in enumerate(np.arange(cmz)):
                    self.c.append("MR", z_checks[z_check])
                # self.c.append("X_ERROR", z_checks, 1e-5)
            else:
                self.c += prepare_x_checks(np.arange(cmx)).without_noise()
                # self.c.append("X_ERROR", x_checks, 1e-4)
                for i, x_check in enumerate(np.arange(cmx)):
                    self.c.append("MR", x_checks[x_check])
                # self.c.append("X_ERROR", x_checks, 1e-5)

            self.s.do_circuit(self.c)
            if self.stab_type:
                self.z_syndrome_history[0] = self.s.current_measurement_record()
            else:
                self.x_syndrome_history[0] = self.s.current_measurement_record()


        def QED(self):
            def measure_z_qed_checks(c):
                c += prepare_z_checks(np.arange(qcode.qedzm))
                c.append("X_ERROR", [z_checks[z_check] for z_check in np.arange(qcode.qedzm)], meas_error_rate)
                for i, z_check in enumerate(np.arange(qcode.qedzm)):
                    c.append("MR", z_checks[z_check])
                return c

            def measure_x_qed_checks(c):
                c += prepare_x_checks(np.arange(qcode.qedxm))
                c.append("X_ERROR", [x_checks[x_check] for x_check in np.arange(qcode.qedxm)], meas_error_rate)
                for i, x_check in enumerate(np.arange(qcode.qedxm)):
                    c.append("MR", x_checks[x_check])
                return c

            c = stim.Circuit()
            if self.stab_type:
                c = measure_x_qed_checks(c)
                c = measure_z_qed_checks(c)
            else:
                c = measure_z_qed_checks(c)
                c = measure_x_qed_checks(c)
            return c

        def QEC(self):
            def measure_z_qec_checks(c, curr_z_checks):
                c += prepare_z_checks(curr_z_checks)
                c.append("X_ERROR", [z_checks[z_check] for z_check in curr_z_checks], meas_error_rate)
                for i, z_check in enumerate(curr_z_checks):
                    c.append("MR", z_checks[z_check])
                return c

            def measure_x_qec_checks(c, curr_x_checks):
                c += prepare_x_checks(curr_x_checks)
                c.append("X_ERROR", [x_checks[x_check] for x_check in curr_x_checks], meas_error_rate)
                for i, x_check in enumerate(curr_x_checks):
                    c.append("MR", x_checks[x_check])
                return c

            c = stim.Circuit()
            if self.stab_type:
                c = measure_x_qec_checks(c, self.curr_x_checks)
                c = measure_z_qec_checks(c, self.curr_z_checks)
            else:
                c = measure_z_qec_checks(c, self.curr_z_checks)
                c = measure_x_qec_checks(c, self.curr_x_checks)
            return c

        def final_synd_and_observables(self):
            self.curr_z_checks = np.arange(cmz)
            self.curr_x_checks = np.arange(cmx)
            confirmation_z = np.ones(cmz, dtype=int)
            confirmation_x = np.ones(cmx, dtype=int)
            self.z_check_history = np.vstack([self.z_check_history, confirmation_z])
            self.x_check_history = np.vstack([self.x_check_history, confirmation_x])

            QEC_circuit = self.QEC().without_noise()
            self.s.do_circuit(QEC_circuit)
            self.c += QEC_circuit

            meas = self.s.current_measurement_record()
            lookback = lambda x: -len(x) if len(x) else None

            if self.stab_type:
                pass
            else:
                if len(self.curr_z_checks):
                    self.z_syndrome_history[self.curr_round][self.curr_z_checks] = meas[lookback(np.concatenate([self.curr_z_checks, self.curr_x_checks])):
                                                                            lookback(self.curr_x_checks)]
                if len(self.curr_x_checks):
                    self.x_syndrome_history[self.curr_round][self.curr_x_checks] = meas[lookback(self.curr_x_checks):]

                guessed_error = decode(self.z_syndrome_history[self.curr_round], 0, self.concat)
                self.s.x(*np.where(guessed_error)[0])

            ######################################
            c = stim.Circuit()

            if self.stab_type: self.c.append("H", [qbt for qbt in data_qbts])
            c.append("M", data_qbts)

            self.s.do_circuit(c)
            self.c += c

            meas = self.s.current_measurement_record()
            H = Hx if self.stab_type else Hz
            for i in range(H.shape[0]):
                incl_qbts = np.where(H[i])[0]
                incl_qbts = np.array([j-cn for j in incl_qbts])

                if self.stab_type:
                    self.x_syndrome_history[-1][i] = np.sum(np.take(meas, incl_qbts)) % 2
                else:
                    self.z_syndrome_history[-1][i] = np.sum(np.take(meas, incl_qbts)) % 2

            for i, logical in enumerate(Lx if self.stab_type else Lz):
                incl_qbts = np.where(logical)[0]
                incl_qbts = [j-cn for j in incl_qbts]

                if self.stab_type:
                    self.x_observables[i] = np.sum(np.take(meas, incl_qbts)) % 2
                else:
                    self.z_observables[i] = np.sum(np.take(meas, incl_qbts)) % 2


        def simulate(self):
            for _ in range(1, self.num_rounds+1):
                # self.c.append("X_ERROR", data_qbts, qubit_error_rate)  # idle errors
                # self.s.x_error(*data_qbts, p=qubit_error_rate)

                self.curr_z_checks = np.zeros(cmz)
                self.curr_x_checks = np.zeros(cmx)
                if not self.adaptive:
                    self.curr_z_checks = np.arange(cmz)
                    self.curr_x_checks = np.arange(cmx)
                else:
                    QED_circuit = self.QED()
                    self.s.do_circuit(QED_circuit)
                    self.c += QED_circuit

                    # determining which of the QEC stabilizers to measure
                    meas = self.s.current_measurement_record()
                    if self.stab_type:
                        self.x_syndrome_history[self.curr_round][:qcode.qedxm] = meas[-(qcode.qedxm+qcode.qedzm):-qcode.qedxm]
                        self.z_syndrome_history[self.curr_round][:qcode.qedzm] = meas[-qcode.qedzm:]
                    else:
                        self.z_syndrome_history[self.curr_round][:qcode.qedzm] = meas[-(qcode.qedzm+qcode.qedxm):-qcode.qedzm]
                        self.x_syndrome_history[self.curr_round][:qcode.qedxm] = meas[-qcode.qedxm:]

                    z_qed_synd_diff = self.z_syndrome_history[self.curr_round-1][:qcode.qedzm] ^ self.z_syndrome_history[self.curr_round][:qcode.qedzm]
                    x_qed_synd_diff = self.x_syndrome_history[self.curr_round-1][:qcode.qedxm] ^ self.x_syndrome_history[self.curr_round][:qcode.qedxm]

                    self.curr_z_checks = sorted(get_overlapping(z_qed_synd_diff, False))
                    self.curr_x_checks = sorted(get_overlapping(x_qed_synd_diff, True))

                confirmation_z = np.concatenate([np.ones(qcode.qedzm, dtype=int), np.zeros(cmz-qcode.qedzm, dtype=int)])
                confirmation_z[self.curr_z_checks] = 1
                confirmation_x = np.concatenate([np.ones(qcode.qedxm, dtype=int), np.zeros(cmx-qcode.qedxm, dtype=int)])
                confirmation_x[self.curr_x_checks] = 1
                self.z_check_history = np.vstack([self.z_check_history, confirmation_z])
                self.x_check_history = np.vstack([self.x_check_history, confirmation_x])

                QEC_circuit = self.QEC()
                self.s.do_circuit(QEC_circuit)
                self.c += QEC_circuit

                meas = self.s.current_measurement_record()
                lookback = lambda x: -len(x) if len(x) else None
                if self.stab_type:
                    if len(self.curr_x_checks):
                        self.x_syndrome_history[self.curr_round][self.curr_x_checks] = meas[lookback(np.concatenate([self.curr_x_checks, self.curr_z_checks])):
                                                                            lookback(self.curr_z_checks)]
                    if len(self.curr_z_checks):
                        self.z_syndrome_history[self.curr_round][self.curr_z_checks] = meas[lookback(self.curr_z_checks):]
                else:
                    if len(self.curr_z_checks):
                        self.z_syndrome_history[self.curr_round][self.curr_z_checks] = meas[lookback(np.concatenate([self.curr_z_checks, self.curr_x_checks])):
                                                                            lookback(self.curr_x_checks)]
                    if len(self.curr_x_checks):
                        self.x_syndrome_history[self.curr_round][self.curr_x_checks] = meas[lookback(self.curr_x_checks):]

                if self.stab_type:
                    pass
                else:
                    guessed_error = decode(self.z_syndrome_history[self.curr_round], 1, self.concat)
                    self.s.x(*np.where(guessed_error)[0])

                self.curr_round += 1

            self.final_synd_and_observables()


    rs = []
    for ii in range(num_shots):
        s = Simulation(num_rounds, stab_type, concat=concat, adaptive=adaptive)
        s.simulate()

        success = not np.any(s.z_observables)
        num_gens = np.count_nonzero(s.z_check_history[1:-1]) / num_rounds

        res = Result(concat, adaptive, qcode.n, qcode.k, num_rounds,
                     qubit_error_rate, meas_error_rate, num_gens, 1, int(success))
        rs.append(res)

        if (ii % 100 == 0):
            with lock:
                save_new_res(res_f_name, rs)
            rs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-q', default="./codes/qcodes/HGP_100_4/HGP_C422_200_4.qcode", help="Code to simulate")
    # parser.add_argument('-q', default="./codes/qcodes/HGP_400_16/HGP_C422_800_16.qcode", help="Code to simulate")

    # parser.add_argument('-q', default="./codes/qcodes/HGP_100_4/HGP_100_4.qcode", help="Code to simulate")
    # parser.add_argument('-q', default="./codes/qcodes/HGP_400_16/HGP_400_16.qcode", help="Code to simulate")
    parser.add_argument('-q', default="./codes/qcodes/HGP_900_36/HGP_900_36.qcode", help="Code to simulate")


    parser.add_argument('-c', default=0, help="Concatenated decoding?")
    parser.add_argument('-a', default=0, help="QED+QEC protocol?")
    parser.add_argument('-n', default=1e4, help="Number of shots")
    parser.add_argument('-e', default=0.0002, help="Qubit error rate")
    parser.add_argument('-m', default=0.0002, help="Measurement error rate")
    parser.add_argument('-r', default=10, help="Number of rounds")

    args = parser.parse_args()

    main(args)