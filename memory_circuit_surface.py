import numpy as np
from classical_code import *
from quantum_code import *
import stim
from ldpc import BpDecoder, BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
import argparse
import pymatching
from result_memory import save_new_res, Result
from pathlib import Path
from filelock import Timeout, FileLock
import os

def rotated_surface_code(d, p):
    if (d == 3):
        return stim.Circuit(f"""
            H 2 11 16 25
            DEPOLARIZE1({p/10}) 2 11 16 25
            TICK
            CX 2 3 16 17 11 12 15 14 10 9 19 18
            DEPOLARIZE2({p}) 2 3 16 17 11 12 15 14 10 9 19 18
            TICK
            CX 2 1 16 15 11 10 8 14 3 9 12 18
            DEPOLARIZE2({p}) 2 1 16 15 11 10 8 14 3 9 12 18
            TICK
            CX 16 10 11 5 25 19 8 9 17 18 12 13
            DEPOLARIZE2({p}) 16 10 11 5 25 19 8 9 17 18 12 13
            TICK
            CX 16 8 11 3 25 17 1 9 10 18 5 13
            DEPOLARIZE2({p}) 16 8 11 3 25 17 1 9 10 18 5 13
            TICK
            H 2 11 16 25
            DEPOLARIZE1({p/10}) 2 11 16 25
            TICK
            X_ERROR({p}) 2 9 11 13 14 16 18 25
            MR 2 9 11 13 14 16 18 25
            X_ERROR({p}) 2 9 11 13 14 16 18 25""")

def final_measurement(d):
    if (d == 3):
        return stim.Circuit("M 1 3 5 8 10 12 15 17 19")

def logical(d):
    pass

def main(args):
    distance = int(args.d)

    num_shots = int(args.n)
    # idle_error_rate = float(args.i)
    qubit_error_rate = float(args.e)
    meas_error_rate = float(args.m)
    num_rounds = int(args.r)

    filepath = Path(qcode_fname)
    hgp_qcode = read_qcode(f"{os.path.join(filepath.parent, filepath.parent.name)}.qcode")
    hgp_Hx, hgp_Hz, hgp_Lx, hgp_Lz, _ = hgp_qcode.to_numpy()
    Hx, Hz, Lx, Lz, mapping = qcode.to_numpy()


    stab_type = False

    hgp_H = hgp_Hx if stab_type else hgp_Hz

    qec_aug_dec_Hx = np.hstack([hgp_Hx, np.eye(hgp_Hx.shape[0], dtype=int)])
    qec_aug_dec_Hz = np.hstack([hgp_Hz, np.eye(hgp_Hz.shape[0], dtype=int)])
    qec_aug_channel_probs = [0.01]*hgp_H.shape[1] + [0.01]*(hgp_H.shape[0])

    qec_dec_Hx = hgp_Hx
    qec_dec_Hz = hgp_Hz
    qec_channel_probs = [0.01]*hgp_H.shape[1]


    res_f_name = f"./results/{Path(qcode_fname).name}.res"
    res_f_name_lock = f"./results/locks/{Path(qcode_fname).name}.res.lock"
    lock = FileLock(res_f_name_lock, timeout=10)



    qec_aug_decZ = BpDecoder(
        qec_aug_dec_Hz,
        channel_probs=qec_aug_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial",
        # lsd_method="lsd0"
        # osd_order=4 #the osd search depth
    )

    qec_decZ = BpLsdDecoder(
        qec_dec_Hz,
        channel_probs=qec_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial",
        lsd_method="lsd_cs",
        lsd_order=4 #the osd search depth
    )

    qec_aug_decX = BpDecoder(
        qec_aug_dec_Hx,
        channel_probs=qec_aug_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial",
        # lsd_method="lsd0"
        # osd_order=4 #the osd search depth
    )

    qec_decX = BpLsdDecoder(
        qec_dec_Hx,
        channel_probs=qec_channel_probs,
        bp_method="ps",
        max_iter=30,
        schedule="serial",
        lsd_method="lsd_cs",
        lsd_order=4 #the osd search depth
    )



    def decode(curr_synd, augment, stab_type):
        H = Hx if stab_type else Hz
        guessed_error = np.zeros(H.shape[1], dtype=int)
        qec_aug_dec = qec_aug_decX if stab_type else qec_aug_decZ
        qec_dec = qec_decX if stab_type else qec_decZ

        if not concat:
            # QEC only
            if augment:
                guessed_error ^= qec_aug_dec.decode(curr_synd[qcode.qedzm:])[:hgp_H.shape[1]]
            else:
                guessed_error ^= qec_dec.decode(curr_synd[qcode.qedzm:])
        return guessed_error


    cn = qcode.n
    cmx = qcode.xm
    cmz = qcode.zm
    data_qbts = np.arange(cn)
    x_checks = np.arange(cn,cn+cmx)
    z_checks = np.arange(cn+cmx,cn+cmx+cmz)


    class Simulation:
        def __init__(self, num_rounds, stab_type):
            self.num_rounds = num_rounds
            self.stab_type = stab_type
            self.curr_round = 1

            # self.tmp = np.array([], dtype=int)

            self.z_check_history = np.ones(cmz, dtype=int)
            self.x_check_history = np.ones(cmx, dtype=int)
            self.z_syndrome_history = np.zeros(shape=(num_rounds+2, cmz), dtype=int)
            self.x_syndrome_history = np.zeros(shape=(num_rounds+2, cmx), dtype=int)
            self.z_observables = np.zeros(qcode.k, dtype=int)
            self.x_observables = np.zeros(qcode.k, dtype=int)

            self.c = stim.Circuit()
            self.s = stim.TableauSimulator()

            if self.stab_type:
                self.c.append("H", [qbt for qbt in data_qbts])
                for z in np.arange(qcode.zm):
                    gen_qbts = data_qbts[np.where(Hz[z])[0]]
                    for qbt in gen_qbts:
                        path_qbts = [qbt, z_checks[z]]
                        self.c.append("CNOT", path_qbts)
                for i, z_check in enumerate(np.arange(cmz)):
                    self.c.append("MR", z_checks[z_check])
            else:
                self.c.append("H", x_checks)
                for x in np.arange(qcode.xm):
                    gen_qbts = data_qbts[np.where(Hx[x])[0]]
                    for qbt in gen_qbts:
                        path_qbts = [x_checks[x], qbt]
                        self.c.append("CNOT", path_qbts)
                self.c.append("H", x_checks)
                for i, x_check in enumerate(np.arange(cmx)):
                    self.c.append("MR", x_checks[x_check])

            self.s.do_circuit(self.c)
            if self.stab_type:
                self.z_syndrome_history[:] = self.s.current_measurement_record()
            else:
                self.x_syndrome_history[:] = self.s.current_measurement_record()


        def QEC(self):
            def measure_z_qec_checks(curr_z_checks):
                c = stim.Circuit()
                if np.any(curr_z_checks):
                    c += prepare_z_checks(curr_z_checks)
                    c.append("X_ERROR", [z_checks[z_check] for z_check in curr_z_checks], meas_error_rate)
                    c.append("MR", [z_checks[z_check] for z_check in curr_z_checks])
                    c.append("X_ERROR", [z_checks[z_check] for z_check in curr_z_checks], meas_error_rate)
                return c

            def measure_x_qec_checks(curr_x_checks):
                c = stim.Circuit()
                if np.any(curr_x_checks):
                    c += prepare_x_checks(curr_x_checks)
                    c.append("X_ERROR", [x_checks[x_check] for x_check in curr_x_checks], meas_error_rate)
                    c.append("MR", [x_checks[x_check] for x_check in curr_x_checks])
                    c.append("X_ERROR", [x_checks[x_check] for x_check in curr_x_checks], meas_error_rate)
                return c

            c = stim.Circuit()
            if self.stab_type:
                c += measure_x_qec_checks(self.curr_x_checks)
                c += measure_z_qec_checks(self.curr_z_checks)
            else:
                c += measure_z_qec_checks(self.curr_z_checks)
                c += measure_x_qec_checks(self.curr_x_checks)
            return c

        def final_synd_and_observables(self):
            c = stim.Circuit()

            if self.stab_type: c.append("H", [qbt for qbt in data_qbts])
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
                    self.x_observables[i] = (np.sum(np.take(meas, incl_qbts)) % 2)
                else:
                    self.z_observables[i] = (np.sum(np.take(meas, incl_qbts)) % 2)

            if self.stab_type:
                final_correction = decode(self.x_syndrome_history[-1], 0, True)
                self.x_observables ^= (Lx @ final_correction) % 2
            else:
                final_correction = decode(self.z_syndrome_history[-1], 0, False)
                self.z_observables ^= (Lz @ final_correction) % 2


        def simulate(self):
            print(self.x_syndrome_history[0])
            for _ in range(1, self.num_rounds+1):
                self.curr_z_checks = np.zeros(cmz)
                self.curr_x_checks = np.zeros(cmx)

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

                print(self.x_syndrome_history[self.curr_round])
                if self.stab_type:
                    guessed_z_error = decode(self.x_syndrome_history[self.curr_round], 1, True)
                    self.s.z(*np.where(guessed_z_error)[0])
                    guessed_x_error = decode(self.z_syndrome_history[self.curr_round] ^ self.z_syndrome_history[0], 1, False)
                    self.s.x(*np.where(guessed_x_error)[0])
                else:
                    guessed_x_error = decode(self.z_syndrome_history[self.curr_round], 1, False)
                    self.s.x(*np.where(guessed_x_error)[0])
                    guessed_z_error = decode(self.x_syndrome_history[self.curr_round] ^ self.x_syndrome_history[0], 1, True)
                    self.s.z(*np.where(guessed_z_error)[0])

                self.curr_round += 1

            self.final_synd_and_observables()


    rs = []
    for ii in range(num_shots):
        s = Simulation(num_rounds, stab_type)
        s.simulate()
        c = s.c

        success = not np.any(s.x_observables) if stab_type else (not np.any(s.z_observables))
        if qcode.qedxm:
            num_CNOTs = (str(c).count("CX") - 1 - (num_rounds * 8)) / num_rounds + (2 * Hx.shape[1])
        else:
            num_CNOTs = (str(c).count("CX") - 1) / num_rounds


        res = Result(0, 0, 0, distance**2, 1, num_rounds,
                     qubit_error_rate, meas_error_rate, num_CNOTs, 1, int(success))
        rs.append(res)

        if (ii % 10 == 0):
            with lock:
                save_new_res(res_f_name, rs)
            rs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', default=4, help="Distance")

    parser.add_argument('-n', default=1e4, help="Number of shots")
    parser.add_argument('-e', default=0.001, help="Qubit error rate")
    # parser.add_argument('-i', default=0.001, help="Idle error rate")
    parser.add_argument('-m', default=0.001, help="Measurement error rate")
    parser.add_argument('-r', default=10, help="Number of rounds")

    args = parser.parse_args()

    main(args)