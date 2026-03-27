import os
import sys
import json
sys.path.append('..')
from Utils.simulator import to_p, to_Y
from Utils.kraus import *
from Utils.qec import *
from copy import deepcopy

def multi_kron(*args: np.ndarray):
    result = args[0]
    for _ in range (1, len(args)):
        result = np.kron(result,args[_])
    return result

def _PI3():
    PI3_0 = np.zeros(8)
    PI3_1 = np.zeros(8)
    for i in range(0, 3):
        PI3_0[1 << i] = 1

    PI3_1[-1] = 1

    dutta_5 = np.array(
        [PI3_0 / np.linalg.norm(PI3_0), PI3_1 / np.linalg.norm(PI3_1)]
    )

    return dutta_5


def get_th_data_all(T1: float, T2: float, times: np.ndarray[float], relaxation_time:float, sigma: float = 0.0):

    codes = _PI3()
    state_1L = codes[1]
    rho_1L = np.outer(state_1L, state_1L.conj().T)
    state_0L = codes[0]
    rho_0L = np.outer(state_0L, state_0L.conj().T)

    qubit1_T1 = np.random.normal(T1, sigma)
    qubit2_T1 = np.random.normal(T1, sigma)
    qubit3_T1 = np.random.normal(T1, sigma)
    qubit1_T2 = np.random.normal(T2, sigma)
    qubit2_T2 = np.random.normal(T2, sigma)
    qubit3_T2 = np.random.normal(T2, sigma)

    def fluctuating_noise(t):
        qubit1_Aks = Krauser.AD_full(1, to_Y(qubit1_T1, t))
        qubit2_Aks = Krauser.AD_full(1, to_Y(qubit2_T1, t))
        qubit3_Aks = Krauser.AD_full(1, to_Y(qubit3_T1, t))
        qubit1_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit1_T2, qubit1_T1, t))
        qubit2_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit2_T2, qubit2_T1, t))
        qubit3_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit3_T2, qubit3_T1, t))

        Aks = []
        for Ak1 in qubit1_Aks:
            for Ak2 in qubit2_Aks:
                for Ak3 in qubit3_Aks:
                    Aks.append(multi_kron(Ak1, Ak2, Ak3))
        Zks = []
        for Zk1 in qubit1_Zks:
            for Zk2 in qubit2_Zks:
                for Zk3 in qubit3_Zks:
                    Zks.append(multi_kron(Zk1, Zk2, Zk3))

        return Aks, Zks
    
    fids_bare = [1.0]
    fids_bare_stds = [0.0]
    state = np.array([0,1])
    rho = np.outer(state,state.conj().T)
    for t in times[1:]:
        qubit1_Aks = Krauser.AD_full(1, to_Y(qubit1_T1, t))
        qubit2_Aks = Krauser.AD_full(1, to_Y(qubit2_T1, t))
        qubit3_Aks = Krauser.AD_full(1, to_Y(qubit3_T1, t))
        qubit1_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit1_T2, qubit1_T1, t))
        qubit2_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit2_T2, qubit2_T1, t))
        qubit3_Zks = Krauser.Pauli_full(1, ['Z'], to_p(qubit3_T2, qubit3_T1, t))

        rho_1 = sum([Ek @ rho @ Ek.conj().T for Ek in qubit1_Aks])
        rho_1 = sum([Ek @ rho_1 @ Ek.conj().T for Ek in qubit1_Zks])
        rho_2 = sum([Ek @ rho @ Ek.conj().T for Ek in qubit2_Aks])
        rho_2 = sum([Ek @ rho_2 @ Ek.conj().T for Ek in qubit2_Zks])
        rho_3 = sum([Ek @ rho @ Ek.conj().T for Ek in qubit3_Aks])
        rho_3 = sum([Ek @ rho_3 @ Ek.conj().T for Ek in qubit3_Zks])
        
        data = [
            Fidelity.fid(rho_1, state),
            Fidelity.fid(rho_2, state),
            Fidelity.fid(rho_3, state)
        ]
        
        fids_bare.append(np.mean(data))
        fids_bare_stds.append(np.std(data))

    Aks_fixed, Zks_fixed = fluctuating_noise(relaxation_time)
    Eks_fixed = Krauser.AD(to_Y(T1, relaxation_time), True)
    Rks_fixed = Recovery.dutta(Eks_fixed, codes)

    u0, d0, vh0 = np.linalg.svd(Rks_fixed[0])
    u1, d1, vh1 = np.linalg.svd(Rks_fixed[1])
    d = np.diag([1,1,0,0,0,0,0,0])
    Rks_approx = [
        u0@d@vh0, u1@d@vh1,
    ]

    fids_0L = [1.0]
    probs_0L = [1.0]
    fids_1L = [1.0]
    probs_1L = [1.0]
    fids_multi_0 = [1.0]
    probs_multi_0 = [1.0]
    fids_multi_1 = [1.0]
    probs_multi_1 = [1.0]
    fids_multi_0_approx = [1.0]
    probs_multi_0_approx = [1.0]
    fids_multi_1_approx = [1.0]
    probs_multi_1_approx = [1.0]

    for t in times[1:]:
        Y = to_Y(T1, t)
        Eks = Krauser.AD(3, 1, Y, True)
        Aks, Zks = fluctuating_noise(t)
        Rks = Recovery.dutta(Eks, codes)

        rho_noisy_single_0L = sum([Ek @ rho_0L @ Ek.conj().T for Ek in Aks])
        rho_noisy_single_0L = sum([Ek @ rho_noisy_single_0L @ Ek.conj().T for Ek in Zks])
        rho_single_0L = sum([Rk @ rho_noisy_single_0L @ Rk.conj().T for Rk in Rks])
        
        rho_noisy_single_1L = sum([Ek @ rho_1L @ Ek.conj().T for Ek in Aks])
        rho_noisy_single_1L = sum([Ek @ rho_noisy_single_1L @ Ek.conj().T for Ek in Zks])
        rho_single_1L = sum([Rk @ rho_noisy_single_1L @ Rk.conj().T for Rk in Rks])

        total_qec = int(np.ceil(t/relaxation_time))
        rho_multi_0 = deepcopy(rho_0L)
        rho_multi_1 = deepcopy(rho_1L)
        rho_multi_0_approx = deepcopy(rho_0L)
        rho_multi_1_approx = deepcopy(rho_1L)
        for i in range(total_qec-1):
            rho_multi_0 = sum([Ek @ rho_multi_0 @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_0 = sum([Ek @ rho_multi_0 @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_0 = sum([Rk @ rho_multi_0 @ Rk.conj().T for Rk in Rks_fixed])
            rho_multi_1 = sum([Ek @ rho_multi_1 @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_1 = sum([Ek @ rho_multi_1 @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_1 = sum([Rk @ rho_multi_1 @ Rk.conj().T for Rk in Rks_fixed])
            rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_0_approx = sum([Rk @ rho_multi_0_approx @ Rk.conj().T for Rk in Rks_approx])
            rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_1_approx = sum([Rk @ rho_multi_1_approx @ Rk.conj().T for Rk in Rks_approx])
        remaining_time = np.round(t - relaxation_time*(total_qec-1), 5)
        Y = to_Y(T1, remaining_time)
        Eks = Krauser.AD(Y, True)
        Aks, Zks = fluctuating_noise(remaining_time)
        Rks = Recovery.dutta(Eks, codes)
        rho_multi_0 = sum([Ek @ rho_multi_0 @ Ek.conj().T for Ek in Aks])
        rho_multi_0 = sum([Ek @ rho_multi_0 @ Ek.conj().T for Ek in Zks])
        rho_multi_0 = sum([Rk @ rho_multi_0 @ Rk.conj().T for Rk in Rks])
        rho_multi_1 = sum([Ek @ rho_multi_1 @ Ek.conj().T for Ek in Aks])
        rho_multi_1 = sum([Ek @ rho_multi_1 @ Ek.conj().T for Ek in Zks])
        rho_multi_1 = sum([Rk @ rho_multi_1 @ Rk.conj().T for Rk in Rks])
        rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Aks])
        rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Zks])
        rho_multi_0_approx = sum([Rk @ rho_multi_0_approx @ Rk.conj().T for Rk in Rks_approx])
        rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Aks])
        rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Zks])
        rho_multi_1_approx = sum([Rk @ rho_multi_1_approx @ Rk.conj().T for Rk in Rks_approx])

        probs_0L.append(np.abs(np.trace(rho_single_0L)))
        fids_0L.append(Fidelity.fid(rho_single_0L/probs_0L[-1], state_0L))
        probs_1L.append(np.abs(np.trace(rho_single_1L)))
        fids_1L.append(Fidelity.fid(rho_single_1L/probs_1L[-1], state_1L))
        
        probs_multi_0.append(np.round(np.abs(np.trace(rho_multi_0)),10))
        fids_multi_0.append(np.round(Fidelity.fid(rho_multi_0/probs_multi_0[-1], state_0L),10))
        probs_multi_1.append(np.round(np.abs(np.trace(rho_multi_1)),10))
        fids_multi_1.append(np.round(Fidelity.fid(rho_multi_1/probs_multi_1[-1], state_1L),10))

        probs_multi_0_approx.append(np.round(np.abs(np.trace(rho_multi_0_approx)),10))
        fids_multi_0_approx.append(np.round(Fidelity.fid(rho_multi_0_approx/probs_multi_0_approx[-1], state_0L),10))
        probs_multi_1_approx.append(np.round(np.abs(np.trace(rho_multi_1_approx)),10))
        fids_multi_1_approx.append(np.round(Fidelity.fid(rho_multi_1_approx/probs_multi_1_approx[-1], state_1L),10))

    result = {
        'bare_qubit': {
            'fids': fids_bare,
            'fids_stds': fids_bare_stds
        },
        'multi_qec_0': {
            'fids': fids_multi_0,
            'probs': probs_multi_0,
        },
        'multi_qec_1': {
            'fids': fids_multi_1,
            'probs': probs_multi_1,
        },
        'multi_qec_0_approx': {
            'fids': fids_multi_0_approx,
            'probs': probs_multi_0_approx,
        },
        'multi_qec_1_approx': {
            'fids': fids_multi_1_approx,
            'probs': probs_multi_1_approx,
        }
    }

    return result

def get_th_approx_data(T1: float, T2: float, times: np.ndarray[float], relaxation_time:float):

    codes = _PI3()
    state_1L = codes[1]
    rho_1L = np.outer(state_1L, state_1L.conj().T)
    state_0L = codes[0]
    rho_0L = np.outer(state_0L, state_0L.conj().T)


    if times[0] == 0.0:
        fids_multi_0_approx = [1.0]
        probs_multi_0_approx = [1.0]
        fids_multi_1_approx = [1.0]
        probs_multi_1_approx = [1.0]
        fids_bare = [1.0]
        start = 1

    else:
        fids_multi_0_approx = []
        probs_multi_0_approx = []
        fids_multi_1_approx = []
        probs_multi_1_approx = []
        fids_bare = []
        start = 0

    state = np.array([0,1])
    rho = np.outer(state,state.conj().T)
    for t in times[start:]:
        qubit_Aks = Krauser.AD_full(1, to_Y(T1, t))
        qubit_Zks = Krauser.Pauli_full(1, ['Z'], to_p(T2, T1, t))

        rho_b = sum([Ek @ rho @ Ek.conj().T for Ek in qubit_Aks])
        rho_b = sum([Ek @ rho_b @ Ek.conj().T for Ek in qubit_Zks])
        
        fids_bare.append(Fidelity.fid(rho_b, state))

    
    Aks_fixed = Krauser.AD_full(3, to_Y(T1, relaxation_time))
    Zks_fixed = Krauser.Pauli_full(3, ['Z'], to_p(T2, T1, relaxation_time))
    Eks_fixed = Krauser.AD(3, 1, to_Y(T1, relaxation_time), True)
    Rks_fixed = Recovery.dutta(Eks_fixed, codes)

    u0, _, vh0 = np.linalg.svd(Rks_fixed[0])
    u1, _, vh1 = np.linalg.svd(Rks_fixed[1])
    d = np.diag([1,1,0,0,0,0,0,0])
    Rks_approx = [
        u0@d@vh0, u1@d@vh1,
    ]

    for t in times[start:]:
        Y = to_Y(T1, t)
        total_qec = int(np.ceil(t/relaxation_time))
        rho_multi_0_approx = deepcopy(rho_0L)
        rho_multi_1_approx = deepcopy(rho_1L)
        for i in range(total_qec-1):
            rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_0_approx = sum([Rk @ rho_multi_0_approx @ Rk.conj().T for Rk in Rks_approx])
            rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Aks_fixed])
            rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Zks_fixed])
            rho_multi_1_approx = sum([Rk @ rho_multi_1_approx @ Rk.conj().T for Rk in Rks_approx])
        remaining_time = np.round(t - relaxation_time*(total_qec-1), 5)
        Aks = Krauser.AD_full(3, to_Y(T1, remaining_time))
        Zks = Krauser.Pauli_full(3, ['Z'], to_p(T2, T1, remaining_time))
        rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Aks])
        rho_multi_0_approx = sum([Ek @ rho_multi_0_approx @ Ek.conj().T for Ek in Zks])
        rho_multi_0_approx = sum([Rk @ rho_multi_0_approx @ Rk.conj().T for Rk in Rks_approx])
        rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Aks])
        rho_multi_1_approx = sum([Ek @ rho_multi_1_approx @ Ek.conj().T for Ek in Zks])
        rho_multi_1_approx = sum([Rk @ rho_multi_1_approx @ Rk.conj().T for Rk in Rks_approx])

        probs_multi_0_approx.append(np.round(np.abs(np.trace(rho_multi_0_approx)),10))
        fids_multi_0_approx.append(np.round(Fidelity.fid(rho_multi_0_approx/probs_multi_0_approx[-1], state_0L),10))
        probs_multi_1_approx.append(np.round(np.abs(np.trace(rho_multi_1_approx)),10))
        fids_multi_1_approx.append(np.round(Fidelity.fid(rho_multi_1_approx/probs_multi_1_approx[-1], state_1L),10))

    return {
        'bare_qubit': {
            'fids': np.array(fids_bare)
        },
        'multi_qec_0_approx': {
            'fids': np.array(fids_multi_0_approx),
            'probs': np.array(probs_multi_0_approx),
        },
        'multi_qec_1_approx': {
            'fids': np.array(fids_multi_1_approx),
            'probs': np.array(probs_multi_1_approx)
        }
    }

def snr_exp(f, sigma, N):
    return (
        f/(np.sqrt(N)*sigma)
    )

def snr_th(f, p: np.ndarray = 1.0, f_measure: float = 1.0, n_measure: int = 4):

    f_ = f*(f_measure**n_measure) + (1-f)*((1-f_measure)**n_measure)
    f_std = np.sqrt(f_*(1-f_))

    return f_*np.sqrt(p)/f_std

