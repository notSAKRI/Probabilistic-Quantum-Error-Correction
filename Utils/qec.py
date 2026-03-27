from numpy.linalg import multi_dot as MD
from scipy.optimize import minimize
from scipy import linalg as LA
from typing import List

import numpy as np

LNP = List[np.ndarray]

def rinv(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    evals, evecs = np.linalg.eigh(matrix)
    inv_sqrt_evals = np.zeros_like(evals)
    for i, val in enumerate(evals):
        if val > tol:
            inv_sqrt_evals[i] = 1.0 / np.sqrt(val)

    return MD([evecs, np.diag(inv_sqrt_evals), evecs.conj().T])


class Recovery:
    # Ak
    def leung(error_kraus: LNP, codes: LNP) -> LNP:
        P = sum([np.outer(state, state.conj().T) for state in codes])
        Rks = []
        for Ek in error_kraus:
            Uk, _ = LA.polar(np.dot(Ek, P), side="right")
            Rks.append(np.dot(P, Uk.conj().T))

        return Rks

    # Ak
    def cafaro(error_kraus: LNP, codes: LNP) -> LNP:
        Rks = []
        for Ek in error_kraus:
            Rks.append(
                sum(
                    [
                        np.dot(np.outer(state, state.conj().T), Ek.conj().T)
                        / np.sqrt(MD([state.conj().T, Ek.conj().T, Ek, state]))
                        for state in codes
                    ]
                )
            )
        return Rks

    # Ak_
    def petz(kraus: LNP, codes: LNP) -> LNP:
        P = sum([np.outer(state, state.conj().T) for state in codes])
        channel = sum([MD([Ek, P, Ek.conj().T]) for Ek in kraus])
        norm = LA.fractional_matrix_power(channel, -0.5)

        return [MD([P, Ek.conj().T, norm]) for Ek in kraus]

    @staticmethod # group=True
    def dutta(error_kraus: LNP, codes: LNP) -> list[np.ndarray]:
        Rks = []
        for Eks in error_kraus:
            Rk = []
            for i in codes:
                chis = []
                for En in Eks:
                    chis.append(
                        sum([MD([i.conj().T, Em.conj().T, En, i]) for Em in Eks])
                    )
                X_av = np.average(chis, weights=[Eks[j].P for j in range(len(chis))])
                # X_av = np.max(chis)
                Rk.append(
                    sum([np.outer(i, np.dot(Em, i).conj().T) for Em in Eks]) / X_av
                )
            Rk = sum(Rk)

            Rks.append(Rk / np.sqrt(np.linalg.eigvalsh(np.dot(Rk.conj().T, Rk))[-1]))

        return Rks

    @staticmethod
    def dutta_returns(error_kraus: LNP, codes: LNP) -> list[np.ndarray]:
        Rks = []
        P = sum([np.outer(i, i.conj().T) for i in codes])

        for Eks in error_kraus:
            M = np.zeros_like(P, dtype=complex)

            for Em in Eks:
                M += MD([P, Em.conj().T, Em, P])

            # Invert M strictly within the code subspace
            evals, evecs = np.linalg.eigh(M)
            inv_evals = np.zeros_like(evals)
            for idx, val in enumerate(evals):
                if val > 1e-14:
                    inv_evals[idx] = 1.0 / val
            M_inv = MD([evecs, np.diag(inv_evals), evecs.conj().T])

            Rk = np.zeros_like(P, dtype=complex)
            for Em in Eks:
                Rk += MD([M_inv, P, Em.conj().T])

            max_eig = np.linalg.eigvalsh(np.dot(Rk.conj().T, Rk))[-1]
            if max_eig > 0:
                Rks.append(Rk / np.sqrt(max_eig))
            else:
                Rks.append(Rk)

        return Rks


    # Universal Probabilistic Petz Recovery Map.
    @staticmethod # group=True
    def universal(error_kraus_grouped: list[LNP], codes: LNP) -> LNP:
        P = sum([np.outer(state, state.conj().T) for state in codes])
        Rks_flat = []

        for Eks in error_kraus_grouped:
            rho_a = sum([MD([Em, P, Em.conj().T]) for Em in Eks])
            rho_a_inv_sqrt = rinv(rho_a)

            # 3. Petz ops for this branch
            for Em in Eks:
                # R_m = P * E_m^dagger * rho_a^{-1/2}
                Rm = MD([P, Em.conj().T, rho_a_inv_sqrt])
                Rks_flat.append(Rm)

        return Rks_flat

    def dutta_projectors(error_kraus: LNP, codes: LNP):

        Ps = []
        I = np.eye(len(codes[0]))

        for Eks in error_kraus:
            remains = np.zeros_like(codes[0])
            for En in Eks:
                remains += sum([np.dot(En, state) for state in codes])
            P = np.zeros_like(error_kraus[0][0])
            remains = np.round(remains, 5)
            for i in range(len(remains)):
                if remains[i] > 0.0:
                    P += np.outer(I[i], I[i].conj().T)
            Ps.append(P)

        return Ps

    @staticmethod
    def make_tp(Rks: LNP) -> LNP:

        O = sum([(Rks[i].conj().T @ Rks[i]) for i in range(len(Rks))])
        return Rks / np.sqrt(np.linalg.eigvalsh(O)[-1])


class Fidelity:
    # Entanglement fidelity calculation using Cafaro's formula
    def cafaro(Rks: LNP, Eks: LNP, code_words: LNP) -> float:
        f_ent = 0.0
        for Al in Eks:
            for Rk in Rks:
                f_ent += (
                    np.sum(
                        [MD([state.conj().T, Rk, Al, state]) for state in code_words]
                    )
                    ** 2
                )

        return np.real(f_ent / (len(code_words) ** 2))

    def pure_state_fidelity(Rks: LNP, Eks: LNP, state: np.ndarray) -> float:

        rho = np.outer(state, state.conj().T)
        rho = sum([MD([Ek, rho, Ek.conj().T]) for Ek in Eks])
        rho = sum([MD([Rk, rho, Rk.conj().T]) for Rk in Rks])

        return np.abs(MD([state.conj().T, rho, state]))


    def entanglement(R_kraus: LNP, E_kraus: LNP, codes: LNP) -> float:
        l = len(codes)
        R = np.eye(l)

        QR = (1 / np.sqrt(l)) * sum([np.kron(codes[i], R[i]) for i in range(l)])

        rho = np.outer(QR, QR.conj().T)

        Eks = [np.kron(Ek, R) for Ek in E_kraus]
        Rks = [np.kron(Rk, R) for Rk in R_kraus]

        rho_new = sum([MD([Ek, rho, Ek.conj().T]) for Ek in Eks])
        rho_new = sum([MD([Rk, rho_new, Rk.conj().T]) for Rk in Rks])

        p = np.trace(rho_new)
        rho_new /= p

        fid = np.dot(QR.conj().T, np.dot(rho_new, QR))
        return np.abs(fid), np.abs(p)

    def _worst_case_fid_cost(parameters, Rks: LNP, Eks: LNP, codes: LNP):

        [theta, phi] = parameters
        state = (
            np.cos(theta / 2.0) * codes[0]
            + np.sin(theta / 2.0) * np.exp(1j * phi) * codes[1]
        )
        rho = np.outer(state, state.conj().T)
        rho = sum([MD([Ek, rho, Ek.conj().T]) for Ek in Eks])
        rho = sum([MD([Rk, rho, Rk.conj().T]) for Rk in Rks])

        rho /= np.trace(rho)
        fid = MD([state.conj().T, rho, state])

        return np.abs(fid)

    def worst_case_fid(
        R_kraus: LNP,
        E_kraus: LNP,
        codes: LNP,
        tol: float = 1e-5,
    ) -> float:

        result = minimize(
            fun=Fidelity._worst_case_fid_cost,
            x0=np.random.rand(2),
            args=(R_kraus, E_kraus, codes),
            method="Powell",
            bounds=[(0, np.pi), (0, 2 * np.pi)],
            tol=tol,
        )

        return result.fun, result.x
    
    def fid(rho: np.ndarray, state: np.ndarray) -> float:
        return np.abs(MD([state.conj().T, rho, state]))
