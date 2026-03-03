import numpy as np
from scipy.linalg import fractional_matrix_power, polar
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.circuit import IfElseOp
from qiskit.quantum_info import Operator

# --- General Circuit generation for a random matrix M --

# Generates a circuit for a random matrix M using SVD
# Note that, if the ancilla is 0, then it's a successful implementation in this case
def svd_circuit(M: np.ndarray, name: str = 'U'):
   n = int(np.log2(len(M)))
   u, d, vh = np.linalg.svd(M)
   qc = QuantumCircuit(n+1, name=name)
   qc.append(Operator(vh), qc.qubits[:-1])
   for i in range(len(d)-1, -1, -1):
       if np.round(d[i],5) != 0.0:
           qc.append(
               RYGate(2.0*np.arcsin(d[i])).control(num_ctrl_qubits = n, ctrl_state = i), qc.qubits
           )
   qc.x(qc.qubits[-1])
   qc.append(Operator(u), qc.qubits[:-1])
   qc.name = name
   return qc

# Encodes a matrix M into a block matrix
def block_encode(M: np.ndarray) -> np.ndarray:
    N = fractional_matrix_power(np.eye(len(M)) - M @ M, 0.5)
    return np.vstack([
        np.hstack([M, -N]),  # Top block: [M | -N]
        np.hstack([N, M])    # Bottom block: [N | M]
    ])


# Generates a circuit for a random matrix M using Polar Decomposition
# Note that, if the ancilla is 0, then it's a successful implementation in this case
def polar_circuit(M: np.ndarray, name: str ='U'):

    u, v = polar(M)
    v_big = block_encode(v)
    qc = QuantumCircuit(np.log2(len(v_big)), name = name )
    qc.append(Operator(v_big), qc.qubits)
    qc.append(Operator(u), qc.qubits[:-1])
    qc.name = name
    return qc
