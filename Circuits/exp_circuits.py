import sys
sys.path.append('..')
from Utils.CHaDD_utils import *
from Utils.simulator import *
from Circuits.three_qubit_circuits import *

# ------------- All delays are in dt units ---------------


# Returns the PI encoded state. Default is the state |1L>
def encoder(state: str = '1'):

    if state == '1_simplified':
        qc = QuantumCircuit(3, name = '|1L> Encoder')
        qc.x([0,1,2])
        return qc

    # 428 ns
    elif state == '0_simplified':
        qc = QuantumCircuit(3, name = '|0L> Encoder')
        qc = QuantumCircuit(3)
        qc.ry(2*np.arccos(1/np.sqrt(3)), 0)
        qc.ch(0,1)
        qc.cx(1,2)
        qc.cx(0,1)
        qc.x(0)
        return qc

    elif state == '1':
        qc = QuantumCircuit(3, name = '|1L> Encoder')
        qc.x(0)
    
    elif state == '0':
        qc = QuantumCircuit(3, name = '|0L> Encoder')

    elif state == '+':
        qc = QuantumCircuit(3, name = '|+L> Encoder')
        qc.h(0)

    elif state == '-':
        qc = QuantumCircuit(3, name = '|-L> Encoder')
        qc.x(0)
        qc.h(0)
    
    else:
        raise ValueError('Invalid state')
    
    qc.append(dutta3_encoder(), qc.qubits)
    return qc

# T1 exp circuit
def bare_qubit_T1_circ(delay: int):
    qc = QuantumCircuit(1, 1, name = 'Bare_T1')
    qc.x(0)
    qc.barrier()
    qc.delay(int(delay), 0)
    qc.barrier()
    qc.x(0)
    qc.measure(qc.qubits,qc.clbits)

    return qc

# T2 exp circuit (Hahn Echo)
def bare_qubit_T2_circ(num_echoes: int, delay: int):
    hahn_delay = int(np.round( delay/(num_echoes + 1.0)))
    qc = QuantumCircuit(1, 1, name = 'Bare_T2')

    qc.rx(np.pi / 2, 0)  # Brings the qubit to the X Axis
    qc.delay(hahn_delay, 0)

    for _ in range(num_echoes):
        qc.rx(np.pi, 0)
        qc.delay(hahn_delay, 0)
        
    if num_echoes % 2 == 1:
        qc.rx(np.pi / 2, 0)  # X90 again since the num of echoes is odd
    else:
        qc.rx(-np.pi / 2, 0)  # X(-90) again since the num of echoes is even
                    
    qc.measure(qc.qubits,qc.clbits)

    return qc

# Single QEC circuit. Implements the approximate recovery circuit.
# delay: int - delay time (in dt of the system)
# flag: int - Takes only 0 and 1. Flag to indicate whether to use the classical controlled recovery circuit or not.
def dutta3_single_qec_circ(enc: QuantumCircuit, delay: int, flag: int = 0):
    
    if flag == 1:
        Rk = dutta3_recovery(clbits = 1)
        carg = [0]
    else:
        Rk = dutta3_recovery()
        carg = []

    qc = QuantumCircuit(5,4 + flag, name = 'Single_QEC')
    qc.append(enc, qc.qubits[1:-1])

    qc.barrier()
    qc.delay(int(delay), qc.qubits[1:-1])
    qc.barrier()

    qc.append(Rk, qargs=qc.qubits, cargs=carg)
    qc.append(enc.inverse(), qc.qubits[1:-1])
    qc.measure(qc.qubits[1:], qc.clbits[flag:])

    return qc

# Multi QEC circuit. Implements the approximate recovery circuit.
# delay: int - delay time (in dt of the system)
# flag: int - Takes only 0 and 1. Flag to indicate whether to use the classical controlled recovery circuit or not.
def dutta3_multi_qec_circ(enc: QuantumCircuit, delay: int, qec_cycle_relaxation: int, rec_time: int, flag: int = 0):
    total_qec = int(np.ceil(delay/qec_cycle_relaxation))
    if total_qec == 0:
        total_qec = 1
    if (delay - (total_qec - 1)*qec_cycle_relaxation) < rec_time and total_qec > 1:
        total_qec -= 1

    if flag == 1:
        Rk = dutta3_recovery(clbits = 1)
        carg = [0]
    else:
        Rk = dutta3_recovery()
        carg = []

    qc = QuantumCircuit(5, 3+flag+total_qec)

    qc.append(enc, qc.qubits[1:-1])
    qc.barrier()

    for i in range(total_qec-1):
        qc.delay(qec_cycle_relaxation, qc.qubits[1:4])
        qc.barrier()

        qc.append(Rk, qargs=qc.qubits, cargs=carg)
        
        qc.measure(4, 3+flag+i)
        qc.barrier()
        qc.reset([0,4])

    remaining_time = delay - qec_cycle_relaxation*(total_qec-1)
    qc.delay(int(remaining_time), [1,2,3])
    qc.barrier()

    qc.append(Rk, qargs=qc.qubits, cargs=carg)
    qc.measure(4, 2+flag+total_qec)
    qc.append(enc.inverse(), qc.qubits[1:-1])

    qc.measure(qc.qubits[1:4], qc.clbits[flag:3+flag])

    return qc

# Implements the actual recovery circuit with some finite Y in the D matrix.
# delay: int - delay time (in dt of the system)
# flag: int - Takes only 0 and 1. Flag to indicate whether to use the classical controlled recovery circuit or not.
# estimated_T1: float - Needed to calculate the Y then generate the D part of the recovery circuit.
def dutta3_single_qec_actual_circ(enc: QuantumCircuit, delay: int, flag: int = 0, estimated_T1: float = 200):
    
    if flag == 1:
        Rk = dutta3_recovery(clbits = 1)
        carg = [0]
    else:
        Rk = dutta3_recovery(to_Y(estimated_T1, delay))
        carg = []

    qc = QuantumCircuit(5,4 + flag, name = 'Single_QEC')
    qc.append(enc, qc.qubits[1:-1])

    qc.barrier()
    qc.delay(int(delay), qc.qubits[1:-1])
    qc.barrier()

    qc.append(Rk, qargs=qc.qubits, cargs=carg)
    qc.append(enc.inverse(), qc.qubits[1:-1])
    qc.measure(qc.qubits[1:], qc.clbits[flag:])

    return qc