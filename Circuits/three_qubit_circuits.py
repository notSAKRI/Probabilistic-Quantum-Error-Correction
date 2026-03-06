import sys
sys.path.append('..')
from Circuits.general_circuits import *

# Encoder circuit for the three qubit PI code.
def PI3_encoder():
    qc = QuantumCircuit(3, name='Encoder')
    qc = QuantumCircuit(3)
    qc.rx(1.0471975511952163,0)
    qc.rx(2.1862760354653132,1)
    qc.rz(2.356194490231186,2)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(3.9269908169860313,0)
    qc.rx(0.9553166181137515,1)
    qc.rx(1.5707963267941292,2)
    qc.rz(4.712388980380326,2)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(2.356194490192739,0)
    qc.rx(1.5707963267948308,1)
    qc.rx(1.5707963267876839,2)
    qc.rz(3.6674656737989215,0)
    qc.rz(1.570796326797784,1)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(1.5707963267950869,1)
    qc.rx(1.5707963267938683,2)
    qc.rz(1.8303214700009482,0)
    qc.rz(2.356194490191206,1)
    qc.rz(0.7853981634215288,2)
    qc.name = 'Encoder'
    return qc

# Circuit for the approximate diagonal matrix of the recovery
def PI3_aprrox_d():
    # Total Time = 5*(CZ Time) + 6*(SX Time) = 532 ns = 133 dt 
    theta = np.array([2.35625852, 1.57079633, 2.35625853, 1.57079633, 4.71238898,
       3.92692678, 4.71238898, 0.78533413])
    qc = QuantumCircuit(3)
    qc.rx(theta[0],2)
    qc.cz(1,2)
    qc.rx(theta[1],1)
    qc.rx(theta[2],2)
    qc.cz(0,1)
    qc.rx(theta[3],1)
    qc.cz(1,2)
    qc.rx(theta[4],1)
    qc.rx(theta[5],2)
    qc.cz(0,1)
    qc.rx(theta[6],1)
    qc.cz(1,2)
    qc.rx(theta[7],2)
    qc.name = 'The D'
    return qc

# Parity check circuit
def PI3_zzz():
    qc = QuantumCircuit(4)

    qc.cx(1,0)
    qc.swap(1,2)
    qc.cx(1,0)
    qc.swap(2,3)
    qc.swap(1,2)
    qc.cx(1,0)

    qc.name = 'ZZZ'
    return qc

# Parity check and a small part of the Vh matrix. It should be used if we are not using control if.
def PI3_zzz_without_cif():
    # Total Time = 22*(CZ Time) + 27*(SX Time) = 2392 ns = 598 dt 
    qc = QuantumCircuit(4)
    
    qc.swap(0,1)
    qc.cx(2,1)
    qc.cx(0,1)
    qc.swap(2,3)
    qc.cx(2,1)
    qc.x(1)
    qc.cx(1,2)
    qc.cx(1,0)
    qc.swap(2,3)
    qc.cx(1,2)
    qc.swap(0,1)

    qc.name = 'ZZZ + CXXX'

    return qc

# U part of the circuit of the recovery matrix
def PI3_u(): 
    theta = np.array([2.0943951 , 0.95531662, 0.78539816, 0.78539816, 5.32786869,
       4.71238898, 1.57079633, 0.78539816, 1.57079633, 1.57079633,
       1.57079633, 1.57079633, 1.57079633, 0.78539816, 0.78539816,
       2.35619449])
    qc = QuantumCircuit(3)
    qc.rx(theta[0],0)
    qc.rx(theta[1],1)
    qc.rz(theta[2],2)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(theta[3],0)
    qc.rx(theta[4],1)
    qc.rx(theta[5],2)
    qc.rz(theta[6],1)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(theta[7],0)
    qc.rx(theta[8],1)
    qc.rz(theta[9],1)
    qc.rz(theta[10],2)
    qc.cz(1,2)
    qc.cz(0,1)
    qc.rx(theta[11],1)
    qc.rx(theta[12],2)
    qc.rz(theta[13],0)
    qc.rz(theta[14],1)
    qc.rz(theta[15],2)
    qc.name = 'U'
    return qc

# Vh part of the circuit of the recovery matrix
def PI3_v(): 
    theta = np.array([1.57079633, 1.57079633, 0.78539816, 1.57079633, 1.57079633,
       1.57079633, 2.35619449, 4.09690927, 4.71238898, 1.57079633,
       1.04719755, 0.95531662, 2.35619449, 2.35619449, 0.78539816,
       2.35619449])
    qc = QuantumCircuit(3)
    qc.rz(theta[12],0)
    qc.rz(theta[13],1)
    qc.rz(theta[14],2)
    qc.rx(theta[0],1)
    qc.rx(theta[1],2)
    qc.cz(0,1)
    qc.cz(1,2)
    qc.rx(theta[2],0)
    qc.rz(theta[4],1)
    qc.rx(theta[3],1)
    qc.rz(theta[5],2)
    qc.cz(0,1)
    qc.cz(1,2)
    qc.rx(theta[6],0)
    qc.rz(theta[9],1)
    qc.rx(theta[7],1)
    qc.rx(theta[8],2)
    qc.cz(0,1)
    qc.cz(1,2)
    qc.rx(theta[10],0)
    qc.rx(theta[11],1)
    qc.rz(theta[15],2)
    qc.name = 'Vh'
    return qc

# Returns the recovery circuit of the three qubit PI code.
# If Y is not specified it will return the arrprximate recovery circuit (i.e. Y = 0.0).
# If clbits is not specified the circuit won't have classical control.
def PI3_recovery(Y: float = 0.0, clbits: int = 0):

    if Y == 0.0:
        d_circ = PI3_aprrox_d()
    else:
        d_circ = QuantumCircuit(4)
        d_circ.append(
            RYGate(2.0*np.arcsin(1-Y)).control(num_ctrl_qubits = 3, ctrl_state = 1), d_circ.qubits
        )
        d_circ.append(
            RYGate(np.pi).control(num_ctrl_qubits = 3, ctrl_state = 0), d_circ.qubits
        )
        d_circ.x(3)
        d_circ.name = f"D({Y})"
    
    if clbits != 0:

        qc = QuantumCircuit(5, clbits)
        qc.append(PI3_zzz(), qc.qubits[:-1])
        qc.measure(0, 0)
        xxx = QuantumCircuit(3, name = 'XXX')
        xxx.x(0)
        xxx.x(1)
        xxx.x(2)
        x = QuantumCircuit(1, name = 'X')
        x.x(0)
        if_op_1 = IfElseOp((qc.clbits[0], 0), xxx)
        if_op_2 = IfElseOp((qc.clbits[0], 0), x)
        qc.append(if_op_1, qc.qubits[1:-1])
        qc.append(PI3_v(), qc.qubits[1:-1])
        qc.append(if_op_2, [1])
        if Y == 0.0:
            qc.append(d_circ, qc.qubits[2:])
        else:
            qc.append(d_circ, qc.qubits[1:])
        qc.append(PI3_u(), qc.qubits[1:-1])
    
    else:

        # Total time = 2392 + 632 + 532 + 632 = 4188 ns = 1047 dt
        qc = QuantumCircuit(5)
        qc.append(PI3_zzz_without_cif(), qc.qubits[:-1])
        qc.append(PI3_v(), qc.qubits[1:-1])
        qc.cx(0,1)
        if Y == 0.0:
            qc.append(d_circ, qc.qubits[2:])
        else:
            qc.append(d_circ, qc.qubits[1:])
        qc.append(PI3_u(), qc.qubits[1:-1])

    qc.name = 'Recovery'
    return qc
