import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties

# Linear Connectivity of qubits
def linear_coupling(n):
    map = []
    for i in range(n-1):
        map.append([i, i+1])
        map.append([i+1, i])
    return map

# All to All Conncectivity for the qubits
def all_to_all(n):
    map = []
    for i in range(n):
        for j in range(n):
            if i != j:
                map.append([i, j])

    return map

# Fake Backend that can be used to simulate the protocol
# n: int - number of qubits
# coupling_map: list[tuple] - qubit connectivity. By default it generates a linear connectivity
# gate_error: bool - Adds gate noise
# processor_type: str - Type of processor. Currently supported: heron_r1, heron_r2, eagle_r3 processors of IBM
def fake_backend(n: int, coupling_map: list[tuple] = [], gate_error: bool = True, processor_type: str = 'heron_r1'):


    if processor_type == 'heron_r1':
        instructions = {
            'reset': {'duration': 2.72e-06, 'error': None, 'deviation': 0},
            'x': {'duration': 3.2e-08, 'error': 0.0003, 'deviation': 2e-5},
            'cz': {'duration': 6.8e-08, 'error': 0.005, 'deviation': 1e-3},
            'measure': {'duration': 1.56e-06, 'error': 0.028, 'deviation': 4e-3}
        }
        instruction_names = ['reset', 'x', 'sx', 'rx', 'id', 'measure', 'cz', 'rz']
    
    elif processor_type == 'heron_r2':
        instructions = {
            'reset': {'duration': 2.72e-06, 'error': None, 'deviation': 0}, # Still don't know reset time (expected to be higher)
            'x': {'duration': 3.2e-08, 'error': 0.00025, 'deviation': 2e-5},
            'cz': {'duration': 6.8e-08, 'error': 0.002, 'deviation': 4e-4},
            'measure': {'duration': 2.6e-06, 'error': 0.007, 'deviation': 1e-3}
        }
        instruction_names = ['reset', 'x', 'sx', 'rx', 'id', 'measure', 'cz', 'rz']
    
    elif processor_type == 'eagle_r3':
        instructions = {
            'reset': {'duration': 1.86e-06, 'error': None, 'deviation': 0},
            'x': {'duration': 6e-08, 'error': 0.0003, 'deviation': 5e-5},
            'ecr': {'duration': 6.6e-07, 'error': 0.007, 'deviation': 2e-3},
            'measure': {'duration': 1.3e-06, 'error': 0.02, 'deviation': 5e-3}
        }
        instruction_names = ['reset', 'x', 'sx', 'id', 'measure', 'ecr', 'rz']
    
    else:
        raise Exception('Known Processors: heron_r1, heron_r2, eagle_r3')

    if gate_error == False:
        for key in instructions:
            if key != 'reset':
                instructions[key]['error'] = 0.0
                instructions[key]['deviation'] = 0.0
        
    if len(coupling_map) == 0:
        coupling_map = linear_coupling(n)

    backend = GenericBackendV2(
        num_qubits = n,
        basis_gates = instruction_names,
        coupling_map= coupling_map,
        noise_info = True,
        control_flow = True
    )

    qargs_local = list(backend.target['x'].items())
    qargs_local = [qargs_local[i][0] for i in range(len(qargs_local))]

    if 'heron' in processor_type:
        qargs_coup = list(backend.target['cz'].items())
        qargs_coup = [qargs_coup[i][0] for i in range(len(qargs_coup))]

        for qarg in qargs_local:
            if instructions['reset']['error'] is not None:
                prop_reset = InstructionProperties(instructions['reset']['duration'], np.random.normal(instructions['reset']['error'], instructions['reset']['deviation']))
            else:
                prop_reset = InstructionProperties(instructions['reset']['duration'], instructions['reset']['error'])
            prop_x = InstructionProperties(instructions['x']['duration'], np.random.normal(instructions['x']['error'], instructions['x']['deviation']))
            prop_meas = InstructionProperties(instructions['measure']['duration'], np.random.normal(instructions['measure']['error'], instructions['measure']['deviation']))
            backend.target.update_instruction_properties('reset', qarg, prop_reset)
            backend.target.update_instruction_properties('x', qarg, prop_x)
            backend.target.update_instruction_properties('rx', qarg, prop_x)
            backend.target.update_instruction_properties('sx', qarg, prop_x)
            backend.target.update_instruction_properties('id', qarg, prop_x)
            backend.target.update_instruction_properties('measure', qarg, prop_meas)
        for qarg in qargs_coup:
            prop_cz = InstructionProperties(instructions['cz']['duration'], np.random.normal(instructions['cz']['error'], instructions['cz']['deviation']))
            backend.target.update_instruction_properties('cz', qarg, prop_cz)
        
        backend.target.dt = 4e-09

    elif 'eagle' in processor_type:
        qargs_coup = list(backend.target['ecr'].items())
        qargs_coup = [qargs_coup[i][0] for i in range(len(qargs_coup))]

        for qarg in qargs_local:
            if instructions['reset']['error'] is not None:
                prop_reset = InstructionProperties(instructions['reset']['duration'], np.random.normal(instructions['reset']['error'], instructions['reset']['deviation']))
            else:
                prop_reset = InstructionProperties(instructions['reset']['duration'], instructions['reset']['error'])
            prop_x = InstructionProperties(instructions['x']['duration'], np.random.normal(instructions['x']['error'], instructions['x']['deviation']))
            prop_meas = InstructionProperties(instructions['measure']['duration'], np.random.normal(instructions['measure']['error'], instructions['measure']['deviation']))
            backend.target.update_instruction_properties('reset', qarg, prop_reset)
            backend.target.update_instruction_properties('x', qarg, prop_x)
            backend.target.update_instruction_properties('sx', qarg, prop_x)
            backend.target.update_instruction_properties('id', qarg, prop_x)
            backend.target.update_instruction_properties('measure', qarg, prop_meas)
        for qarg in qargs_coup:
            prop_cz = InstructionProperties(instructions['ecr']['duration'], np.random.normal(instructions['ecr']['error'], instructions['ecr']['deviation']))
            backend.target.update_instruction_properties('ecr', qarg, prop_cz)
        
        backend.target.dt = 5e-10
    
    else:
        raise Exception('Known Processors: heron_r1, heron_r2, eagle_r3')
    
    return backend

def get_data(result):
    return result.result()[0].data.c.get_counts()

# Returns the gamma or strength of AD noise for the given T1 and t
def to_Y(T1: float, t: float):
    return (1 - np.exp(-(t/T1)))

# Returns the dephasing strength for the given T2 and T1 and t. T1 and T2 is required to calculate the T_phi.
def to_p(T2: float, T1: float, t: float):
    gamma_phi = (1/T2) - (1/(2*T1))
    return (1 - np.exp(-gamma_phi*t))/2.0

# Code to fix the T1 and T2 properties of the qubits of the fake backend
def fix_qubit_properties(backend: GenericBackendV2, T1: float = 0.0, T2: float = 0.0):
    for i in range(backend.num_qubits):
        if T1 != 0.0:
            backend.target.qubit_properties[i].t1 = T1
        if T2 != 0.0:
            backend.target.qubit_properties[i].t2 = T2