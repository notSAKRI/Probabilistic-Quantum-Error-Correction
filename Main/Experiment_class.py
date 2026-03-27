import sys, os, json
sys.path.append('..')
from Circuits.experiment_circuits import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
from qiskit_experiments.library import LocalReadoutError
from qiskit.result import Counts
from qiskit_experiments.data_processing import LocalReadoutMitigator

# The IBM-Heron timings are used in this class. 
class PIcode_experiment:

    # Suggested to not use any mitigation techniques. Replace with any other mitigation techniques if needed.
    def __init__(self, backend = None, use_cif: bool = False, use_mitigation: bool = False, encoder_time: int = 548, recovery_time: int = 3072):

        self.backend = backend
        self.dt = 4e-09
        if backend is not None:
            self.dt = backend.dt
            self.x_time = int(backend.target['x'][(0,)].duration/self.dt)
            self.reset_time = int(backend.target['reset'][(0,)].duration/self.dt)
            self.min_dd_delay = self.x_time
            
        self.use_cif = use_cif
        self.use_mitigation = use_mitigation
        self.mitigator_data = {}
        self.encoder_time = int(np.round(encoder_time/self.dt))
        self.recovery_time = int(np.round(recovery_time/self.dt))

        self.storage = {}

    # A slight modification with the timings, only for this experiment class
    def changed_dd_sequence(self, delay, W, W_row, robust: bool = False):

        n = W.shape[0]

        if robust:
            n*=2
            W = W.tolist()
            for i in range(len(W)): 
                W[i] = W[i] + W[i]
        
        min_time = n*(self.x_time + self.min_dd_delay)
        qc = QuantumCircuit(1)
        remaining_time = self.reset_time

        if delay < (min_time + self.reset_time):
            return qc
        
        elif robust:
            dd_delay = int(np.round(delay/n))
            if remaining_time > dd_delay:
                remaining_time -= dd_delay
            else:
                qc.delay(dd_delay - remaining_time, 0)
                remaining_time = 0
            sequence = W[W_row]
            sign = 0
            for i in range(1, n):
                if sequence[i] == 1:
                    if sequence[i-1] == -1:
                        if sign == 0 or sign == 3:
                            qc.rx(np.pi, 0)
                        else:
                            qc.rx(-np.pi, 0)
                        sign = (sign+1)%4
                    else:
                        if remaining_time > self.x_time:
                            remaining_time -= self.x_time
                        else:
                            qc.delay(self.x_time - remaining_time, 0)
                            remaining_time = 0
                else:
                    if sequence[i-1] == 1:
                        if sign == 0 or sign == 3:
                            qc.rx(np.pi, 0)
                        else:
                            qc.rx(-np.pi, 0)
                        sign = (sign+1)%4
                    else:
                        if remaining_time > self.x_time:
                            remaining_time -= self.x_time
                        else:
                            qc.delay(self.x_time - remaining_time, 0)
                            remaining_time = 0
                    
                if remaining_time > dd_delay:
                    remaining_time -= dd_delay
                else:
                    qc.delay(dd_delay - remaining_time, 0)
                    remaining_time = 0
                if remaining_time > dd_delay:
                    remaining_time -= dd_delay
                else:
                    qc.delay(dd_delay - remaining_time, 0)
                    remaining_time = 0
                    
            if sequence[n-1] == -1:
                qc.rx(sign*np.pi, 0)
            
        else:
            dd_delay = int(np.round(delay/n))
            if remaining_time > dd_delay:
                remaining_time -= dd_delay
            else:
                qc.delay(dd_delay - remaining_time, 0)
                remaining_time = 0
            sequence = W[W_row]
            for i in range(1, n):
                if sequence[i] == 1:
                    if sequence[i-1] == -1:
                        qc.rx(np.pi, 0)
                    else:
                        if remaining_time > self.x_time:
                            remaining_time -= self.x_time
                        else:
                            qc.delay(self.x_time - remaining_time, 0)
                            remaining_time = 0
                else:
                    if sequence[i-1] == 1:
                        qc.rx(np.pi, 0)
                    else:
                        if remaining_time > self.x_time:
                            remaining_time -= self.x_time
                        else:
                            qc.delay(self.x_time - remaining_time, 0)
                            remaining_time = 0
                if remaining_time > dd_delay:
                    remaining_time -= dd_delay
                else:
                    qc.delay(dd_delay - remaining_time, 0)
                    remaining_time = 0

            if sequence[n-1] == -1:
                qc.rx(np.pi, 0)

        return qc
        
    # T1 Experiment
    def add_bare_qubit_circs(
            self, 
            delays: list[int], qubit_pos: list[int], 
            shots: int = 2**11
    ):
        
        tag = f'bare_qubit'
        self.storage[tag] = {'delays': delays}
        delays = np.round((delays*1e-6)/self.dt)
        circs = []
        l = len(qubit_pos)

        for delay in delays:
            circ = bare_qubit_T1_circ(delay)
            qc = QuantumCircuit(l, l)
            for k in range(l):
                qc.compose(circ, qubits = [k], clbits = [k], inplace=True)

            qc = transpile(qc, backend = self.backend, optimization_level=0, routing_method = 'none', initial_layout = qubit_pos, layout_method='trivial')

            circs.append((qc,None,shots))

        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['shots'] = shots

    # T2 Experiment
    def T2_Hahn(
            self, 
            delays: list[int], qubit_pos: list[int], 
            num_echoes: int = 1, 
            shots: int = 2**11
    ):

        if num_echoes > 0:
            tag = f'T2_Hahn'
        else:
            tag = f'T2_Ramsey'
        self.storage[tag] = {'delays': delays}
        delays = np.round((delays*1e-6)/self.dt)
        circs = []

        for delay in delays:

            circ = bare_qubit_T2_circ(num_echoes, delay)
            
            qc = QuantumCircuit(len(qubit_pos),len(qubit_pos))
            for k in range(len(qubit_pos)):
                qc.compose(circ, qubits = [k], clbits = [k], inplace=True)

            qc = transpile(qc, backend = self.backend, optimization_level=0, routing_method = 'none', initial_layout = qubit_pos, layout_method='trivial')

            circs.append((qc,None,shots))
        
        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['shots'] = shots

    # PI Single-QEC experiment
    # The inputs:
    # delays: list[int] - list of delay times (in microseconds)
    # qubit_pos: list[list[int]] - list of set of qubits used for the protocol. Each set should have 5 qubits
    # state: str - state of the qubits (default is '1') Options: '1', '0', '+', '-'.
    # tag: str - tag for the experiment (default is 'single_qec'). Used for retrieving the results.
    # shots: int - number of shots for each experiment (default is 2**11).
    def add_single_qec_circs(
            self, 
            delays: list[int], qubit_pos: list[list[int]], 
            state: str = '1', 
            tag: str = None, 
            shots: int = 2**11
    ):

        if tag is None:
            tag = f'single_qec'

        if isinstance(qubit_pos[0], int):
            qubit_pos = [qubit_pos]
        
        physical_qubits = []
        for q_set in qubit_pos:
            physical_qubits += q_set

        total_set = len(qubit_pos)
        
        if self.use_cif:
            flag = 1
        else:
            flag = 0

        self.storage[tag] = {'delays': delays}
        delays = np.round((delays*1e-6)/self.dt)
        circs = []
        enc = encoder(state)

        for delay in delays:
            circ = PI3_single_qec_circ(enc, delay, flag)
            num_clbits = circ.num_clbits
            qc = QuantumCircuit(len(physical_qubits), total_set*num_clbits)
            for k in range(len(qubit_pos)):
                qc.compose(circ, qubits = qc.qubits[k*5:(k+1)*5], clbits = qc.clbits[k*num_clbits:(k+1)*num_clbits], inplace=True)

            qc = transpile(qc, backend = self.backend, optimization_level = 3, routing_method = 'none', initial_layout = physical_qubits, layout_method='trivial')

            circs.append((qc,None,shots))

        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['shots'] = shots

    # PI Multi-QEC experiment
    # The inputs:
    # delays: list[int] - list of delay times (in microseconds)
    # qubit_pos: list[list[int]] - list of set of qubits used for the protocol. Each set should have 5 qubits
    # state: str - state of the qubits (default is '1') Options: '1', '0', '+', '-'.
    # qec_cycle_relaxation: int - maximum relaxation time in between two QEC cycles. 
    # tag: str - tag for the experiment (default is 'multi_qec_<qec_cycle_relaxation>'). Used for retrieving the results.
    # shots: int - number of shots for each experiment (default is 2**13).
    def add_multi_qec_circs(
            self, 
            delays: list[int], qubit_pos: list[int], 
            state: str = '1', 
            qec_cycle_relaxation: int = 10, 
            tag: str = None, 
            shots: int = 2**13
    ):

        if tag is None:
            tag = f'multi_qec_{qec_cycle_relaxation}'

        if isinstance(qubit_pos[0], int):
            qubit_pos = [qubit_pos]
        
        physical_qubits = []
        for q_set in qubit_pos:
            physical_qubits += q_set

        total_set = len(qubit_pos)
        
        if self.use_cif:
            flag = 1
        else:
            flag = 0

        self.storage[tag] = {'delays': delays}
        delays = np.round((delays*1e-6)/self.dt)
        qec_cycle_relaxation = int(np.round(qec_cycle_relaxation*1e-6/self.dt))
        circs = []
        enc = encoder(state)
        rec_time = self.recovery_time

        for delay in delays:
            
            delay = int(delay)
            circ = PI3_multi_qec_circ(enc, delay, qec_cycle_relaxation, rec_time, flag)
            num_clbits = circ.num_clbits
            qc = QuantumCircuit(len(physical_qubits), total_set*num_clbits)
            for k in range(len(qubit_pos)):
                qc.compose(circ, qubits = qc.qubits[k*5:(k+1)*5], clbits = qc.clbits[k*num_clbits:(k+1)*num_clbits], inplace=True)

            qc = transpile(qc, backend = self.backend, optimization_level = 3, routing_method = 'none', initial_layout = physical_qubits, layout_method='trivial')
            circs.append((qc,None,shots))

        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['shots'] = shots

    # Pi Single-QEC experiment with CHaDD
    # The inputs:
    # delays: list[int] - list of delay times (in microseconds)
    # qubit_pos: list[list[int]] - list of set of qubits used for the protocol. Each set should have 5 qubits
    # state: str - state of the qubits (default is '1') Options: '1', '0', '+', '-'.
    # spectator_qubits: list[int] - list of qubits which will take part in the CHaDD protocol. 
    # If the qubits in the qubit_pos is a subset of spectator_qubits, no new color will be assigned to the qubits in the qubit_pos, i.e. 2 colored CHaDD will be implemented.
    # tag: str - tag for the experiment (default is 'single_qec_with_chadd'). Used for retrieving the results.
    # shots: int - number of shots for each experiment (default is 2**11).
    def add_single_qec_with_chadd_circs(
            self, 
            delays: list[int], qubit_pos: list[list[int]], 
            colors: dict, 
            state: str = '1', 
            spectator_qubits: list[int] = [], 
            tag: str = None, 
            shots: int = 2**11
    ):
        if tag is None:
            tag = f'single_qec_with_chadd'
            
        total_qubits = len(colors['0']) + len(colors['1'])

        if isinstance(qubit_pos[0], int):
            qubit_pos = [qubit_pos]

        total_set = len(qubit_pos)

        data_qubits = []
        ancilla_qubits = []

        for qubit_set in qubit_pos:
            if set(qubit_set).issubset(set(spectator_qubits)):
                colors = colors
            elif qubit_set[0] in spectator_qubits and qubit_set[4] in spectator_qubits:
                colors = assign_color(qubit_set[1:-1], total_qubits, colors)
            else:
                colors = assign_color(qubit_set, total_qubits, colors)

            data_qubits += qubit_set[1:4]
            ancilla_qubits += [qubit_set[0], qubit_set[4]]

        if len(spectator_qubits) == 0:
            spectator_qubits = colors['0'] + colors['1']

        self.storage[tag] = {'delays': delays}

        W = W_matrix(len(colors))
        delays = (delays*1e-6)/self.dt
        circs = []  

        enc = encoder(state)
        if state == '1_simplified':
            enc_time = self.x_time
        elif state == '0_simplified':
            enc_time = int(np.round(428*1e-9/self.dt)) # approx 428 ns for heron
        else:
            enc_time = self.recovery_time
        rec_time = self.recovery_time

        if len(colors) == 2:
            W_rows = [2,3]
        elif len(colors) == 4:
            W_rows = [2,3,4,6]
        else:
            W_rows = np.arange(2, len(colors)+2, 1, dtype=int)

        dd_circs_enc = dd_sequence(enc_time, colors, W, W_rows, self.x_time, self.min_dd_delay)
        # dd_circs_rec = self.dd_sequence(rec_time, colors, W, [2,3,4,6])
        dd_circs_rec_and_enc = dd_sequence(rec_time + enc_time, colors, W, W_rows, self.x_time, self.min_dd_delay)

        if self.use_cif:
            flag = 1
            Rk = PI3_recovery(clbits = 1)
        else:
            flag = 0
            Rk = PI3_recovery()

        qubits = {}
        i = 0
        for qubit in data_qubits:
            qubits[qubit] = i
            i += 1
        for qubit in ancilla_qubits:
            qubits[qubit] = i
            i += 1
        for qubit in spectator_qubits:
            if qubit not in qubits:
                qubits[qubit] = i
                i += 1

        for delay in delays:
            delay = int(delay)

            dd_circs_delay = dd_sequence(delay, colors, W, W_rows, self.x_time, self.min_dd_delay)

            qc = QuantumCircuit(len(qubits), total_set*(4 + flag))

            for k in range(total_set):
                qc.append(enc, [qubits[qubit] for qubit in data_qubits[k*3: (k+1)*3]])

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits and qubit not in data_qubits:
                        qc.append(dd_circs_enc[key], [qubits[qubit]])

            qc.barrier()

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits:
                        qc.append(dd_circs_delay[key], [qubits[qubit]])

            qc.barrier()

            if flag == 1:
                for k in range(total_set):
                    qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], clbits = [k*5], inplace=True)
                    qc.append(enc.inverse(), [qubits[qubit] for qubit in data_qubits[k*3: (k+1)*3]])
                    qc.measure(
                        [qubits[qubit] for qubit in qubit_pos[k][1:]], 
                        np.arange((k*5)+1, ((k+1)*5), 1, dtype=int)
                    )
            else:
                for k in range(total_set):
                    qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], inplace=True)
                    qc.append(enc.inverse(), [qubits[qubit] for qubit in data_qubits[k*3: (k+1)*3]])
                    qc.measure(
                        [qubits[qubit] for qubit in qubit_pos[k][1:]], 
                        np.arange((k*4), ((k+1)*4), 1, dtype=int)
                    )

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits and qubit not in (data_qubits + ancilla_qubits):
                        qc.append(dd_circs_rec_and_enc[key], [qubits[qubit]])


            qc = transpile(qc, backend = self.backend, optimization_level = 3, routing_method = 'none', initial_layout = list(qubits.keys()), layout_method='trivial')
            circs.append((qc,None,shots))

        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['spectator_qubits'] = spectator_qubits
        self.storage[tag]['shots'] = shots
        
    # PI Multi-QEC experiment with CHaDD
    # The inputs:
    # delays: list[int] - list of delay times (in microseconds)
    # qubit_pos: list[list[int]] - list of set of qubits used for the protocol. Each set should have 5 qubits
    # state: str - state of the qubits (default is '1') Options: '1', '0', '+', '-'.
    # qec_cycle_relaxation: int - maximum relaxation time in between two QEC cycles. 
    # spectator_qubits: list[int] - list of qubits which will take part in the CHaDD protocol. 
    # If the qubits in the qubit_pos is a subset of spectator_qubits, no new color will be assigned to the qubits in the qubit_pos, i.e. 2 colored CHaDD will be implemented.
    # tag: str - tag for the experiment (default is 'multi_qec_<qec_cycle_relaxation>_with_chadd'). Used for retrieving the results.
    # shots: int - number of shots for each experiment (default is 2**13).
    def add_multi_qec_with_chadd_circs(
            self, 
            delays: list[int], qubit_pos: list[int], 
            colors: dict, state: str = '1', 
            qec_cycle_relaxation: int = 10, 
            spectator_qubits: list[int] = [], 
            tag: str = None, 
            shots: int = 2**13
    ):
        if tag is None:
            tag = f'multi_qec_{qec_cycle_relaxation}_with_chadd'

        total_qubits = len(colors['0']) + len(colors['1'])

        if isinstance(qubit_pos[0], int):
            qubit_pos = [qubit_pos]

        total_set = len(qubit_pos)

        data_qubits = []
        ancilla_qubits = []

        for qubit_set in qubit_pos:
            if set(qubit_set).issubset(set(spectator_qubits)):
                colors = colors
            elif qubit_set[0] in spectator_qubits and qubit_set[4] in spectator_qubits:
                colors = assign_color(qubit_set[1:-1], total_qubits, colors)
            else:
                colors = assign_color(qubit_set, total_qubits, colors)

            data_qubits += qubit_set[1:4]
            ancilla_qubits += [qubit_set[0], qubit_set[4]]

        if len(spectator_qubits) == 0:
            spectator_qubits = colors['0'] + colors['1']

        self.storage[tag] = {'delays': delays}

        qec_cycle_relaxation = int(np.round(qec_cycle_relaxation*1e-6/self.dt))

        W = W_matrix(len(colors))
        delays = np.round((delays*1e-6)/self.dt)
        circs = []  

        enc = encoder(state)
        if state == '1_simplified':
            enc_time = self.x_time
        elif state == '0_simplified':
            enc_time = int(np.round(428*1e-9/self.dt)) # approx 428 ns for heron
        else:
            enc_time = self.encoder_time
        rec_time = self.recovery_time
        if len(colors) == 2:
            W_rows = [2,3]
        elif len(colors) == 4:
            W_rows = [2,3,4,6]
        else:
            W_rows = np.arange(2, len(colors)+2, 1, dtype=int)

        ancilla_key = {}
        for key in colors:
            for qubit in colors[key]:
                if qubit in ancilla_qubits and key not in ancilla_key:
                    ancilla_key[key] = {}

        dd_circs_enc = dd_sequence(enc_time, colors, W, W_rows, self.x_time, self.min_dd_delay)
        dd_circs_rec = dd_sequence(rec_time, colors, W, W_rows, self.x_time, self.min_dd_delay)
        dd_circs_rec_and_enc = dd_sequence(rec_time + enc_time, colors, W, W_rows, self.x_time, self.min_dd_delay)
        dd_circs_cycle = dd_sequence(qec_cycle_relaxation, colors, W, W_rows, self.x_time, self.min_dd_delay)
        for key in ancilla_key:
            ancilla_key[key]['dd_circ'] = self.changed_dd_sequence(qec_cycle_relaxation, W, W_rows[int(key)])

        if self.use_cif:
            flag = 1
            Rk = PI3_recovery(clbits = 1)
        else:
            flag = 0
            Rk = PI3_recovery()
        
        qubits = {}
        i = 0
        for qubit in data_qubits:
            qubits[qubit] = i
            i += 1
        for qubit in ancilla_qubits:
            qubits[qubit] = i
            i += 1
        for qubit in spectator_qubits:
            if qubit not in qubits:
                qubits[qubit] = i
                i += 1

        for delay in delays:
            
            delay = int(delay)

            total_qec = int(np.ceil(delay/qec_cycle_relaxation))
            if total_qec == 0:
                total_qec = 1
            if (delay - (total_qec - 1)*qec_cycle_relaxation) < rec_time and total_qec > 1:
                total_qec -= 1

            cbits = 3+flag+total_qec

            qc = QuantumCircuit(len(qubits), total_set*cbits)

            for k in range(total_set):
                qc.append(enc, [qubits[qubit] for qubit in qubit_pos[k][1:-1]])

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits and qubit not in data_qubits:
                        qc.append(dd_circs_enc[key], [qubits[qubit]])
            
            qc.barrier()

            for i in range(total_qec-1):

                for key in colors:
                    for qubit in colors[key]:
                        if qubit in qubits:
                            if qubit in ancilla_qubits and i>0:
                                qc.append(ancilla_key[key]['dd_circ'], [qubits[qubit]])
                            else:
                                qc.append(dd_circs_cycle[key], [qubits[qubit]])
                        
                qc.barrier()

                if flag == 1:
                    for k in range(total_set):
                        qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], clbits = [k*5], inplace = True)
                        qc.measure(
                            qubits[qubit_pos[k][-1]], 
                            k*cbits + 4 + i
                        )
                else:
                    for k in range(total_set):
                        qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], inplace=True)
                        qc.measure(
                            qubits[qubit_pos[k][-1]], 
                            k*cbits + 3 + i
                        )

                for key in colors:
                    for qubit in colors[key]:
                        if qubit in qubits and qubit not in (data_qubits + ancilla_qubits):
                            qc.append(dd_circs_rec[key], [qubits[qubit]])

                qc.barrier()
                qc.reset([qubits[qubit] for qubit in ancilla_qubits])

            remaining_time = delay - qec_cycle_relaxation*(total_qec-1)

            if total_qec > 1:
                for key in ancilla_key:
                    ancilla_key[key]['remaining_dd'] = self.changed_dd_sequence(remaining_time, W, W_rows[int(key)])
            remaining_dd = dd_sequence(remaining_time, colors, W, W_rows, self.x_time, self.min_dd_delay)

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits:
                        if qubit in ancilla_qubits and total_qec > 1:
                            qc.append(ancilla_key[key]['remaining_dd'], [qubits[qubit]])
                        else:
                            qc.append(remaining_dd[key], [qubits[qubit]])

            qc.barrier()

            if flag == 1:
                for k in range(total_set):
                    qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], clbits = [k*5], inplace = True)
                    qc.measure(
                        qubits[qubit_pos[k][-1]], 
                        k*cbits + 3 + total_qec
                    )
            else:
                for k in range(total_set):
                    qc.compose(Rk, qubits=[qubits[qubit] for qubit in qubit_pos[k]], inplace=True)
                    qc.measure(
                        qubits[qubit_pos[k][-1]], 
                        k*cbits + 2 + total_qec
                    )

            for k in range(total_set):
                qc.append(enc.inverse(), [qubits[qubit] for qubit in qubit_pos[k][1:-1]])
                qc.measure(
                    [qubits[qubit] for qubit in qubit_pos[k][1:4]], 
                    np.arange((k*cbits), (k*cbits) + flag + 3, 1, dtype=int)
                )

            for key in colors:
                for qubit in colors[key]:
                    if qubit in qubits and qubit not in (data_qubits + ancilla_qubits):
                        qc.append(dd_circs_rec_and_enc[key], [qubits[qubit]])

            
            qc = transpile(qc, backend = self.backend, optimization_level = 3, routing_method = 'none', initial_layout = list(qubits.keys()), layout_method='trivial')
            circs.append((qc, None, shots))

        self.storage[tag]['circs'] = circs
        self.storage[tag]['qubit_pos'] = qubit_pos
        self.storage[tag]['spectator_qubits'] = spectator_qubits
        self.storage[tag]['shots'] = shots

    # Runs the circuits and stores the results in the storage dictionary.
    def run(self):

        circs = []  
        for key in self.storage:
            circs += self.storage[key]['circs']

        if self.use_mitigation:

            physical_qubits = []
            for key in self.storage:
                if key != 'Mitigator':
                    for qubit in self.storage[key]['qubit_pos']:
                        if qubit not in physical_qubits:
                            physical_qubits.append(qubit)

            with Batch(backend=self.backend) as batch:
                
                exp = LocalReadoutError(physical_qubits, self.backend)
                mitigation_job = exp.run(shots = 2**11)

                sampler = Sampler(mode=batch)
                # sampler.options.update(twirling={"enable_gates": True})
                
                job = sampler.run(circs)
                
            self.mitigator = mitigation_job.analysis_results('Local Readout Mitigator', dataframe=True).iloc[0].value
            assignment_matrix = []
            for qubit in self.mitigator.qubits:
                assignment_matrix.append(self.mitigator.assignment_matrix(qubit).tolist())

            self.mitigator_data = {
                'qubits': self.mitigator.qubits,
                'assignment_matrix': assignment_matrix
            }

        else:

            sampler = Sampler(mode = self.backend)
            # sampler.options.update(twirling={"enable_gates": True})

            job = sampler.run(circs)

        print("Job ID:", job.job_id())
        raw_data = job.result()
        i = 0

        for key in self.storage:
            data = [ res.data.c.get_bitstrings() for res in raw_data[i:(i+len(self.storage[key]['delays']))] ]
            self.storage[key]['result'] = data
            i += len(self.storage[key]['delays'])

        self._post_process()
        
    def _post_process(self):

        if self.use_cif:
            flag = 1
        else:
            flag = 0

        def _result_dict(data):
            result = {}
            for state in data:
                if state not in result:
                    result[state] = 1
                else:
                    result[state] += 1
            return result
        
        for key in self.storage:

            shots = self.storage[key]['shots']
            batch_shots = int(np.round((shots/8)))

            if key == 'bare_qubit' or key == 'T2_Hahn' or key == 'T2_Ramsey':
                fids = {}
                fids_stds = {}
                for qubit in self.storage[key]['qubit_pos']:
                    fids[qubit] = []
                    fids_stds[qubit] = []

                l = len(self.storage[key]['qubit_pos'])

                for data in self.storage[key]['result']:
                    data_sets = [[] for _ in range(l) ]

                    for i in range(8):

                        batch = data[i*(batch_shots): (i+1)*(batch_shots)]

                        probs = _result_dict(batch)

                        if self.use_mitigation:
                            probs = self.mitigator.quasi_probabilities(Counts(probs), qubits=self.storage[key]['qubit_pos'], clbits=list(range(l))).binary_probabilities()

                        success = [0 for _ in range(l) ]
                        for state in probs:
                            for position in range(l):
                                if state[-1-position] == '0':
                                    success[position] += probs[state]

                        for position in range(l):
                            if self.use_mitigation:
                                data_sets[position].append(success[position])
                            else:
                                data_sets[position].append(success[position]/batch_shots)
                    
                    for j in range(l):
                        qubit = self.storage[key]['qubit_pos'][j]
                        fids[qubit].append(np.mean(data_sets[j]))
                        fids_stds[qubit].append(np.std(data_sets[j]))

                self.storage[key]['fids'] = fids
                self.storage[key]['fids_stds'] = fids_stds

            else:

                fids = {}
                fids_stds = {}
                success_rates = {}

                total_set = len(self.storage[key]['qubit_pos'])
                for k in range(total_set):
                    fids[k] = []
                    fids_stds[k] = []
                    success_rates[k] = []

                for data in self.storage[key]['result']:

                    data_sets = []
                    batch_success = []
                    for k in range(total_set):
                        data_sets.append([])
                        batch_success.append([])

                    cbits = int(len(data[0])/total_set)
                    total_qec = cbits - 3 - flag

                    for i in range(8):
                        success = []
                        flag_success = []
                        for k in range(total_set):
                            success.append(0)
                            flag_success.append(0)

                        batch = data[i*(batch_shots): (i+1)*(batch_shots)]

                        probs = _result_dict(batch)

                        if self.use_mitigation:
                            probs = self.mitigator.quasi_probabilities(
                                Counts(probs),
                                qubits=self.storage[key]['qubit_pos'][1 - flag:-1] + [self.storage[key]['qubit_pos'][-1]]*total_qec,
                                clbits=list(range(3+flag+total_qec))
                            ).binary_probabilities()

                        for big_state in probs:
                            for k in range(total_set):
                                state = big_state[k*cbits: (k+1)*cbits]
                                if state[0: total_qec] == '0'*total_qec:
                                    flag_success[k] += probs[big_state]
                                    if self.use_cif and state[total_qec: -1] == '000':
                                        success[k] += probs[big_state]
                                    elif not self.use_cif and state[total_qec:] == '000':
                                        success[k] += probs[big_state]
                        
                        for k in range(total_set):
                            if flag_success[k] > 0:
                                data_sets[k].append(success[k]/flag_success[k])
                            else:
                                data_sets[k].append(0.0)
                            
                            if self.use_mitigation:
                                batch_success[k].append(flag_success[k])
                            else:
                                batch_success[k].append(flag_success[k]/batch_shots)

                    for k in range(total_set):
                        fids[k].append(np.mean(data_sets[k]))
                        fids_stds[k].append(np.std(data_sets[k]))
                        success_rates[k].append(np.mean(batch_success[k]))
                
                self.storage[key]['fids'] = fids
                self.storage[key]['fids_stds'] = fids_stds
                self.storage[key]['success_rates'] = success_rates

    # Tag: The tag of the experiment whose result will be shown.
    def show(self, tag: str, include_offset: bool = False, qec_cycle_relaxation: int = None, save_fig: bool = False, fig_name: str = None):

        if tag not in self.storage:
            raise ValueError(f'Invalid tag: {tag}')
        
        if self.storage[tag]['result'] is None:
            raise Exception('Run the circuits first')

        delays_us = np.array(self.storage[tag]['delays'])

        if tag == 'bare_qubit' or tag == 'T2_Hahn' or tag == 'T2_Ramsey':
            if len(self.storage[tag]['qubit_pos']) == 1:
                qubit = int(self.storage[tag]['qubit_pos'][0])
            else:
                qubit = int(input(f"Please choose from {self.storage[tag]['qubit_pos']}. Qubit to show: "))
            if qubit not in self.storage[tag]['qubit_pos']:
                raise ValueError(f'Invalid qubit: {qubit}')
            try:
                fids = np.array(self.storage[tag]['fids'][qubit])
                fids_stds = np.array(self.storage[tag]['fids_stds'][qubit])
            except:
                fids = np.array(self.storage[tag]['fids'][str(qubit)])
                fids_stds = np.array(self.storage[tag]['fids_stds'][str(qubit)])
            tag += f" {qubit}"
        
        else:
            total_set = len(self.storage[tag]['qubit_pos'])
            if total_set == 1:
                qubit_set = 0
            else:
                qubit_set = int(input(f"Please choose qubit sets from {np.arange(total_set)}. Qubit set to show: "))

            if qubit_set > total_set or qubit_set < 0:
                raise ValueError(f'Invalid qubit set: {qubit_set}')
            
            try:
                fids = np.array(self.storage[tag]['fids'][qubit_set])
                fids_stds = np.array(self.storage[tag]['fids_stds'][qubit_set])
            except:
                fids = np.array(self.storage[tag]['fids'][str(qubit_set)])
                fids_stds = np.array(self.storage[tag]['fids_stds'][str(qubit_set)])

            if include_offset:
                enc_time = (self.encoder_time*self.dt)/1e3
                rec_time = (self.recovery_time*self.dt)/1e3
                if qec_cycle_relaxation is not None:
                    for i in range(len(delays_us)):
                        t = int(np.ceil(delays_us[i]/qec_cycle_relaxation))
                        if t  == 0:
                            t = 1
                        delays_us[i] += t*rec_time
                else:
                    delays_us += rec_time
                delays_us += 2*enc_time
                print(delays_us)


        def exp_decay(x, tau, B, C, D):
            return np.exp(-x / tau)*np.sin(C*x + D) + B

        popt, pcov = curve_fit(exp_decay, delays_us, fids, p0=[200, 0.0, 0.0, 0.0], bounds=([0.0, -1.0, -np.inf, -np.pi], [np.inf, 0.0, np.inf, np.pi]))
        tau_fit, B_fit, C_fit, D_fit = popt

        param_errors = np.sqrt(np.diag(pcov))
        tau_error = param_errors[1]

        y_fit = exp_decay(delays_us, *popt)
        residuals = fids - y_fit
        chi_squared = np.sum((residuals / fids_stds)**2)
        reduced_chi_squared = chi_squared / (len(delays_us) - len(popt))

        x_smooth = np.linspace(delays_us[0], delays_us[-1], 500)
        y_smooth = exp_decay(x_smooth, *popt)
        y_smooth_negative = exp_decay(x_smooth, tau_fit-tau_error, B_fit, C_fit, D_fit)
        y_smooth_positive = exp_decay(x_smooth, tau_fit+tau_error, B_fit, C_fit, D_fit)

        plt.figure(figsize=(16, 10))

        plt.plot(x_smooth, y_smooth, '-', color='royalblue', linewidth=2, label=f'Fit')

        plt.errorbar(delays_us, fids, yerr=fids_stds, 
                    fmt='o', color='black', ecolor = 'dimgrey', markersize=6, 
                    capsize=4, capthick=1, elinewidth=3, 
                    label='Data', ls = '--')

        plt.grid(True, alpha=0.75)


        plt.xlabel('Delay [μs]', fontsize=18)
        plt.ylabel(r'${Fidelity}^2$', fontsize=18)
        plt.xlim(delays_us[0], delays_us[-1])
        plt.ylim(0.0, 1.0)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.fill_between(x_smooth, y_smooth_negative, y_smooth_positive, alpha=0.2, color='steelblue')

        plt.tick_params(axis='both', direction='in')

        fit_text = f'T1 for {tag} = {tau_fit:.2f} ± {tau_error:.2f} μs\n' + r'Reduced-$\chi^2$' + f' = {reduced_chi_squared:.2f}'

        plt.text(0.368, 0.9, fit_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                fontsize=18, verticalalignment='bottom')

        plt.legend(fontsize=18, frameon = False)
        plt.tight_layout()

        if save_fig:
            plt.savefig(f'{tag}_{fig_name}.png', dpi=300)

    def save(self, path: str):

        if os.path.exists(path):
            resp = input(f"{path} exists. Overwrite? (y/n): ")
            if resp.lower() != 'y':
                print("Aborted.")
                exit()

        store = {}
        for key in self.storage:
            store[key] = {}
            if type(self.storage[key]['delays']) is np.ndarray:
                store[key]['delays'] = self.storage[key]['delays'].tolist()
            else:
                store[key]['delays'] = self.storage[key]['delays']

            store[key]['result'] = self.storage[key]['result']
            store[key]['fids'] = self.storage[key]['fids']
            store[key]['fids_stds'] = self.storage[key]['fids_stds']
            store[key]['qubit_pos'] = self.storage[key]['qubit_pos']
            store[key]['shots'] = self.storage[key]['shots']
            if 'spectator_qubits' in self.storage[key]:
                store[key]['spectator_qubits'] = self.storage[key]['spectator_qubits']
            if 'success_rates' in self.storage[key]:
                store[key]['success_rates'] = self.storage[key]['success_rates']

        store['Mitigator'] = self.mitigator_data

        with open(path, 'w') as f:
            json.dump(store, f, indent=2)
        
    def load(self, path: str):

        with open(path, 'r') as f:
            store = json.load(f)

            for key in store:
                if key == 'Mitigator' and len(store[key]) != 0:
                    self.mitigator = LocalReadoutMitigator(qubits = store[key]['qubits'], assignment_matrices = store[key]['assignment_matrix'])
                else:
                    self.storage[key] = store[key]

    def retrieve_job(self, service, mitigator_job_id: str = '', job_id: str = ''):

        if len(mitigator_job_id) != 0:
            
            job = service.job(mitigator_job_id)

            physical_qubits = []
            for key in self.storage:
                if key != 'Mitigator':
                    for qubit in self.storage[key]['qubit_pos']:
                        if qubit not in physical_qubits:
                            physical_qubits.append(qubit)
            

            exp = LocalReadoutError(physical_qubits, self.backend)    
            exp._finalize()
            expdata = exp._initialize_experiment_data()
            expdata.add_jobs(job)
            exp.analysis.run(expdata)
            self.mitigator = exp.analysis.run(expdata).analysis_results('Local Readout Mitigator', dataframe=True).iloc[0].value
            assignment_matrix = []
            for qubit in self.mitigator.qubits:
                assignment_matrix.append(self.mitigator.assignment_matrix(qubit).tolist())

            self.mitigator_data = {
                'qubits': self.mitigator.qubits,
                'assignment_matrix': assignment_matrix
            }

        if len(job_id) != 0:

            job = service.job(job_id)
    
            raw_data = job.result()
            i = 0

            for key in self.storage:
                data = [ res.data.c.get_bitstrings() for res in raw_data[i:(i+len(self.storage[key]['delays']))] ]
                self.storage[key]['result'] = data
                i += len(self.storage[key]['delays'])

            self._post_process()
