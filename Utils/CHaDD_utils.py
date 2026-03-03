import numpy as np
from qiskit import QuantumCircuit

# Heavy Hex Coloring for IBM machines. Returns a dictionary of lists of qubits for color '0' and '1'.
# The inputs:
# n: int - number of qubits
# row_n: int - number of qubits in each row
# shorten: int - number of qubits to shorten for the first or last row (depends how the qubits are numbered in the hardware)
# even_qubit_bridge: bool - whether to add a bridge between even or odd qubit numbers
# uno_reverse: bool - whether to reverse the starting color for the even rows (depends on the hardware)
def heavyhex_color(n: int, row_n: int, shorten: int, even_qubit_bridge: bool, reverse_color: bool = False):
    if n <= 0:
        return {'0': [], '1': []}

    color_0 = []
    color_1 = []

    i = 0
    while i < n-1:
        if i == 0 or i + row_n > n:
            row_current = row_n - shorten
        else:
            row_current = row_n
        
        if i + row_n > n and reverse_color:    
            color_0.extend(range(i + 1, i + row_current, 2))
            color_1.extend(range(i, i + row_current, 2))
        else:
            color_0.extend(range(i, i + row_current, 2))
            color_1.extend(range(i + 1, i + row_current, 2))

        i += row_current

        if i < n:
            bridge_indices = list(range(i, i + 4))

            if even_qubit_bridge:
                color_1.extend(bridge_indices)
            else:
                color_0.extend(bridge_indices)

            i += 4

    return {'0': color_0, '1': color_1}

# Returns the W matrix for the given chi/chromaticity/number of colors 
def W_matrix(chi: int):
    n = int(1 + np.floor(np.log2(chi)))
    N = 2**n
    W = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            p = bin(i & j).count('1') % 2
            W[i][j] = (-1)**p

    return W

# Returns a dictionary of QuantumCircuits for the given delay and colors. The dictionary keys are the colors of the qubits.
# The inputs:
# delay: int - delay time (in dt of the system)
# colors: dict - dictionary of lists of qubits for color '0' and '1'
# W: np.ndarray - W matrix
# W_rows: list - list of qubits for each color
# x_time: int - time for a single X gate (in dt of the system)
# min_dd_delay: int - minimum delay time rquired to implement the CHaDD protocol (in dt of the system)
# robust: bool - implements the robust CHaDD protocol (see https://arxiv.org/abs/2406.13901)
def dd_sequence(delay: int, colors: dict, W: np.ndarray, W_rows: list, x_time: int, min_dd_delay: int = None, robust: bool = False):

    if min_dd_delay is None:
        min_dd_delay = x_time
    
    n = W.shape[0]

    if robust:
        n*=2
        W = W.tolist()
        for i in range(len(W)):
            W[i] = W[i] + W[i]

    min_time = n*(x_time + min_dd_delay)
    qcs = {}
    for key in colors:
        qcs[key] = QuantumCircuit(1)

    if delay < min_time:
        return qcs
    
    elif robust:
        dd_delay = int(np.round(delay/n))
        for key in colors:
            qcs[key].delay(dd_delay, 0)
            sequence = W[W_rows[int(key)]]
            sign = 0
            for i in range(1, n):
                if sequence[i] == 1:
                    if sequence[i-1] == -1:
                        if sign == 0 or sign == 3:
                            qcs[key].rx(np.pi, 0)
                        else:
                            qcs[key].rx(-np.pi, 0)
                        sign = (sign+1)%4
                    else:
                        qcs[key].delay(x_time, 0)
                else:
                    if sequence[i-1] == 1:
                        if sign == 0 or sign == 3:
                            qcs[key].rx(np.pi, 0)
                        else:
                            qcs[key].rx(-np.pi, 0)
                        sign = (sign+1)%4
                    else:
                        qcs[key].delay(x_time, 0)
                qcs[key].delay(dd_delay, 0)
            if sequence[n-1] == -1:
                qcs[key].rx(np.pi, 0)
        
    else:
        dd_delay = int(np.round(delay/n))
        for key in colors:
            qcs[key].delay(dd_delay, 0)
            sequence = W[W_rows[int(key)]]
            for i in range(1, n):
                if sequence[i] == 1:
                    if sequence[i-1] == -1:
                        qcs[key].rx(np.pi, 0)
                    else:
                        qcs[key].delay(x_time, 0)
                else:
                    if sequence[i-1] == 1:
                        qcs[key].rx(np.pi, 0)
                    else:
                        qcs[key].delay(x_time, 0)
                qcs[key].delay(dd_delay, 0)
            if sequence[n-1] == -1:
                qcs[key].rx(np.pi, 0)
        
    return qcs

# Assigns different color codes for the qubits that are going to be used in the quantum process. Initially all the qubits are assigned to color '0' or '1'.
def assign_color(qubit_positions: list[int], total_qubits: int, color: dict):
    if len(qubit_positions) == total_qubits:
        return color
    color_2 = []
    color_3 = []

    if '2' in color:
        color_2 = color['2']
    if '3' in color:
        color_3 = color['3']

    for qubit in qubit_positions:
        if qubit in color['0']:
            color_2.append(qubit)
            color['0'].remove(qubit)
        elif qubit in color['1']:
            color_3.append(qubit)
            color['1'].remove(qubit)
    
    if len(color_2) == 0:
        return {'0': color['0'], '1': color['1'], '2': color_3}
    elif len(color_3) == 0:
        return {'0': color['0'], '1': color['1'], '2': color_2}
    else:
        return {'0': color['0'], '1': color['1'], '2': color_2, '3': color_3}