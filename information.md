# Fitting Function:

$f(t, T_1, B) = \exp{(-t/T_1)} + B$

B is the parameter added to account for the measurement errors.

# Qubit Information

## Individual Qubit $T_1$ values:

| Qubit | T1 (in $\mu s$)|
|---|---|
| 57 | 326.3 ± 9.5 |
| 58 | 179.2 ± 4.0 |
| 59 | 257.3 ± 9.1 |
| 60 | 264.3 ± 4.9 |
| 61 | 338.2 ± 17.7 |
| 32 | 57.3 ± 4.0 |
| 33 | 269.4 ± 16.9 |
| 37 | 169.7 ± 4.2 |
| 52 | 256.9 ± 12.7 |
| 51 | 249.8 ± 4.5 |
| 102 | 297.3 ± 6.7 |
| 101 | 272.7 ± 15.7 |
| 111 | 260.8 ± 6.8 |
| 120 | 235.8 ± 9.9 |
| 121 | 330.1 ± 7.7 |

## Average Noise Parameters:

| Parameter | Value (in $\mu s$)|
|---|---|
| $T_1$ | 224 ± 80 |
| $T_2$ | 85 ± 11 |


## Gate Noise Parameters:

| Gate | Time (in $ns$) | Error |
|---|---|---|
| X | 32 | $(2.5 ± 0.2) \times 10^{-4}$ |
| SX | 32 | $(2.5 ± 0.2) \times 10^{-4}$ |
| $R_Z$ | 0 | $0$ |
| CZ | 68 | $(2.0 ± 0.4) \times 10^{-3}$ |
| Measurement | 1560 | $(7.0 ± 1.0) \times 10^{-3}$ |
| Reset | 2720 | $0$ |

# Optimization Results:

| Implemented Gate | Frobenius Norm Distance (Cost-function) | Channel Fidelity |
|---|---|---|
| Encoder | $ 7 \times 10^{-22}$ | 1.0 |
| U | $ 7 \times 10^{-17}$ | 1.0 |
| V | $ 5 \times 10^{-17}$ | 1.0 |
| $\text{D}_\text{approx}$ | $ 1 \times 10^{-16}$ | 1.0 |