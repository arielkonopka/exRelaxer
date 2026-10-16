## Neuron models on the nonlinearity tasks

Medians over 5 seeds; lr in {0.001, 0.003, 0.01, 0.03} chosen per architecture on validation; feedback alignment with a learned bias, habituation off, default model parameters (LIF/ALIF leak 0.9, threshold 1; Izhikevich regular spiking, gain 10; resonator f 0.1, damping 0.9).

### nl_temporal (test accuracy; t4: test MSE)

Cell: best test result (smallest solving network, neurons; success = accuracy ≥ 0.95, MSE ≤ 0.001), active fraction of the best architecture. `~0.5`: rate units with smoothing 0.5.

| model | t1 | t2 | t3 n4 | t3 n8 | t4 |
|---|---|---|---|---|---|
| relu | 0.51, act 0.50 | 0.748, act 0.50 | 0.489, act 0.50 | 0.518, act 0.40 | 0.0916, act 0.23 |
| er | 0.636, act 0.78 | 0.779, act 0.79 | 0.523, act 0.79 | 0.518, act 0.92 | 0.0925, act 0.64 |
| er_memoryless | 0.51, act 1.00 | 0.748, act 1.00 | 0.495, act 1.00 | 0.518, act 1.00 | 0.0919, act 0.78 |
| clamp | 0.51, act 1.00 | 0.748, act 1.00 | 0.495, act 1.00 | 0.518, act 1.00 | 0.0918, act 1.00 |
| sigmoid | 0.496, act 1.00 | 0.748, act 1.00 | 0.489, act 1.00 | 0.518, act 1.00 | 0.0915, act 1.00 |
| sigmoid~0.5 | 0.496, act 1.00 | 0.758, act 1.00 | – | – | 0.0915, act 1.00 |
| tanh | 0.51, act 1.00 | 0.748, act 1.00 | 0.495, act 1.00 | 0.518, act 1.00 | 0.0918, act 1.00 |
| tanh~0.5 | 1 (8), act 1.00 | 0.758, act 1.00 | – | – | 0.0918, act 1.00 |
| softplus | 0.496, act 1.00 | 0.748, act 1.00 | 0.489, act 1.00 | 0.518, act 1.00 | 0.0917, act 1.00 |
| softplus~0.5 | 1 (16), act 1.00 | 0.758, act 1.00 | – | – | 0.0917, act 1.00 |
| lif | 0.949, act 0.13 | 0.86, act 0.17 | 0.676, act 0.14 | 0.522, act 0.13 | 0.0849, act 0.08 |
| alif | 0.908, act 0.10 | 0.818, act 0.12 | 0.693, act 0.08 | 0.544, act 0.08 | 0.0879, act 0.06 |
| izhikevich | 0.496, act 0.02 | 0.763, act 0.02 | 0.492, act 0.02 | 0.526, act 0.02 | 0.0916, act 0.00 |
| resonator | 0.677, act 0.30 | 0.834, act 0.35 | 0.774, act 0.24 | 0.518, act 0.18 | 0.087, act 0.17 |

### nl_static (test MSE)

Cell: best test MSE (smallest network with MSE ≤ 1e-3 in ≥ 80 % of seeds, neurons), active fraction.

| model | l0 | l1 | l2 | l3 | l4 k1 | l4 k16 | l4 k4 |
|---|---|---|---|---|---|---|---|
| relu | 0.000349 (4), act 0.49 | 0.000613 (16), act 0.46 | 0.000595 (16), act 0.46 | 0.000723 (32), act 0.51 | 0.000801 (24), act 0.56 | 0.000765 (32), act 0.48 | 0.000782 (32), act 0.48 |
| er | 0.00162, act 0.62 | 0.0657, act 0.65 | 0.0515, act 0.65 | 0.0807, act 0.64 | 0.167, act 0.65 | 0.0871, act 0.65 | 0.11, act 0.65 |
| clamp | 3.39e-15 (4), act 1.00 | 0.111, act 1.00 | 0.0986, act 1.00 | 0.136, act 1.00 | 0.333, act 1.00 | 0.125, act 1.00 | 0.196, act 1.00 |
| sigmoid | 0.00041 (4), act 1.00 | 0.000649 (32), act 1.00 | 0.000696 (32), act 1.00 | 0.0128, act 1.00 | 0.034, act 1.00 | 0.0402, act 1.00 | 0.035, act 1.00 |
| tanh | 0.000824 (32), act 1.00 | 0.000526 (16), act 1.00 | 0.000596 (16), act 1.00 | 0.000637 (16), act 1.00 | 0.000697 (16), act 1.00 | 0.00106, act 1.00 | 0.000634 (16), act 1.00 |
| softplus | 0.000254 (4), act 1.00 | 0.000244 (8), act 1.00 | 0.000244 (8), act 1.00 | 0.00101, act 1.00 | 0.0135, act 1.00 | 0.0371, act 1.00 | 0.0203, act 1.00 |
| lif | 0.0327, act 0.34 | 0.0354, act 0.07 | 0.03, act 0.09 | 0.0519, act 0.09 | 0.113, act 0.11 | 0.0866, act 0.19 | 0.0913, act 0.14 |
| alif | 0.116, act 0.09 | 0.043, act 0.07 | 0.0361, act 0.07 | 0.0694, act 0.08 | 0.162, act 0.08 | 0.139, act 0.12 | 0.132, act 0.08 |
| izhikevich | 0.612, act 0.02 | 0.11, act 0.01 | 0.0977, act 0.01 | 0.134, act 0.01 | 0.382, act 0.02 | 0.542, act 0.02 | 0.351, act 0.01 |
| resonator | 0.0273, act 0.41 | 0.0282, act 0.11 | 0.0228, act 0.13 | 0.0445, act 0.17 | 0.108, act 0.13 | 0.0743, act 0.17 | 0.0777, act 0.15 |
