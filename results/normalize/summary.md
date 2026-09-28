## Normalised weighted sum (hidden layers), habituation off

Medians over trials; "raw" is the same run without normalisation (results/rerun, variant off).

### er_economy

| model | metric | raw | normalised |
|---|---|---|---|
| er | accuracy | 1 | 0.867 |
| er | active_fraction | 0.47 | 0.463 |
| er | spikes_per_inference | 126 | 124 |
| er | accuracy_per_100_spikes | 0.795 | 0.698 |
| gate | accuracy | 0.823 | 0.817 |
| gate | active_fraction | 0.465 | 0.49 |
| gate | spikes_per_inference | 124 | 131 |
| gate | accuracy_per_100_spikes | 0.661 | 0.622 |
| linear | accuracy | 0.997 | 1 |
| linear | active_fraction | 1 | 1 |
| linear | spikes_per_inference | 268 | 268 |
| linear | accuracy_per_100_spikes | 0.372 | 0.373 |

### er_paths

| model | metric | raw | normalised |
|---|---|---|---|
| er | accuracy | 0.995 | 0.85 |
| er | active_fraction | 0.469 | 0.465 |
| er | switch_rate | 0.257 | 0.275 |
| er | internal_switch_rate | 0.0799 | 0.098 |
| gate | accuracy | 0.831 | 0.831 |
| gate | active_fraction | 0.469 | 0.493 |
| gate | switch_rate | 0.146 | 0.158 |
| gate | internal_switch_rate | 0 | 0 |
| linear | accuracy | 0.994 | 0.999 |
| linear | active_fraction | 1 | 1 |
| linear | switch_rate | 0.158 | 0.152 |
| linear | internal_switch_rate | 0 | 0 |

### er_fatigue

| model | metric | raw | normalised |
|---|---|---|---|
| er | rest_first_accuracy | 1 | 1 |
| er | A_first_accuracy | 1 | 0.833 |
| er | A_first_share | 0.136 | 0 |
| er | A_recovery_ticks | 12 | 16 |
| er | all_first_accuracy | 1 | 0.333 |
| gate | rest_first_accuracy | 0.833 | 1 |
| gate | A_first_accuracy | 0.833 | 1 |
| gate | A_first_share | 1 | 1 |
| gate | A_recovery_ticks | 0 | 0 |
| gate | all_first_accuracy | 0.833 | 1 |
| linear | rest_first_accuracy | 1 | 1 |
| linear | A_first_accuracy | 1 | 1 |
| linear | A_first_share | 1 | 1 |
| linear | A_recovery_ticks | 0 | 0 |
| linear | all_first_accuracy | 1 | 1 |

### er_history

| model | metric | raw | normalised |
|---|---|---|---|
| er | rest_accuracy | 1 | 1 |
| er | busy_pattern_change | 0.458 | 0.332 |
| er | all_pattern_change | 0.859 | 0.77 |
| er | all_decision_change | 0.15 | 0.7 |
| er | all_accuracy | 0.85 | 0.3 |
| gate | rest_accuracy | 0.8 | 0.85 |
| gate | busy_pattern_change | 0 | 0 |
| gate | all_pattern_change | 0 | 0 |
| gate | all_decision_change | 0 | 0 |
| gate | all_accuracy | 0.8 | 0.85 |
| linear | rest_accuracy | 1 | 1 |
| linear | busy_pattern_change | 0 | 0 |
| linear | all_pattern_change | 0 | 0 |
| linear | all_decision_change | 0 | 0 |
| linear | all_accuracy | 1 | 1 |

### er_silence

| model | metric | raw | normalised |
|---|---|---|---|
| er | rest_accuracy | 1 | 1 |
| er | after_accuracy | 1 | 1 |
| er | silence_active_w4 | 0.005 | 0.005 |
| er | self_active_ticks | 0.0966 | 0.099 |
| er | end_threshold_max | 0.017 | 0.00506 |
| gate | rest_accuracy | 0.8 | 0.8 |
| gate | after_accuracy | 0.8 | 0.8 |
| gate | silence_active_w4 | 0 | 0 |
| gate | self_active_ticks | 0 | 0 |
| gate | end_threshold_max | 0.2 | 0.2 |
| linear | rest_accuracy | 1 | 1 |
| linear | after_accuracy | 1 | 1 |
| linear | silence_active_w4 | 0 | 0 |
| linear | self_active_ticks | 0 | 0 |
| linear | end_threshold_max | 0.2 | 0.2 |

### er_habituation

| model | metric | raw | normalised |
|---|---|---|---|
| er | h4_accuracy_sum | 1 | 0.842 |
| er | h500_spikes_per_sample | 30107 | 24968 |
| er | h500_accuracy_sum | 1 | 1 |
| er | h500_accuracy_end | 1 | 1 |
| er | h500_accuracy_per_100_spikes | 0.00332 | 0.004 |
| gate | h4_accuracy_sum | 0.817 | 0.867 |
| gate | h500_spikes_per_sample | 15442 | 16250 |
| gate | h500_accuracy_sum | 0.817 | 0.867 |
| gate | h500_accuracy_end | 0.817 | 0.867 |
| gate | h500_accuracy_per_100_spikes | 0.00534 | 0.00534 |
| linear | h4_accuracy_sum | 1 | 1 |
| linear | h500_spikes_per_sample | 33500 | 33500 |
| linear | h500_accuracy_sum | 1 | 1 |
| linear | h500_accuracy_end | 1 | 1 |
| linear | h500_accuracy_per_100_spikes | 0.00299 | 0.00299 |

### nl_static

Best median test result over depth {1, 2, 3} × width {4 … 64} (accuracy for binary tasks, MSE otherwise); in brackets the smallest network solving it in ≥ 80 % of seeds (neurons).

| task | model | off | normalised |
|---|---|---|---|
| l0 | relu | 0.000487 (4) | 0.000349 (4) |
| l0 | er | 0.0464 | 0.0464 |
| l0 | gate | 0.000769 (32) | 0.0009 (32) |
| l0 | clamp | 3.47e-15 (4) | 3.39e-15 (4) |
| l1 | relu | 0.000656 (16) | 0.000613 (16) |
| l1 | er | 0.0342 | 0.0517 |
| l1 | gate | 0.091 | 0.101 |
| l1 | clamp | 0.111 | 0.111 |
| l2 | relu | 0.000667 (16) | 0.000595 (16) |
| l2 | er | 0.0297 | 0.0455 |
| l2 | gate | 0.0783 | 0.0878 |
| l2 | clamp | 0.0986 | 0.0988 |
| l3 | relu | 0.000855 (32) | 0.000723 (32) |
| l3 | er | 0.0604 | 0.0691 |
| l3 | gate | 0.119 | 0.128 |
| l3 | clamp | 0.137 | 0.136 |
| l4 k1 | relu | 0.000829 (24) | 0.000801 (24) |
| l4 k1 | er | 0.149 | 0.143 |
| l4 k1 | gate | 0.251 | 0.308 |
| l4 k1 | clamp | 0.333 | 0.333 |
| l4 k16 | relu | 0.000885 (32) | 0.000765 (32) |
| l4 k16 | er | 0.0996 | 0.0913 |
| l4 k16 | gate | 0.102 | 0.115 |
| l4 k16 | clamp | 0.127 | 0.126 |
| l4 k4 | relu | 0.00072 (32) | 0.000782 (32) |
| l4 k4 | er | 0.1 | 0.163 |
| l4 k4 | gate | 0.158 | 0.179 |
| l4 k4 | clamp | 0.196 | 0.196 |

### nl_temporal

Best median test result over depth {1, 2, 3} × width {4 … 64} (accuracy for binary tasks, MSE otherwise); in brackets the smallest network solving it in ≥ 80 % of seeds (neurons).

| task | model | off | normalised |
|---|---|---|---|
| t1 | relu | 0.51 | 0.51 |
| t1 | er | 1 (8) | 0.999 (8) |
| t1 | er_memoryless | 0.51 | 0.51 |
| t1 | gate | 0.51 | 0.51 |
| t1 | clamp | 0.51 | 0.51 |
| t2 | relu | 0.748 | 0.748 |
| t2 | er | 0.755 | 0.797 |
| t2 | er_memoryless | 0.743 | 0.748 |
| t2 | gate | 0.743 | 0.748 |
| t2 | clamp | 0.743 | 0.748 |
| t3 n4 | relu | 0.489 | 0.489 |
| t3 n4 | er | 0.586 | 0.581 |
| t3 n4 | er_memoryless | 0.504 | 0.495 |
| t3 n4 | gate | 0.504 | 0.495 |
| t3 n4 | clamp | 0.504 | 0.495 |
| t3 n8 | relu | 0.518 | 0.518 |
| t3 n8 | er | 0.514 | 0.515 |
| t3 n8 | er_memoryless | 0.518 | 0.518 |
| t3 n8 | gate | 0.518 | 0.518 |
| t3 n8 | clamp | 0.518 | 0.518 |
| t4 | relu | 0.0921 | 0.0916 |
| t4 | er | 0.0932 | 0.0926 |
| t4 | er_memoryless | 0.093 | 0.0924 |
| t4 | gate | 0.0931 | 0.0922 |
| t4 | clamp | 0.092 | 0.092 |
