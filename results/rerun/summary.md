## Rerun: linear threshold growth, habituation off / cut after 5 / fade 0.9 after 2

Medians over trials. "log, off": the earlier runs (log growth, no habituation), where available.

### er_economy

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | accuracy | 0.997 | 1 | 1 | 1 |
| er | active_fraction | 0.497 | 0.47 | 0.469 | 0.329 |
| er | spikes_per_inference | 133 | 126 | 126 | 88.17 |
| er | accuracy_per_100_spikes | 0.749 | 0.795 | 0.795 | 1.13 |
| gate | accuracy | 0.785 | 0.823 | 0.827 | 0.957 |
| gate | active_fraction | 0.49 | 0.465 | 0.465 | 0.326 |
| gate | spikes_per_inference | 131 | 124 | 125 | 87.35 |
| gate | accuracy_per_100_spikes | 0.593 | 0.661 | 0.664 | 1.07 |
| linear | accuracy | 0.995 | 0.997 | 0.993 | 0.997 |
| linear | active_fraction | 1 | 1 | 1 | 1 |
| linear | spikes_per_inference | 268 | 268 | 268 | 268 |
| linear | accuracy_per_100_spikes | 0.371 | 0.372 | 0.371 | 0.372 |

### er_paths

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | accuracy | 0.994 | 0.995 | 0.995 | 0.997 |
| er | active_fraction | 0.498 | 0.469 | 0.469 | 0.329 |
| er | switch_rate | 0.251 | 0.257 | 0.25 | 0.267 |
| er | internal_switch_rate | 0.0783 | 0.0799 | 0.0807 | 0.0993 |
| gate | accuracy | 0.785 | 0.831 | 0.869 | 0.962 |
| gate | active_fraction | 0.492 | 0.469 | 0.467 | 0.33 |
| gate | switch_rate | 0.149 | 0.146 | 0.155 | 0.217 |
| gate | internal_switch_rate | 0 | 0 | 8.33e-05 | 0.0584 |
| linear | accuracy | 0.992 | 0.994 | 0.989 | 0.99 |
| linear | active_fraction | 1 | 1 | 1 | 1 |
| linear | switch_rate | 0.15 | 0.158 | 0.152 | 0.153 |
| linear | internal_switch_rate | 0 | 0 | 0.000333 | 8.33e-05 |

### er_fatigue

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | rest_first_accuracy | 1 | 1 | 1 | 1 |
| er | A_first_accuracy | 1 | 1 | 1 | 1 |
| er | A_first_share | 0.392 | 0.136 | 0.137 | 0 |
| er | A_recovery_ticks | 8 | 12 | 12 | 12 |
| er | all_first_accuracy | 0.667 | 1 | 1 | 1 |
| gate | rest_first_accuracy | 0.667 | 0.833 | 0.833 | 1 |
| gate | A_first_accuracy | 0.667 | 0.833 | 0.833 | 1 |
| gate | A_first_share | 1 | 1 | 1 | 1 |
| gate | A_recovery_ticks | 0 | 0 | 0 | 0 |
| gate | all_first_accuracy | 0.667 | 0.833 | 0.833 | 1 |
| linear | rest_first_accuracy | 1 | 1 | 1 | 1 |
| linear | A_first_accuracy | 1 | 1 | 1 | 1 |
| linear | A_first_share | 1 | 1 | 1 | 1 |
| linear | A_recovery_ticks | 0 | 0 | 0 | 0 |
| linear | all_first_accuracy | 1 | 1 | 1 | 1 |

### er_history

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | rest_accuracy | 1 | 1 | 1 | 1 |
| er | busy_pattern_change | 0.28 | 0.458 | 0.451 | 0.358 |
| er | all_pattern_change | 0.534 | 0.859 | 0.858 | 0.661 |
| er | all_decision_change | 0.1 | 0.15 | 0.15 | 0.2 |
| er | all_accuracy | 0.9 | 0.85 | 0.85 | 0.8 |
| gate | rest_accuracy | 0.75 | 0.8 | 0.85 | 0.95 |
| gate | busy_pattern_change | 0 | 0 | 0 | 0 |
| gate | all_pattern_change | 0 | 0 | 0 | 0 |
| gate | all_decision_change | 0 | 0 | 0 | 0 |
| gate | all_accuracy | 0.75 | 0.8 | 0.85 | 0.95 |
| linear | rest_accuracy | 1 | 1 | 1 | 1 |
| linear | busy_pattern_change | 0 | 0 | 0 | 0 |
| linear | all_pattern_change | 0 | 0 | 0 | 0 |
| linear | all_decision_change | 0 | 0 | 0 | 0 |
| linear | all_accuracy | 1 | 1 | 1 | 1 |

### er_silence

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | rest_accuracy | 1 | 1 | 1 | 1 |
| er | after_accuracy | 1 | 1 | 1 | 1 |
| er | silence_active_w4 | 0.004 | 0.005 | 0.005 | 0.005 |
| er | self_active_ticks | 0.0774 | 0.0966 | 0.101 | 0.0966 |
| er | end_threshold_max | 0.463 | 0.017 | 0.017 | 0.0137 |
| gate | rest_accuracy | 0.8 | 0.8 | 0.8 | 1 |
| gate | after_accuracy | 0.8 | 0.8 | 0.8 | 1 |
| gate | silence_active_w4 | 0 | 0 | 0 | 0 |
| gate | self_active_ticks | 0 | 0 | 0 | 0 |
| gate | end_threshold_max | 0.2 | 0.2 | 0.2 | 0.2 |
| linear | rest_accuracy | 1 | 1 | 1 | 1 |
| linear | after_accuracy | 1 | 1 | 1 | 1 |
| linear | silence_active_w4 | 0 | 0 | 0 | 0 |
| linear | self_active_ticks | 0 | 0 | 0 | 0 |
| linear | end_threshold_max | 0.2 | 0.2 | 0.2 | 0.2 |

### er_habituation

| model | metric | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| er | h4_accuracy_sum | 1 | 1 | 1 | 1 |
| er | h500_spikes_per_sample | 26523 | 30107 | 446 | 321 |
| er | h500_accuracy_sum | 1 | 1 | 1 | 1 |
| er | h500_accuracy_end | 1 | 1 | 0 | 0 |
| er | h500_accuracy_per_100_spikes | 0.00377 | 0.00332 | 0.224 | 0.312 |
| gate | h4_accuracy_sum | 0.758 | 0.817 | 0.842 | 0.967 |
| gate | h500_spikes_per_sample | 16384 | 15442 | 155 | 135 |
| gate | h500_accuracy_sum | 0.758 | 0.817 | 0.842 | 0.967 |
| gate | h500_accuracy_end | 0.758 | 0.817 | 0 | 0 |
| gate | h500_accuracy_per_100_spikes | 0.0046 | 0.00534 | 0.551 | 0.724 |
| linear | h4_accuracy_sum | 1 | 1 | 0.992 | 0.992 |
| linear | h500_spikes_per_sample | 33500 | 33500 | 335 | 9318 |
| linear | h500_accuracy_sum | 1 | 1 | 0.992 | 0.992 |
| linear | h500_accuracy_end | 1 | 1 | 0 | 0.992 |
| linear | h500_accuracy_per_100_spikes | 0.00299 | 0.00299 | 0.296 | 0.0106 |

### nl_static

Best median test result over depth {1, 2, 3} × width {4 … 64} (accuracy for binary tasks, MSE otherwise); in brackets the smallest network solving it in ≥ 80 % of seeds (neurons).

| task | model | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| l0 | relu | 0.000487 (4) | 0.000487 (4) | 0.000487 (4) | 0.000448 (4) |
| l0 | er | 0.0447 | 0.0464 | 0.0464 | 0.255 |
| l0 | gate | 0.000769 (32) | 0.000769 (32) | 0.000769 (32) | 0.000743 (16) |
| l0 | clamp | 3.47e-15 (4) | 3.47e-15 (4) | 3.47e-15 (4) | 4.08e-15 (4) |
| l1 | relu | 0.000656 (16) | 0.000656 (16) | 0.000656 (16) | 0.000696 (16) |
| l1 | er | 0.0877 | 0.0342 | 0.037 | 0.0449 |
| l1 | gate | 0.091 | 0.091 | 0.091 | 0.0452 |
| l1 | clamp | 0.111 | 0.111 | 0.111 | 0.111 |
| l2 | relu | 0.000667 (16) | 0.000667 (16) | 0.000667 (16) | 0.000672 (16) |
| l2 | er | 0.0773 | 0.0297 | 0.0297 | 0.0404 |
| l2 | gate | 0.0783 | 0.0783 | 0.0783 | 0.031 |
| l2 | clamp | 0.0986 | 0.0986 | 0.0986 | 0.0986 |
| l3 | relu | 0.000855 (32) | 0.000855 (32) | 0.000855 (32) | 0.000745 (32) |
| l3 | er | 0.074 | 0.0604 | 0.0604 | 0.0741 |
| l3 | gate | 0.119 | 0.119 | 0.119 | 0.0681 |
| l3 | clamp | 0.137 | 0.137 | 0.137 | 0.137 |
| l4 k1 | relu | 0.000829 (24) | 0.000829 (24) | 0.000829 (24) | 0.000863 (24) |
| l4 k1 | er | 0.302 | 0.149 | 0.149 | 0.307 |
| l4 k1 | gate | 0.251 | 0.251 | 0.251 | 0.0842 |
| l4 k1 | clamp | 0.333 | 0.333 | 0.333 | 0.333 |
| l4 k16 | relu | 0.000885 (32) | 0.000885 (32) | 0.000885 (32) | 0.00084 (32) |
| l4 k16 | er | 0.188 | 0.0996 | 0.0996 | 0.347 |
| l4 k16 | gate | 0.102 | 0.102 | 0.102 | 0.0391 |
| l4 k16 | clamp | 0.125 | 0.127 | 0.127 | 0.125 |
| l4 k4 | relu | 0.00072 (32) | 0.00072 (32) | 0.00072 (32) | 0.000749 (32) |
| l4 k4 | er | 0.22 | 0.1 | 0.1 | 0.263 |
| l4 k4 | gate | 0.158 | 0.158 | 0.158 | 0.0545 |
| l4 k4 | clamp | 0.196 | 0.196 | 0.196 | 0.196 |

### nl_temporal

Best median test result over depth {1, 2, 3} × width {4 … 64} (accuracy for binary tasks, MSE otherwise); in brackets the smallest network solving it in ≥ 80 % of seeds (neurons).

| task | model | log, off | off | cut5 | fade2 |
|---|---|---|---|---|---|
| t1 | relu | 0.51 | 0.51 | 1 (4) | 1 (4) |
| t1 | er | 0.952 (64) | 1 (8) | 0.967 (4) | 0.982 (24) |
| t1 | er_memoryless | 0.496 | 0.51 | 0.51 | 0.51 |
| t1 | gate | 0.51 | 0.51 | 1 (8) | 1 (16) |
| t1 | clamp | 0.51 | 0.51 | 1 (8) | 1 (16) |
| t2 | relu | 0.748 | 0.748 | 0.748 | 0.802 |
| t2 | er | 0.844 | 0.755 | 0.775 | 0.748 |
| t2 | er_memoryless | 0.743 | 0.743 | 0.743 | 0.748 |
| t2 | gate | 0.743 | 0.743 | 0.732 | 0.755 |
| t2 | clamp | 0.743 | 0.743 | 0.732 | 0.746 |
| t3 n4 | relu | 0.489 | 0.489 | 0.511 | 0.555 |
| t3 n4 | er | 0.621 | 0.586 | 0.547 | 0.559 |
| t3 n4 | er_memoryless | 0.504 | 0.504 | 0.504 | 0.495 |
| t3 n4 | gate | 0.504 | 0.504 | 0.566 | 0.525 |
| t3 n4 | clamp | 0.504 | 0.504 | 0.513 | 0.517 |
| t3 n8 | relu | 0.518 | 0.518 | 0.518 | 0.518 |
| t3 n8 | er | 0.516 | 0.514 | 0.516 | 0.519 |
| t3 n8 | er_memoryless | 0.518 | 0.518 | 0.518 | 0.518 |
| t3 n8 | gate | 0.518 | 0.518 | 0.518 | 0.518 |
| t3 n8 | clamp | 0.518 | 0.518 | 0.518 | 0.518 |
| t4 | relu | 0.0921 | 0.0921 | 0.0921 | 0.0917 |
| t4 | er | 0.093 | 0.0932 | 0.0932 | 0.09 |
| t4 | er_memoryless | 0.0916 | 0.093 | 0.093 | 0.0917 |
| t4 | gate | 0.0931 | 0.0931 | 0.0916 | 0.0916 |
| t4 | clamp | 0.092 | 0.092 | 0.092 | 0.0921 |

### Other experiments (current defaults, their own habituation settings)

| experiment | metric | median |
|---|---|---|
| bar_orientation | accuracy | 0.99 |
| bar_orientation | accuracy_before | 0.548 |
| bar_orientation | accuracy_control | 0.548 |
| chirp_direction | accuracy | 1 |
| chirp_direction | accuracy_before | 0.54 |
| chirp_direction | accuracy_control | 0.54 |
| snake | apples | 13.44 |
| snake | apples_control | 0.05 |
| snake | train_apples | 7.85 |
| stereo_depth | accuracy_mono | 0.495 |
| stereo_depth | accuracy_stereo | 0.98 |
| stereo_depth | accuracy_two_eyes | 0.49 |
| audiovisual | accuracy_both | 0.965 |
| audiovisual | accuracy_sight | 0.805 |
| audiovisual | accuracy_sound | 0.953 |
| audiovisual | accuracy_teacher | 0.788 |
| audiovisual | accuracy_transfer | 0.677 |
| audiovisual | accuracy_transfer_control | 0.28 |
| snake_rules | apples | 13.44 |
| snake_rules | apples_control | 0.05 |
| snake_rules | apples_per_100_steps | 13.23 |
| snake_rules | apples_per_100_steps_control | 0.434 |
| snake_rules | train_apples | 7.85 |
