## Recalibrated thresholds: activity experiments by learning rate

Medians over trials; columns are learning rates 0.0003, 0.001, 0.003.

### er_economy

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | accuracy | raw | 1 | 0.06 | 0.237 |
| er | accuracy | normalised, rest 0.2 | 0.867 | 0.99 | 1 |
| er | accuracy | normalised, rest auto | 0.863 | 0.987 | 1 |
| er | active_fraction | raw | 0.47 | 0.587 | 0.528 |
| er | active_fraction | normalised, rest 0.2 | 0.463 | 0.469 | 0.472 |
| er | active_fraction | normalised, rest auto | 0.462 | 0.471 | 0.475 |
| gate | accuracy | raw | 0.823 | 0.383 | 0.323 |
| gate | accuracy | normalised, rest 0.2 | 0.817 | 0.963 | 0.993 |
| gate | accuracy | normalised, rest auto | 0.83 | 0.963 | 0.993 |
| gate | active_fraction | raw | 0.465 | 0.73 | 0.679 |
| gate | active_fraction | normalised, rest 0.2 | 0.49 | 0.489 | 0.489 |
| gate | active_fraction | normalised, rest auto | 0.49 | 0.487 | 0.487 |
| linear | accuracy | raw | 0.997 | 0.34 | 0.26 |
| linear | accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | accuracy | normalised, rest auto | 1 | 1 | 1 |
| linear | active_fraction | raw | 1 | 1 | 1 |
| linear | active_fraction | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | active_fraction | normalised, rest auto | 1 | 1 | 1 |

### er_paths

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | accuracy | raw | 0.995 | 0.068 | 0.298 |
| er | accuracy | normalised, rest 0.2 | 0.85 | 0.987 | 0.998 |
| er | accuracy | normalised, rest auto | 0.849 | 0.988 | 0.999 |
| er | internal_switch_rate | raw | 0.0799 | 0.127 | 0.117 |
| er | internal_switch_rate | normalised, rest 0.2 | 0.098 | 0.0861 | 0.0689 |
| er | internal_switch_rate | normalised, rest auto | 0.0964 | 0.0807 | 0.0725 |
| gate | accuracy | raw | 0.831 | 0.366 | 0.308 |
| gate | accuracy | normalised, rest 0.2 | 0.831 | 0.956 | 0.987 |
| gate | accuracy | normalised, rest auto | 0.842 | 0.952 | 0.989 |
| gate | internal_switch_rate | raw | 0 | 0 | 0 |
| gate | internal_switch_rate | normalised, rest 0.2 | 0 | 0 | 0 |
| gate | internal_switch_rate | normalised, rest auto | 0 | 0 | 0 |
| linear | accuracy | raw | 0.994 | 0.332 | 0.252 |
| linear | accuracy | normalised, rest 0.2 | 0.999 | 1 | 1 |
| linear | accuracy | normalised, rest auto | 0.999 | 1 | 1 |
| linear | internal_switch_rate | raw | 0 | 0 | 0 |
| linear | internal_switch_rate | normalised, rest 0.2 | 0 | 0 | 0 |
| linear | internal_switch_rate | normalised, rest auto | 0 | 0 | 0 |

### er_fatigue

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | A_first_accuracy | raw | 1 | 0 | 0.333 |
| er | A_first_accuracy | normalised, rest 0.2 | 0.833 | 1 | 1 |
| er | A_first_accuracy | normalised, rest auto | 0.833 | 1 | 1 |
| er | all_first_accuracy | raw | 1 | 0 | 0 |
| er | all_first_accuracy | normalised, rest 0.2 | 0.333 | 0.667 | 1 |
| er | all_first_accuracy | normalised, rest auto | 0.333 | 0.5 | 1 |
| gate | A_first_accuracy | raw | 0.833 | 0.333 | 0 |
| gate | A_first_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| gate | A_first_accuracy | normalised, rest auto | 1 | 1 | 1 |
| gate | all_first_accuracy | raw | 0.833 | 0.333 | 0 |
| gate | all_first_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| gate | all_first_accuracy | normalised, rest auto | 1 | 1 | 1 |
| linear | A_first_accuracy | raw | 1 | 0.5 | 0.167 |
| linear | A_first_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | A_first_accuracy | normalised, rest auto | 1 | 1 | 1 |
| linear | all_first_accuracy | raw | 1 | 0.5 | 0.167 |
| linear | all_first_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | all_first_accuracy | normalised, rest auto | 1 | 1 | 1 |

### er_history

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | all_accuracy | raw | 0.85 | 0.1 | 0.4 |
| er | all_accuracy | normalised, rest 0.2 | 0.3 | 0.4 | 0.85 |
| er | all_accuracy | normalised, rest auto | 0.3 | 0.35 | 0.9 |
| er | all_pattern_change | raw | 0.859 | 0.573 | 0.583 |
| er | all_pattern_change | normalised, rest 0.2 | 0.77 | 0.792 | 0.781 |
| er | all_pattern_change | normalised, rest auto | 0.77 | 0.794 | 0.782 |
| gate | all_accuracy | raw | 0.8 | 0.45 | 0.35 |
| gate | all_accuracy | normalised, rest 0.2 | 0.85 | 1 | 1 |
| gate | all_accuracy | normalised, rest auto | 0.85 | 0.95 | 1 |
| gate | all_pattern_change | raw | 0 | 0 | 0 |
| gate | all_pattern_change | normalised, rest 0.2 | 0 | 0 | 0 |
| gate | all_pattern_change | normalised, rest auto | 0 | 0 | 0 |
| linear | all_accuracy | raw | 1 | 0.35 | 0.35 |
| linear | all_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | all_accuracy | normalised, rest auto | 1 | 1 | 1 |
| linear | all_pattern_change | raw | 0 | 0 | 0 |
| linear | all_pattern_change | normalised, rest 0.2 | 0 | 0 | 0 |
| linear | all_pattern_change | normalised, rest auto | 0 | 0 | 0 |

### er_silence

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | rest_accuracy | raw | 1 | 0.1 | 0.1 |
| er | rest_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| er | rest_accuracy | normalised, rest auto | 1 | 1 | 1 |
| er | after_accuracy | raw | 1 | 0.1 | 0.4 |
| er | after_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| er | after_accuracy | normalised, rest auto | 1 | 1 | 1 |
| gate | rest_accuracy | raw | 0.8 | 0.4 | 0.2 |
| gate | rest_accuracy | normalised, rest 0.2 | 0.8 | 1 | 1 |
| gate | rest_accuracy | normalised, rest auto | 0.8 | 1 | 1 |
| gate | after_accuracy | raw | 0.8 | 0.4 | 0.2 |
| gate | after_accuracy | normalised, rest 0.2 | 0.8 | 1 | 1 |
| gate | after_accuracy | normalised, rest auto | 0.8 | 1 | 1 |
| linear | rest_accuracy | raw | 1 | 0.4 | 0.3 |
| linear | rest_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | rest_accuracy | normalised, rest auto | 1 | 1 | 1 |
| linear | after_accuracy | raw | 1 | 0.4 | 0.3 |
| linear | after_accuracy | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | after_accuracy | normalised, rest auto | 1 | 1 | 1 |

### er_habituation

| model | metric | variant | lr 0.0003 | lr 0.001 | lr 0.003 |
|---|---|---|---|---|---|
| er | h4_accuracy_sum | raw | 1 | 0.142 | 0.258 |
| er | h4_accuracy_sum | normalised, rest 0.2 | 0.842 | 1 | 1 |
| er | h4_accuracy_sum | normalised, rest auto | 0.833 | 1 | 1 |
| er | h500_accuracy_end | raw | 1 | 0.108 | 0.242 |
| er | h500_accuracy_end | normalised, rest 0.2 | 1 | 1 | 1 |
| er | h500_accuracy_end | normalised, rest auto | 1 | 1 | 1 |
| gate | h4_accuracy_sum | raw | 0.817 | 0.333 | 0.275 |
| gate | h4_accuracy_sum | normalised, rest 0.2 | 0.867 | 0.975 | 1 |
| gate | h4_accuracy_sum | normalised, rest auto | 0.858 | 0.975 | 1 |
| gate | h500_accuracy_end | raw | 0.817 | 0.333 | 0.275 |
| gate | h500_accuracy_end | normalised, rest 0.2 | 0.867 | 0.975 | 1 |
| gate | h500_accuracy_end | normalised, rest auto | 0.858 | 0.975 | 1 |
| linear | h4_accuracy_sum | raw | 1 | 0.4 | 0.225 |
| linear | h4_accuracy_sum | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | h4_accuracy_sum | normalised, rest auto | 1 | 1 | 1 |
| linear | h500_accuracy_end | raw | 1 | 0.4 | 0.225 |
| linear | h500_accuracy_end | normalised, rest 0.2 | 1 | 1 | 1 |
| linear | h500_accuracy_end | normalised, rest auto | 1 | 1 | 1 |
