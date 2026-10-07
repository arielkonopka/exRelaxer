# Experiment results

Raw results behind the [research log](../doc/research.md), one folder per
topic. Each `*.jsonl.gz` is gzipped nntest output (JSON Lines): one
`"type":"trial"` line per trial, with its parameters, seed, metrics and the
machine, compiler and git commit, plus `"type":"summary"` lines. Read them
with `zcat FILE | python3 …`, or unzip them for `NNtesting/tools/compare.py`
and `NNtesting/tools/capacity.py`.

| Folder | Research log | Contents |
|--------|--------------|----------|
| `multimodal/` | §12 | `stereo_depth`, `audiovisual` |
| `learning-rules/` | §11 | `snake_rules` |
| `er-activity/` | §13 | `er_economy`, `er_paths`, `er_fatigue`, `er_history` (+ summaries and traces) |
| `nonlinearity/explore/` | §14 | `nl_static` grid, one file per task; `analysis/` is `capacity.py`'s output |
| `nonlinearity/settle/` | §14 | `nl_static` with more ticks per sample; `analysis_settle7/`, `analysis_settle15/` |
| `nonlinearity/curves/`, `nonlinearity/state/` | §15 | learning curves, state test |
| `er-silence/`, `er-habituation/` | §15, §18 | `er_silence` (+ growth rules, trace), `er_habituation` (+ rule sweep, fade-after sweep) |
| `nonlinearity/temporal/` | §16 | `nl_temporal` grid, one file per task; `summary.txt` |
| `nonlinearity/growth/` | §16 | threshold growth rules on temporal and static tasks |
| `er-training/` | §17 | pretraining without E-R and switching, vs longer E-R training |
| `er-cycles/` | §17 | spontaneous-firing settings in `er_silence` |
| `normalize/` | §19 | normalised weighted sum in the hidden layers, same grids as `rerun/` (off); `summary.md`, `raw/`, `run.sh`; recalibrated resting thresholds × learning rate: `calibration.md`, `run_calibration.sh` |
| `rerun/` | §18 | every E-R experiment again with the current defaults and three habituation variants; `summary.md` (from `analyze.py`), `raw/`, `run.sh` |
| `video-memory/` | §21 | `video_memory`: side task (`side`), recurrence (`side_recurrence`), recovery sensitivity (`side_recovery`), feedback-alignment hidden layer (`side_fa`), order task (`order`); `audit.json`, `summary.md` (from `NNtesting/tools/video_memory_summary.py`); commands in `NNtesting/experiments/video_memory/sweep.sh` |
| `delayed-credit/` | §25 | `delayed_credit`: Sign and Trace rules × trace decay × E-R recovery × delay 0–64, linear neuron, 1600-episode runs; `summary.md` (from `NNtesting/tools/temporal_summary.py`); commands in `NNtesting/experiments/delayed_credit/sweep.sh` |
| `state-output/` | §26 | the State layer (output, threshold, habituation per neuron) on memory_readability, delayed_credit, video_memory_learned, nl_temporal and dyn_ladder, with and without; `summary.md`; commands in `NNtesting/tools/state_output_sweep.sh` |
| `state-ablation/` | §27 | paired ablation of the State layer's three values (output, threshold, habituation; six combinations) on video_memory's `side` task, gaps 0–64, recovery 0.9 and 0.99, readout lr tuned once; `summary.md`; commands in `NNtesting/experiments/state_ablation/sweep.sh` |
| `memory-readability/` | §25 | `memory_readability`: E-R at recovery 0.9 and 0.99 with and without input noise, reset and ReLU controls, blank 0–128 ticks; `summary.md`; commands in `NNtesting/experiments/memory_readability/sweep.sh` |
| `new-defaults/` | §24 | every experiment but Doom on the build before the new defaults and on the new one: normalised hidden layers, normalised readouts, raw sums, amplitude 0.01 vs 0.1, 10× learning rate; `summary.md` (from `analyze.py`), `raw/`, `run.sh`, `run_python.sh` |
| `defaults/` | §23 | `er_silence` at spontaneous amplitudes 0.01 to 1.0 (linear and log growth, with and without recurrence); `run.sh` |
| `dynamic/` | §20 | `dyn_ladder` pilot and E-R at width 256, `nl_temporal` with a frame window, `doom` imitation, `doom_rl` pilots, architectures (depth, feedback, reservoir) and longer E-R runs; evolution (`es_*`: five topologies on defend_the_center, MAP01), the search retest, and the evolved agents' weights and settings (`doom_agent/`, rebuild with `es.build`) |
| `goe/` | §28 | Gardens of Eris ([doc/goe.md](../doc/goe.md)): `untrained_models.jsonl.gz` (G1: untrained networks and a random player) |

Runs recorded before 2026-09-27 evening used the logarithmic threshold
growth rule, the default at the time; pass `--set growth=log` to reproduce
them with the current library. The `run.sh` files show the exact commands
(run from the repository root after `./build.sh`).
| `snake-growth/` | §30 | `snake_growth`: a growing network learning Snake always-on, from ER(1) and ER(4), with and without growth, frozen or plastic old neurons, recovery 0.99, fixed ER(8); saturation of untrained populations; `summary.md`, `runs.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/sweep.py` |
