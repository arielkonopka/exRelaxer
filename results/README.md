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
| `learning-rules-v2/` | §29 | `delayed_credit` with the Sign, Trace, Eligibility and e-prop rules (trace decay 0.9; `*_097`: 0.97) |
| `positive/` | §33 | `snake_rules` with positive-only values and/or weights on all layers, the mix or the readouts; `summary.md` (from `NNtesting/tools/positive_summary.py`), `run.sh` |
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
| `video-memory-learned/` | §22 | `video_memory_learned`: online readouts with and without eligibility traces, recovery, small lr, hidden-layer learning and its controls, credit diagnostics (`credit/`); `summary.md` (from `NNtesting/tools/video_memory_learned_summary.py`) |
| `delayed-credit/` | §25 | `delayed_credit`: Sign and Trace rules × trace decay × E-R recovery × delay 0–64, linear neuron, 1600-episode runs; `summary.md` (from `NNtesting/tools/temporal_summary.py`); commands in `NNtesting/experiments/delayed_credit/sweep.sh` |
| `state-output/` | §26 | the State layer (output, threshold, habituation per neuron) on memory_readability, delayed_credit, video_memory_learned, nl_temporal and dyn_ladder, with and without; `summary.md`; commands in `NNtesting/tools/state_output_sweep.sh` |
| `state-ablation/` | §27 | paired ablation of the State layer's three values (output, threshold, habituation; six combinations) on video_memory's `side` task, gaps 0–64, recovery 0.9 and 0.99, readout lr tuned once; `summary.md`; commands in `NNtesting/experiments/state_ablation/sweep.sh` |
| `memory-readability/` | §25 | `memory_readability`: E-R at recovery 0.9 and 0.99 with and without input noise, reset and ReLU controls, blank 0–128 ticks; `summary.md`; commands in `NNtesting/experiments/memory_readability/sweep.sh` |
| `new-defaults/` | §24 | every experiment but Doom on the build before the new defaults and on the new one: normalised hidden layers, normalised readouts, raw sums, amplitude 0.01 vs 0.1, 10× learning rate; `summary.md` (from `analyze.py`), `raw/`, `run.sh`, `run_python.sh` |
| `defaults/` | §23 | `er_silence` at spontaneous amplitudes 0.01 to 1.0 (linear and log growth, with and without recurrence); `run.sh` |
| `dynamic/` | §20 | `dyn_ladder` pilot and E-R at width 256, `nl_temporal` with a frame window, `doom` imitation, `doom_rl` pilots, architectures (depth, feedback, reservoir) and longer E-R runs; evolution (`es_*`: five topologies on defend_the_center, MAP01), the search retest, and the evolved agents' weights and settings (`doom_agent/`, rebuild with `es.build`) |
| `goe/` | §28 | Gardens of Eris ([doc/goe.md](../doc/goe.md)): `untrained_models.jsonl.gz` (G1: untrained networks and a random player); `events/` (G2: evolution on the event reward; the evolved nets and `fresh.py`); `columns/` (G3: the grown column network and its control, `agent.load(folder)`); `rules/` (G4: online learning rules on the grown network, `summary.md`); `directions/` (G5: untrained and random players with 4 and 8 directions) |
| `snake-growth/` | §30 | `snake_growth`: a growing network learning Snake always-on, from ER(1) and ER(4), with and without growth, frozen or plastic old neurons, recovery 0.99, fixed ER(8); saturation of untrained populations; `summary.md`, `runs.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/sweep.py` |
| `snake-growth-long/` | §31 | `snake_growth` from ER(16) until the size stops changing: caps, recovery 0.5–0.99, `freeze_from`, wide depth growth, weak pruning; the final networks in `nets/`; commands in `NNtesting/experiments/snake_growth/long_run.py` |
| `snake-copy-init/` | §35 | `snake_growth` width growth with copy-init (new neurons copy the most fatigued / active / read / a random neuron, noise and split variants) against random init, ER(1) and ER(4), 10 seeds, 1,500 and 5,000 games; `summary.md`, `runs.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/copy_init.py` |
| `snake-grown-threshold/` | §38 | `snake_growth` width-grown neurons with a lower resting threshold (× 0.1-0.75) or a minimal start threshold, random and copy init, §35's setups and ER(1) at recovery 0.5 with the retuned trigger (30 seeds, 5,000 games); `summary.md`, `runs.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/copy_init.py` |
| `snake-maturation/` | §39 | `snake_growth` ER(1) at recovery 0.5 (retuned trigger): grown neurons with a critical period (Maturation: low threshold, 3× learning gain, maturing over 2k-100k ticks, frozen at maturity or not), freezing at growth on/off, growth on falling firing-pattern entropy; layer diversity per window (30 seeds × 1,500 games, 20 × 5,000); `summary.md`, `runs.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/maturation.py` |
| `precision/` | §32 | 8-bit weights and outputs: `nl_static` (E-R and ReLU, four formats; `nl_static_summary.txt` from `nl_summary.py`) and Snake ER(16) (`snake_er16.txt`); commands in `run.sh` |
| `neuron-models/` | §34 | `nl_static` and `nl_temporal` with every neuron model ([doc/neuron_models.md](../doc/neuron_models.md)); `summary.md` (from `analyze.py`), `raw/`, `run.sh` |
| `reproduce/` | §36 | `reproduce.py` scan: one recorded trial per result file rerun on main (`scan.jsonl.gz`, `scan_legacy.jsonl.gz` with old settings for later parameters); `summary.md` |
| `recovery/` | §36 | every C++ experiment with E-R neurons at recovery 0.9 and 0.5; `summary.md` (from `analyze.py`), `raw/`, `run.sh` |
| `growth-reeval/` | §37 | `snake_growth` again: §30 at recovery 0.5 / 0.75 / 0.9 with 10 seeds, the growth trigger, 8-bit, long runs from ER(1) / ER(4) / ER(16) with every option, copy-init; `summary.md`, `short.jsonl.gz`, `long.jsonl.gz`; commands in `NNtesting/experiments/snake_growth/reeval.py` |
| `td-actor-critic/` | §41-§50 | TD actor-critic (`NNtesting/experiments/td_actor_critic`): `chain/` (delayed-reward chain, tapped delay line and E-R reservoir with recovery spread; `shared_recovery/`: one recovery 0.9), `snake/` (δ vs raw reward vs the original snake, 10 seeds, `run*.sh`), `doom/` (`basic`, `dtc`, `dtc_long`, `dtc_death` (until death), `map01` (first whole-level run, policy never moved), `map01b` (the whole-level comparison), `health_gathering`, `hg_supreme`, `hg_supreme_1k` (§42, homeostatic drives), `my_way_home`, `map01_place` (§43, place cells), `map01_novelty`, `learned_place` (§44), `map01_reflex`, `path_integration`, `map01_path` (§45), `map01_goal`, `anchored_path` (§46), `map01_killer` (§47), `guided_landmarks` (§48), `map01_exit` (§49), `map01_detector` (§50, `probe/`: offline detector comparison): config, test, gzipped training log per run, `run*.sh`) |

Runs recorded before 2026-09-27 evening used the logarithmic threshold
growth rule, the default at the time; pass `--set growth=log` to reproduce
them with the current library. The `run.sh` files show the exact commands
(run from the repository root after `./build.sh`).
