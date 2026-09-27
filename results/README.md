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
| `rerun/` | §18 | every E-R experiment again with the current defaults and three habituation variants; `run.sh` reproduces it |

Runs recorded before 2026-09-27 evening used the logarithmic threshold
growth rule, the default at the time; pass `--set growth=log` to reproduce
them with the current library. The `run.sh` files show the exact commands
(run from the repository root after `./build.sh`).
