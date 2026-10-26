# Do the recorded results still reproduce? (2026-10-26)

`NNtesting/tools/reproduce.py` reran one parameter combination (its first
recorded seed) from every result file in `results/` on main at 8a3eff1 and
compared every metric but timings. `scan.jsonl.gz`: that pass;
`scan_legacy.jsonl.gz`: the files that changed, rerun with `--legacy`, which
gives parameters added after a run the value of the old behaviour (raw sums,
log growth, spontaneous amplitude 0.01, recovery 0.9 for runs before
2026-10-08; the `recovery` option of nl_static / nl_temporal / dyn_ladder /
snake_rules / bar_orientation comes from the PR that adds it).

    NNtesting/tools/reproduce.py results --per-file 1 --max-seconds 60 --out scan.jsonl
    NNtesting/tools/reproduce.py results --only changed@scan.jsonl --legacy --out scan_legacy.jsonl

Trials per folder (same: every metric bit-identical; close: within 1e-6
relative, float summation order; same, old settings: identical with
`--legacy`):

| folder | same | close | same, old settings | changed | error | skipped_slow |
|---|---|---|---|---|---|---|
| defaults |  |  |  | 2 |  |  |
| delayed-credit | 5 |  |  |  |  |  |
| dynamic | 1 |  | 1 | 1 | 3 | 2 |
| er-activity |  |  |  | 4 |  |  |
| er-cycles |  |  | 1 |  |  |  |
| er-habituation |  |  | 1 | 2 |  |  |
| er-silence |  |  |  | 2 |  |  |
| er-training |  |  |  | 2 |  |  |
| learning-rules |  |  |  | 1 |  |  |
| learning-rules-v2 | 4 | 2 |  | 1 |  |  |
| memory-readability |  | 2 |  |  | 1 |  |
| multimodal | 1 |  |  | 2 |  |  |
| neuron-models | 12 |  |  |  |  |  |
| new-defaults | 73 |  | 30 | 23 | 4 |  |
| nonlinearity/curves |  |  |  | 9 |  |  |
| nonlinearity/explore |  |  |  | 9 |  |  |
| nonlinearity/growth |  |  |  | 2 |  |  |
| nonlinearity/settle |  |  |  | 11 |  |  |
| nonlinearity/state |  |  |  | 1 |  |  |
| nonlinearity/temporal |  |  |  | 6 |  |  |
| normalize | 60 |  |  | 3 |  |  |
| positive | 1 |  |  |  |  |  |
| precision | 1 |  |  |  |  |  |
| rerun | 1 |  | 18 | 12 |  |  |
| state-ablation | 3 |  |  |  |  |  |
| state-output | 4 | 1 | 2 |  |  |  |
| video-memory | 5 |  |  |  |  |  |
| video-memory-learned | 2 | 8 |  |  |  |  |

Not in the table: `snake-growth/` (§30) and `snake-growth-long/` (§31) are
not nntest files; all 24 §30 runs and all 60 §31 runs were rerun with
`sweep.py` / `long_run.py` and are bit-identical (every 50-game score and size).

**Everything recorded from §24 (2026-09-29) on reproduces** on current main,
exactly or to float rounding, given the recovery it ran at. What changed is
explained by recorded changes of defaults or builds:

- `new-defaults/*_old`, `*_rawbuild`: deliberately recorded on older builds
  (99164c0 and a raw-sum build); their `_new` / `_norm` twins reproduce.
- `nonlinearity/`, `er-training/`, `normalize/` nl files, `rerun/` nl files,
  `dynamic/nl_temporal_window`: recorded with alpha 1.2 (2.0 since
  2026-09-28), which only the log growth rule uses; alpha is a constant, not
  a parameter, so these cannot be reproduced exactly. §24 reran them.
- `er-activity/`, `er-silence/`, `er-habituation/` (§13, §15) and
  `learning-rules/snake_rules`, `multimodal/` and the non-activity `rerun/`
  files (§11, §12, §18): recorded before the builders normalised sums and
  before the Sign-rule timing fix; §24 reran them and those reruns reproduce.
- **Since PR #48 (2026-10-08), nl_static, nl_temporal, dyn_ladder,
  snake_rules (mix_er) and bar_orientation (readout_er) run E-R at 0.5 with
  no option to say otherwise**, so their §14-§24 commands no longer give the
  recorded numbers. With the new `recovery` option set to 0.9 they do
  (`same, old settings`).

Unexplained, small: `defaults/silence_amplitude*` (§23, 2026-09-28): one
trial's 5-sample probe accuracy 0.6 -> 1.0 and spontaneous-output masses
moved, other metrics within 1%; `learning-rules-v2/eprop_097`: relative
changes ~1e-5 (float order). Errors: Doom needs ViZDoom (not installed here);
`memory-readability/controls`: a parameter's type changed (bool -> int).
