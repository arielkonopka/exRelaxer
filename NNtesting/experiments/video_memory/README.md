# video_memory

Can a feed-forward E-R layer keep temporal information from a video
stream in its neurons' own state, with no recurrence and no frame history?
And how much explicit frame history is that worth? Results: research log
§21 and [doc/video_memory.md](../../../doc/video_memory.md).

```bash
./build.sh --python
export PYTHONPATH=$PWD/build/EXrelaxer.py/package
python3 NNtesting/experiments/video_memory/audit.py --out audit.json      # the final frame alone is at chance
NNtesting/nntest.py run video_memory --trials 20 --set model=C0,C1,E0,R1,R4,E1,Rg --set gap=1,2,4,8,16 --out side.jsonl
python3 NNtesting/tools/video_memory_summary.py side.jsonl --md summary.md --png curves.png
NNtesting/experiments/video_memory/sweep.sh results/video-memory          # every run behind §21
```

## Clips (clips.py)

20 × 15 grayscale frames in [0, 1], one small square (2 or 3 px, brightness
0.5–1, any row, 1 or 2 px per frame, 3–6 visible frames, background noise
sd 0–0.05, all drawn per clip and independent of the label). Balanced
classes; train, validation and test clips come from different seeds, and a
seed gives the same clips at every gap and for every model.

```
visible motion (3-6 frames) -> gap blank frames (input exactly 0) -> reappearance, static (2 frames: the readout interval)
```

- `side` (primary): the object moves towards the centre and vanishes 1–4
  px short of it, so a right-moving object is last seen left of centre. It
  reappears at the centre. Knowing where it was is enough.
- `order`: motion wraps around the screen and starts anywhere; every
  single frame has the same distribution for both classes, and only the
  order of two frames tells the direction.

`audit.py` checks that a classifier given only the final frame (or both
readout frames) is at chance on 20,000 clips, linear or with 2048 random
ReLU features, and that the frames before the blank do carry the answer.

## Network

Input (20 × 15, plus previous frames for windowed models) → 128 hidden
neurons, frozen after a random initialization → 2 readouts (LEFT, RIGHT).
Each frame is held for 4 ticks; the readout interval is the two
reappearance frames (8 ticks), and the larger mean readout output wins.

| Model | Hidden | Shown each frame |
|-------|--------|------------------|
| C0 | ReLU | current frame |
| C1 | E-R, reset to rest at every frame (the reset ablation: E0's weights and inputs) | current frame |
| E0 | E-R | current frame |
| R1 | ReLU | current + previous frame |
| R4 | ReLU | current + previous 3 frames |
| E1 | E-R | current + previous frame |
| Rg | ReLU | the shortest window that still reaches the last visible frame (side: gap + 1 previous frames; order: gap + 2) |
| D2 | E-R → E-R | current frame (depth control for D2R) |
| D2R | E-R → E-R, the second layer also reads its own previous output (recurrent scale 0.5) | current frame |

Hidden layers: local weight normalisation (x·w / |w|), no habituation,
linear threshold growth, recovery 0.9, resting threshold 0.2, weight and
output clamps 10: the library defaults. One setting for every gap. The
readouts are the repository's plain readout neurons (raw weighted sum plus
bias, clamped); their weights are fitted by ridge regression of ±1
targets on the mean hidden output of the readout interval, the penalty
chosen on validation clips. Since the hidden layer is frozen and
feed-forward, its activity is recorded once per clip and the readouts are
fitted on it (the online error rule of the library is available as
`lms_lr` for comparison; it finds the strong signals, not E0's weak one).

## Metrics

`accuracy` (test, 1000 clips), `probe_pre_occlusion` and
`probe_before_reappearance` (a ridge classifier on the complete hidden
state, E-R thresholds and outputs, after the last visible frame and after
the last blank frame), spikes per frame (all, visible, blank, readout, and
the first readout tick), E-R thresholds (mean before the blank, during it,
and the 10th/50th/90th percentile at the reappearance).

## Separate studies

- `recovery` (0.95, 0.97, 0.99): sensitivity of E0 to the E-R recovery.
- `hidden_rule=fa`: the hidden layer and readouts first trained online by
  feedback alignment from the error of every readout tick, then read out
  like the frozen models (`fa_online_accuracy` is the online readouts').
