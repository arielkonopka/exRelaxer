# delayed_credit

Temporal credit assignment in the smallest network that can show it: one
neuron, a cue at t0, `delay` ticks of irrelevant activity, then one reward.
Does the learning rule move the cue's synapse, and not the distractors'?
Results: research log §25; formal definitions of the traces and of the
Sign rule's eligibility: [doc/model.md](../../../doc/model.md#delayed-reward).

```
t0            cue k (one of `cues` inputs) = 1
t0+1..t0+d    distractor inputs: +1 or -1 with probability `distract` each tick
t0+d          after the step: apply_reward(m, lr)     (the only learning call)

cue 0 -> m = +1, cue 1 -> m = -1, other cues -> +1 or -1 at random (balanced)
```

- Inputs feed the neuron directly: no hidden layer, no recurrence, no
  habituation, raw sums; all weights start at `w0` ± `jitter` (seeded).
- The neuron's state and traces are reset before every episode, so an
  episode cannot borrow credit from the previous one.
- Cues are shown equally often and the reward is +1 in exactly half the
  episodes.
- Rules are the library's, unchanged: `sign` (eligibility = E-R threshold
  above rest; input factor = sign of the input at the reward tick) and
  `trace` (P and X traces with decay `trace`).

Metrics (per trial): `success` (the rewarded cue's weight ends above every
irrelevant weight, the punished cue's below every one), `selectivity`
((Δw₊ − Δw₋)/2) and `selectivity_snr` (over the RMS change of the
irrelevant weights; `null` in the JSON when none moved), the weight changes
of relevant cues, irrelevant cues and distractors, `credit_mean` (|P|·X of
the cue at the reward tick), `eligibility_mean` (Sign), and on test
episodes without learning the correlation between the response at t0 and
the reward (`reward_corr`, `reward_corr_relevant`).

```bash
./build.sh --python
export PYTHONPATH=$PWD/build/EXrelaxer.py/package
NNtesting/nntest.py run delayed_credit --trials 10 --set rule=trace --set trace=0.5,0.9 --set delay=0,1,2,4,8,16,32
NNtesting/experiments/delayed_credit/sweep.sh results/delayed-credit     # every run behind §25
python3 NNtesting/tools/temporal_summary.py results/delayed-credit
```
