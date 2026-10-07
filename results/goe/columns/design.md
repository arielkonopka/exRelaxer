# A growing E-R agent for Gardens of Eris: review of the proposal

2026-10-06. No code was written for this. It is a judgement of the idea, checked against the game's
current agent interface (Gardens-of-Eris main at `3d958c3`, which includes today's commit
`7179997`, "per-cell novelty memory") and against what exRelaxer already measured (doc/goe.md G1–G2,
doc/doom.md D5–D12, research §13–§23).

## The proposal, as I read it

```
 eye (vision grid, + novelty)           body + inventory
   │                                        │
   ▼                                        │
 retina: one habituated neuron per cell,    │
 no weights between them ("unconnected")    │
   │                                        │
   ▼                                        │
 pattern layer                              │
   │                                        │
   ▼                                        ▼
 E-R layer L1 ◄───────────────────────── body, inventory
   │
   ▼            (added when L1 beats random)
 E-R layer L2 ◄── retina output, body, inventory, L1 output
   │
   ▼            (added when L2 beats random)
 E-R layer L3 ◄── retina output, body, inventory, L2 (L1) output
   ...
 + when a layer is "tired" (no neuron free to pass the signal), add neurons to it
```

## Short verdict

The skeleton is sound and most of it already exists in the library. Three parts need to change
before it is worth running:

1. **The growth trigger.** "Better than random" is the wrong reference and, with 6 validation worlds,
   too noisy to act on. Grow on a plateau after demonstrated competence, measured on many fixed
   worlds against the network's own untrained self.
2. **What a new layer reads.** Feeding the whole retina into every new layer makes each layer about
   200 000 evolving weights, which ES cannot search (G2 moved 300 generations for +8 reward on a
   network that size). New layers should read the compact pattern layer, not the raw grid.
3. **"Tired → add neurons".** Fatigue is the thing this project is studying, so treating it as a
   capacity shortage works against the research question. Add neurons on a task signal instead,
   or not at all in the first version.

The cheapest, most informative step is not growth at all: first measure whether the game's new
novelty channel and a habituating retina change what one E-R layer can learn (experiment at the end).

## What changed in the game's interface

The goe_rl experiment still builds its eye from 6 element features mapped to 8 channels (wall, free,
enemy, collectible, danger, door, apple, moving) plus an optional `seen` channel that is just
`in_sight`. Since today the game also offers, per vision cell:

| Cell feature | Meaning |
|---|---|
| `visits` | times the player stepped on this board cell this episode |
| `seen` | steps that ended with the cell in sight |
| `novelty` | 1 / sqrt(1 + seen): 1 for a never-seen cell, falling with familiarity |
| event `explore` | cells seen for the first time this step (weight 0 by default) |

It also exposes many element features (`locked`, `open`, `dying`, `facing`, `direction`, ...) and the
full inventory (5 sections × slots × item features, plus counts). None of that is wired into goe_rl yet.
Note the name clash: the game's new `seen` is a count, goe_rl's `seen` channel is `in_sight`.

**Why this matters for the design.** The novelty channel is anchored to the *world* (by board cell)
and remembered for the whole episode. That is exactly the memory a maze needs ("have I been here?"),
and it is handed to the network as an input. So the network no longer has to build that memory in
E-R thresholds or in deep recurrent stacks, which removes much of the reason to grow deep.

## Part by part

### 1. Habituated unconnected neurons for vision

The library already has this: `Retina` (core/layers/retina.hpp) is one neuron per sample and channel,
fixed weights, habituation and E-R per sample, "like a photoreceptor".

Strengths:
- Cheap (no evolving weights) and true to the project's thesis: activity that adapts to history.
- Complements the game's novelty well. Novelty is long-term and world-anchored ("this place is
  familiar"); retina habituation is short-term and eye-anchored ("this part of my view changed").

Risks:
- **The view moves with the player.** Every move shifts the whole grid by one cell, so every edge in
  the view "changes" and the retina fires everywhere. When the player stands still, everything fades
  except moving things. The retina therefore reports self-motion when moving and enemies when still.
  That is useful, but it is not novelty; do not expect it to replace the novelty channel.
- **The silence trap.** G1: the untrained network plays `MOVE_UP` in 93–99% of moves, mostly
  standing against a wall. A habituating retina makes a still scene go silent, the readouts go to
  zero, the tie picks the first action again, and the player stays still. Doom had the same failure
  (D9: held screen → E-R silent at the last tick → action 0 always → no ES signal). Mitigations: fade
  mode rather than cut (already the default), keep a non-habituated copy of the static channels
  beside the habituated ones, and break readout ties at random instead of by index.
- **Binary inputs.** The grid is 0/1. G1 measured 2 spikes per move of 128 on it without recurrence.
  A retina of E-R neurons on 0/1 inputs will be very sparse; check its spike count before building
  on it.

### 2. A pattern layer

Agreed, and here the choice of layer matters more than anything else in the proposal. In a grid
world the same pattern (a door next to me, a monster two cells left) means the same thing anywhere
in the view, so a **small convolution** (Conv2D, 3×3, a few dozen filters, shared weights) is the
natural choice. It turns ~1500 retina outputs into features with only a few thousand weights, which
is the scale ES handles well. A dense pattern layer on the full grid is ~200 000 weights and is
where G2's slowness came from.

### 3. E-R layer with body and inventory

Fine. Two notes:
- Inventory is new to this agent. Most slots are empty most of the time (type −1); encode type as
  one-hot or as counts per section rather than raw type numbers, or the E-R thresholds will track a
  meaningless constant.
- Memory: the State-layer ablation (PR #30) found E-R memory lives in the threshold,
  and Doom's best agent was one E-R layer with fading habituation. One good layer here is a strong
  baseline that growth must beat.

### 4. Growing layers: the trigger

This is the part I would change most.

**"Better than random" is the wrong bar.**
- In GoE a random player is a strong explorer: event reward 13.8 and score 71.7 on fresh worlds,
  above every untrained network (G1, G2). The best evolved network after 300 generations reached
  18.4. So L1 may cross "random" once, late, and L2 may never get its chance; the trigger would mostly
  measure how hard GoE is for a random player, not whether L1 learned.
- "A layer performs better than random" is also ill-defined for an inner layer: a layer has no
  policy of its own. If it means "the network with this layer", say so; if it means the layer's
  representation, the right reference is the same layer with **random, untrained weights** (Doom:
  untrained E-R features were already worth ~1 kill per episode; readout-only ES on random features
  hit a ceiling in D7).

**Noise.** Validation is 6 worlds a generation. Both projects already paid for this:
- Doom search picked lucky nets (trained ≈ untrained on retest).
- G2 `er_nohab`: best validation +27, but final weights fell back to +9.2 on fresh worlds.
- Doom D10: growth 1→3 triggered at generations 77 and 133 and *led validation* (+4.36), yet scored
  +2.21 on 30 fresh games against +3.78 for the never-grown single layer.

Checking "better than X" every generation is also a repeated test, so false triggers are guaranteed
eventually unless the test corrects for it.

**What I would use instead.**
- Competence: the current network beats its own untrained self (and, for the record, the random
  player) on a **fixed set of ~30 held-out worlds**, as a paired comparison (same worlds), with a
  sign test or bootstrap interval that excludes zero.
- Plateau: the validation trend over the last N generations is flat (slope within noise).
- Grow when **both** hold, and confirm on a second, separate set of worlds before growing.
- Measure afterwards whether the new layer is used: zero its readout lines and replay the fixed
  worlds. If nothing drops, it was dead weight.

This is the cascade-correlation logic: add capacity when learning *stalls*, not when it *starts*.
Growing at the moment a layer begins to work also means the budget moves to a fresh random layer just
when the old one was improving.

The existing `es.py --grow-to` already grows on "the last 20 generations beat the first 20 by a
margin", carries weights over and zero-initialises the new readout lines so play is preserved
exactly. The trigger is the part to replace; the machinery can stay.

### 5. Each new layer reads raw vision + other inputs + the older layer

Structurally this is cascade-correlation (each new unit sees the inputs and every earlier unit) and
progressive networks (new column with lateral connections from frozen old ones). Both are well-tested
ideas, and the skip from the inputs is what makes them work: a new layer does not have to relearn
the world from the old layer's compressed view.

The problem is size. With radius 6 and 9 channels the retina is 1521 outputs, so 128 new neurons
reading it are ~195 000 weights per layer. Recommendation: new layers read the **pattern layer**
output (and body/inventory, and earlier layers), not the raw retina. If raw vision is wanted, give
each new neuron a small random subset of retina inputs (sparse fan-in, e.g. 32) or the novelty
channel only, which is the part most likely to add new information.

### 6. Freezing older layers or continuing to train them

Recommendation: **freeze old hidden layers' input weights, keep evolving all readouts and the new
layer.** Reasons:
- ES cost grows with the number of evolving weights; freezing keeps each stage's search the size of
  one layer.
- No forgetting: what L1 learned is kept by construction (progressive networks' argument).
- Doom showed training hidden weights matters (D9: evolve all +3.78 vs readouts only +1.36), so the
  *new* layer must be trained; freezing old ones does not contradict that.

Optional later: a short fine-tune of everything at a lower step size after each growth, compared
against freezing on the same fixed worlds.

### 7. "Tired" layer → add neurons

I would not do this in the first version.
- It works against the project's question. Fatigue and laziness are what the research is looking
  for; adding neurons whenever they appear removes the pressure that produces them.
- More neurons reading the same input tire in the same way. If a layer is tired because its input is
  constant (a still scene), new neurons fed that input will be tired a few ticks later.
- The audit found the earlier "fatigue rerouting" was a metric artifact (unfatigued paths were
  bit-identical to rest), so there is no measured mechanism yet by which fatigue redirects signal to
  spare neurons.
- If capacity must grow later, trigger on a task signal: neurogenesis models (e.g. Draelos et al.,
  "Neurogenesis deep learning", 2017) add units when the network fails on novel inputs, not when
  units are tired. Add each new neuron with zero outgoing weights (Net2Net-style), so play is
  unchanged until evolution uses it. "Tired" itself needs a definition first, for example the share
  of neurons whose threshold has stayed above their input sum for more than k ticks.
- Tuning recovery and fade (already layer settings) is the cheaper first answer to too much fatigue.

## Recommended first experiment (G3)

Before any growth, find out what the new inputs buy one E-R layer. It also measures the noise that
any growth trigger must beat.

- **Network:** G2's `er_rec` (1 E-R layer, 128 neurons, fading habituation, recurrent), event reward,
  2-minute games, 300 generations, evolve all.
- **2 × 2 factors:**
  - vision: current 8 channels **vs** 8 channels + game `novelty` (and `visits`);
  - front end: none **vs** a habituating `Retina` before L1, with a non-habituated copy beside it.
- **Reward:** the G2 weights, plus one extra arm with a small `explore` weight (e.g. 0.1), since the
  game now counts it.
- **Evaluation:** 3 net seeds per cell; best and final weights on the same 30 fresh worlds
  (4242–4271), next to the untrained network and the random player. Report reward, events, spikes per
  move and the share of moves spent standing still.
- **What it decides:**
  - If novelty alone gives a large gain, memory is solved by input, and deep growth should be judged
    against that stronger one-layer baseline.
  - The seed-to-seed spread on fresh worlds is the noise floor; the growth trigger's margin should be
    set above it.
  - If the retina front makes the player freeze, the silence-trap fixes above come first.

Then G4: conv pattern layer + growth with the plateau-and-competence trigger, frozen old layers, new
layers reading the pattern layer, compared on the same fresh worlds against a fixed network of the
same final size trained from the start.

## Addendum: pre-training on fixed chunks

The game can build chunks from fixed patterns (`setChunkPattern`, `setDefaultPattern`,
example `agent/patterns/rooms.json`); set them before `newEpisode`, since the 5 × 5 chunks around
the start are built with the episode and `generateChunk` does not rebuild them.

Why it helps:
- **Rare events become common.** G2's kills, apples, teleports, uses and mines almost never happened in
  2-minute random mazes, so ES had nothing to reward. Rooms built around one skill (key + door, a
  monster in a corridor, an apple behind a teleporter) make each event frequent enough to learn.
- **Less noise.** Candidates meet the same situations, so the growth trigger's comparison is sharper.
- **It fits growth as a curriculum.** One stage per skill, a layer grown per stage and older layers
  frozen: the progressive-networks setting, where each column is trained on its own task.

Risks:
- **Memorising instead of seeing.** The game is deterministic: same pattern + same seed + same actions
  give the same game. ES can learn an open-loop move sequence that ignores the eye. Vary placement
  (several patterns per skill, positions shuffled by seed) and mix in random mazes (for example half
  of the worlds).
- **Transfer is the real test.** Always validate and report on random fresh worlds (4242–4271), never
  only on the patterns.

Where it goes in the plan: after G3 (which inputs help), as stage-wise pre-training before or together
with growth in G4.
