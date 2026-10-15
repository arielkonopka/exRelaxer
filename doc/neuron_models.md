# Neuron models

Every layer of neurons has a **neuron model** (`NeuronModel`, in
`core/neuron.hpp`): what turns a neuron's weighted sum into its output. The
default, `ER`, is the original neuron: E-R with `hasER`, otherwise linear,
gated (`gate`), ReLU (`rectify`) or binary (`binary`; perceptrons). The
other models are the usual neuron types of other libraries, so that E-R can
be compared with them, and mixed with them, in one network:

| model | kind | output | state |
|---|---|---|---|
| `Sigmoid` | rate | `1 / (1 + e^(-gain s))`, in (0, 1) | none (smoothed output with `smoothing`) |
| `Tanh` | rate | `tanh(gain s)`, in (-1, 1) | likewise |
| `Softplus` | rate | `ln(1 + e^(gain s)) / gain`, a smooth ReLU | likewise |
| `LIF` | spiking | 1 on a spike, else 0 | membrane v, threshold rise a, refractory ticks |
| `Izhikevich` | spiking | 1 on a spike, else 0 | membrane v (mV), recovery u |
| `Resonator` | spiking | 1 on a spike, else 0 | z = Re z + i Im z, refractory ticks |

A layer chooses its model with `LayerSpec::neuronModel` (or
`neuron_layer::setNeuronModel`). Models other than `ER` are for layers
without E-R and without a gate, rectification or binary output; the layer
throws `std::invalid_argument` otherwise.

```cpp
net.addLayer("h", LayerSpec::Dense(64, NeuronModel::tanh()));
net.addLayer("s", LayerSpec::Dense(64, NeuronModel::alif(0.9f, 1.0f, 0.2f)));
net.addLayer("i", LayerSpec::Dense(64, NeuronModel::chattering()));
net.addLayer("r", LayerSpec::Dense(64, NeuronModel::resonator(0.1f)));
```

```python
from exrelaxer import LayerSpec, NeuronModel
net.add_layer("h", LayerSpec.dense(64, neuron_model=NeuronModel.tanh()))
spec = LayerSpec.dense(64, habituation=True, neuron_model=NeuronModel.lif(leak=0.8))
```

`LayerSpec::Dense(size, model, hasHabituation = false)` builds a Dense layer
without E-R and, by default, without habituation. Everything else a layer
does works with every model:

- **Habituation** still acts on the sum before the model: a repeated sum is
  cut (or faded) and the model sees 0.
- **Learning rules**: all of them (see below).
- **Growth**: grown neurons get the layer's model.
- **Saving**: network format 21 saves each layer's model, and each neuron's
  model state (neuron format 5), so a full-state load continues
  bit-identically; a weights-only load puts the model at rest. Files from
  before format 21 load as `ER`, unchanged.
- **State taps** read a model neuron's *charge* in the threshold field (see
  below).
- `network::describe` names the model in the E-R column (`sigm`, `tanh`,
  `splus`, `lif`, `alif`, `izh`, `res`).

One tick is one time step for every model; Izhikevich takes it as 1 ms.

## Rate units: sigmoid, tanh, softplus

`y = f(gain * s)` with `gain > 0` (default 1). With `smoothing` m in
[0, 1) the unit becomes a leaky integrator, the unit of a continuous-time
RNN: `y = m * y_prev + (1 - m) * f(gain * s)`. They are the standard smooth
nonlinearities; tanh is signed like E-R, sigmoid and softplus are not.

## Leaky integrate-and-fire (LIF) and adaptive LIF (ALIF)

```
v = leak * v + s                          (leak in [0, 1), default 0.9)
spike when v > threshold + a              (threshold > 0, default 1)
on a spike: v = reset, `refractory` ticks follow in which v stays at reset
a = adaptationDecay * a + adaptation * spike
```

The output is 1 on a spike, else 0. With `adaptation = 0` (`NeuronModel::lif`)
`a` stays 0: a plain LIF. With `adaptation > 0` (`NeuronModel::alif`) every
spike raises the threshold, which decays back: the adaptive LIF of Bellec et
al.'s LSNN and e-prop. It is spike-frequency adaptation like E-R's, but
added to a membrane that integrates over time, with an all-or-nothing output.
Inhibition (a negative sum) drives `v` below 0; the leak bounds it.

## Izhikevich

Izhikevich (2003), with one tick = 1 ms and `I = gain * s`:

```
v' = 0.04 v^2 + 5 v + 140 - u + I      (two Euler half-steps per tick)
u' = a (b v - u)
spike when v >= 30 mV: v = c, u = u + d
```

Presets (`gain` default 10, so a sum of 1 is I = 10):

| preset | a | b | c | d | behaviour |
|---|---|---|---|---|---|
| `regularSpiking` (default) | 0.02 | 0.2 | -65 | 8 | adapting regular spikes |
| `fastSpiking` | 0.1 | 0.2 | -65 | 2 | fast, non-adapting |
| `chattering` | 0.02 | 0.2 | -50 | 2 | bursts |
| `intrinsicallyBursting` | 0.02 | 0.2 | -55 | 4 | a burst, then single spikes |

The membrane is kept at or above -200 mV, so strong inhibition stays finite.

## Resonate-and-fire

Izhikevich (2001): a damped oscillator with its own frequency,

```
z = leak * e^(2 pi i frequency) * z + i s     (leak = damping in [0, 1), frequency in (0, 0.5])
spike when Im z > threshold: z = 0, then `refractory` ticks ignore the input
```

Input that oscillates near the neuron's frequency builds up and fires it;
slower, faster or constant input cancels out. A layer of resonators with
different frequencies is a bank of band-pass detectors over time.

## Learning

Every learning rule works with every model:

- **Sign**: a neuron is eligible while its output is non-zero (as for any
  neuron without E-R): a spiking neuron learns on spike ticks.
- **Trace, Perturbation, Oja, BCM, Eligibility**: use the output (and its
  traces) as they are.
- **Feedback alignment**: the surrogate derivative is `neuron::slope()`:
  `dy/ds` of a rate unit, and for spiking models a triangular
  pseudo-derivative that is 1 at the threshold (or on a spike) and 0 half
  a threshold away.
- **EProp, Surrogate**: the local derivatives (`Derivatives`) carry the
  model's state variable in place of the E-R threshold: LIF's membrane
  (`dthr'/dthr = leak`, `dthr'/ds = 1`, `dy/ds` the pseudo-derivative at
  `leak * v + s`), which makes e-prop's eligibility exactly Bellec et al.'s
  `psi * (x + leak * a)`; a resonator's Im z (`decay = leak * cos(2 pi f)`);
  an Izhikevich neuron's v (Jacobian of one Euler step, ignoring u, clipped
  to [0, 1]); a smoothed rate unit's output (`m`). As in e-prop, resets are
  not differentiated, and the second state variable (ALIF's a, Izhikevich's
  u, the resonator's Re z) is ignored: these are approximations for the
  spiking models.

## Probes

`neuron::membrane()` (LIF/Izhikevich v, resonator Im z, smoothed rate
output), `neuron::recoveryVariable()` (ALIF's a, Izhikevich's u, resonator
Re z) and `neuron::charge()`, how close the neuron is to firing: 0 at rest,
1 at the threshold (LIF `v / (threshold + a)`, Izhikevich `(v - c) / (30 - c)`,
resonator `Im z / threshold`; 0 for rate units). Python's
`Network.state_probe` returns them as `membrane`, `recovery_variable` and
`charge`; a [State layer](state_output.md) reads the charge in its threshold
field.

## In the experiments

`nl_static`, `nl_temporal` and `dyn_ladder` accept the models by name:
`model=sigmoid,tanh,softplus,lif,alif,izhikevich,resonator`, with
`model_gain`, `model_smoothing`, `model_leak`, `model_threshold`,
`model_adaptation` and `model_frequency` (see
[nonlinearity](nonlinearity.md)).
