# mnist_gabor

Handwritten digits with the library's recipe for vision: frozen features,
then learned readouts.

```
retina 1x28x28 -> Conv2D Gabor bank (4 orientations x 2 phases, 7x7, frozen)
               -> Pool2D max 4x4 -> Dense mix 512 (random, frozen)
               -> 10 readouts, Dense 1 each (learn)
```

Each class has its own readout layer, so each readout gets its own reward
(`Network.apply_reward_to`). A readout learns only when its sign is wrong
(error-driven): +1 for the image's class, -1 for the others. The prediction
is the readout with the largest output.

Files: `experiment.py` (the task, registered with the harness), `network.py`
(the network). Run it with:

```bash
NNtesting/datasets/fetch.py get mnist
NNtesting/nntest.py run mnist_gabor
NNtesting/nntest.py run mnist_gabor --set dataset=fashion_mnist
NNtesting/nntest.py run mnist_gabor --set mix=0,512,2048 --set lr=0.003,0.01,0.03 --out sweep.jsonl
```

## Findings

Measured 2026-09-26 on 4 threads. One epoch, test on the first 2000 test
images. Without learning, accuracy is 0.08 to 0.12.

| setting | accuracy |
|---|---|
| defaults (10 000 training images, mix 512, lr 0.01), 3 trials | 0.900 ± 0.009 |
| 20 000 images, mix 512, lr 0.003 / 0.01 / 0.03 | 0.908 / 0.906 / 0.895 |
| 20 000 images, mix 2048 | 0.904 (lr 0.003) |
| 20 000 images, readouts on the pooled Gabor features (mix 0) | 0.53 to 0.59 |

The frozen random mix is what makes the readouts work. Pooled Gabor
responses have fairly fixed signs, and the learning rule only uses the
sign of each input. A default run takes about 2 seconds per trial.
