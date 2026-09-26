"""The mnist_gabor network: retina -> frozen Gabor bank -> max pool -> frozen random mix -> one readout per class.

Each class has its own 1-neuron output layer, so each can get its own reward
(Network.apply_reward_to): one-vs-rest readouts on shared frozen features.
"""
import exrelaxer as exr


def build(p, shape, classes):
    """`shape`: (channels, height, width) of an image; `classes`: number of readouts. Returns (net, readout ids)."""
    net = exr.Network()
    image = exr.Shape(*shape)
    k = p["kernel"]
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
    v1 = net.add_layer("v1", exr.LayerSpec.conv2d(2 * p["orientations"], exr.Window2D.square(k, 1, k // 2),
                                                  False, False))
    pool = net.add_layer("pool", exr.LayerSpec.pool2d(exr.Window2D.square(p["pool"], p["pool"])))
    net.add_inputs(eye, image)
    net.connect(eye, v1)
    net.connect(v1, pool)
    features = pool
    if p["mix"] > 0:
        features = net.add_layer("mix", exr.LayerSpec.dense(p["mix"], False, False, frozen=True))
        net.connect(pool, features)
    readouts = []
    for c in range(classes):
        out = net.add_layer(f"out{c}", exr.LayerSpec.dense(1, False, p["readout_er"]))
        net.connect(features, out)
        net.add_output(out)
        readouts.append(out)
    net.load_filters(v1, exr.filters.gabor_bank(k, p["orientations"], p["wavelength"], p["sigma"]))
    net.freeze(v1)
    return net, readouts
