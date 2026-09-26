"""The Python experiment runner (exrelaxer.harness) and the datasets helper."""
import json
import textwrap

import numpy as np
import pytest

import exrelaxer as exr
from exrelaxer import datasets, harness


@pytest.fixture
def experiments_dir(tmp_path, monkeypatch):
    """A folder experiment (with its own helper module) and a single-file one, in a fresh registry."""
    monkeypatch.setattr(harness, "_registry", [])
    monkeypatch.setattr(harness, "_loaded", set())
    root = tmp_path / "experiments"
    folder = root / "folder_task"
    folder.mkdir(parents=True)
    (folder / "helper.py").write_text("def scale(x):\n    return 2 * x\n")
    (folder / "experiment.py").write_text(textwrap.dedent("""
        import exrelaxer as exr
        from exrelaxer import harness as nnt
        from helper import scale

        @nnt.experiment(tags=["quick"], params={"n": (3, "neurons"), "er": (False, ""), "gain": (1.5, "")},
                        trials=4, expect={"value": (0, None)})
        def run(t):
            net = exr.Network()
            layer = net.add_layer("a", exr.LayerSpec.dense(t.params["n"], er=t.params["er"]))
            net.add_inputs(layer, 1)
            net.add_output(layer)
            net.set_inputs([1.0])
            net.step()
            t.record("value", scale(t.params["gain"]))
            t.record("first_output", float(net.outputs()[0]))
            t.record("draw", float(t.rng.uniform()))
    """))
    (root / "single_file.py").write_text(textwrap.dedent("""
        from exrelaxer import harness as nnt

        @nnt.experiment(name="single", description="one file", params={"fail": (False, "")},
                        trials=2, expect={"x": (None, 0.5)})
        def run(t):
            if t.params["fail"]:
                raise RuntimeError("asked to fail")
            t.record("x", 1.0)
    """))
    (root / "not_an_experiment").mkdir()
    (root / "cpp_experiment.cpp").write_text("// ignored")
    return root


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_discovers_folders_and_files(experiments_dir, capsys):
    assert harness.main(["--experiments", str(experiments_dir), "list"]) == 0
    listing = capsys.readouterr().out
    assert "folder_task" in listing and "(folder)" in listing
    assert "single" in listing and "(file)" in listing
    assert [e.name for e in harness.experiments()] == ["folder_task", "single"]


def test_run_writes_nntest_results(experiments_dir, tmp_path, capsys):
    out = tmp_path / "results.jsonl"
    status = harness.main(["--experiments", str(experiments_dir), "run", "folder_task",
                           "--set", "gain=1.5,4", "--set", "er=off", "--out", str(out)])
    assert status == 0
    records = read_jsonl(out)
    trials = [r for r in records if r["type"] == "trial"]
    summaries = [r for r in records if r["type"] == "summary"]
    assert len(trials) == 8 and len(summaries) == 2
    # Parameters are recorded as text, like the C++ runner; typed for the experiment.
    assert summaries[0]["params"] == {"n": "3", "er": "off", "gain": "1.5"}
    assert summaries[1]["metrics"]["value"]["mean"] == 8.0
    assert [t["seed"] for t in trials[:4]] == [0, 1, 2, 3]
    assert set(summaries[0]["env"]) >= {"run", "time", "host", "cpu", "compiler", "build", "native", "openmp",
                                        "threads", "git", "dirty"}
    # Seeded: the same seeds give the same trials.
    harness.main(["--experiments", str(experiments_dir), "run", "folder_task", "--set", "gain=1.5,4",
                  "--set", "er=off", "--out", str(tmp_path / "again.jsonl")])
    again = [r for r in read_jsonl(tmp_path / "again.jsonl") if r["type"] == "trial"]
    assert [t["metrics"] for t in again] == [t["metrics"] for t in trials]
    # "er=off" is not the default text "false", so no checks ran.
    assert summaries[0]["defaults"] is False and summaries[0]["checks"] == []


def test_checks_and_failures_set_the_exit_status(experiments_dir, tmp_path, capsys):
    args = ["--experiments", str(experiments_dir), "run"]
    assert harness.main(args + ["folder_task"]) == 0
    assert "check value >= 0: pass" in capsys.readouterr().out
    assert harness.main(args + ["single"]) == 1  # x = 1 fails x <= 0.5
    assert "check x <= 0.5: FAIL" in capsys.readouterr().out
    assert harness.main(args + ["single", "--set", "fail=true"]) == 1
    assert "failed: RuntimeError: asked to fail" in capsys.readouterr().out


def test_selectors_and_usage_errors(experiments_dir, capsys):
    args = ["--experiments", str(experiments_dir), "run"]
    assert harness.main(args + ["tag:quick", "--trials", "1"]) == 0
    assert "folder_task" in capsys.readouterr().out
    assert harness.main(args + ["fold*", "--trials", "1"]) == 0
    assert harness.main(args + ["nothing"]) == 2
    assert harness.main(args + ["folder_task", "--set", "missing=1"]) == 2
    assert harness.main(args + ["folder_task", "--set", "n=abc"]) == 2
    assert harness.main(args + ["folder_task", "--trials", "0"]) == 2
    assert "no experiment matches" in capsys.readouterr().err


def test_compare_reads_python_results(experiments_dir, tmp_path):
    import subprocess
    import sys
    compare = datasets.find_nntesting(__file__)
    if compare is None:
        pytest.skip("not in the repository")
    out = tmp_path / "results.jsonl"
    harness.main(["--experiments", str(experiments_dir), "run", "folder_task", "--set", "gain=1,2",
                  "--out", str(out)])
    table = subprocess.run([sys.executable, str(compare / "tools" / "compare.py"), str(out)],
                           capture_output=True, text=True, check=True).stdout
    assert "folder_task" in table and "gain" in table


def test_params_parse_like_their_defaults():
    p = harness.Param("x", 3)
    assert p.parse("7") == 7
    with pytest.raises(ValueError):
        p.parse("7.5")
    assert harness.Param("f", 0.5).parse("2") == 2.0
    assert harness.Param("b", True).parse("No") is False
    assert harness.Param("s", "error").parse("target") == "target"
    params = harness.Params({"n": 3, "on": True})
    assert params.get_int("n") == 3 and params.get_bool("on") and params.get_string("on") == "true"
    with pytest.raises(KeyError, match="unknown parameter"):
        params["missing"]


def make_dataset(root, name="tiny"):
    directory = root / name
    directory.mkdir(parents=True)
    images = np.arange(4 * 1 * 2 * 3, dtype=np.uint8).reshape(4, 1, 2, 3) * 10
    np.save(directory / "train_images.npy", images)
    np.save(directory / "train_labels.npy", np.array([0, 1, 1, 0], dtype=np.int32))
    (directory / "meta.json").write_text(json.dumps({
        "name": name, "classes": ["a", "b"], "shape": {"channels": 1, "height": 2, "width": 3},
        "splits": {"train": 4}}))
    return images


def test_datasets_load(tmp_path):
    images = make_dataset(tmp_path)
    split = datasets.load("tiny", "train", data_dir=tmp_path)
    assert len(split) == 4 and split.shape == (1, 2, 3)
    assert split.images.dtype == np.float32
    assert np.allclose(split.images, images / 255.0)
    assert split.labels.tolist() == [0, 1, 1, 0] and split.classes == ["a", "b"]
    assert datasets.load("tiny", data_dir=tmp_path, scale=False, limit=2).images.dtype == np.uint8
    assert len(datasets.load("tiny", data_dir=tmp_path, limit=2)) == 2
    assert datasets.available(tmp_path) == ["tiny"]
    with pytest.raises(FileNotFoundError, match="fetch.py get missing"):
        datasets.load("missing", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no split 'test'"):
        datasets.load("tiny", "test", data_dir=tmp_path)


def test_dataset_images_feed_a_network(tmp_path):
    make_dataset(tmp_path)
    split = datasets.load("tiny", data_dir=tmp_path)
    net = exr.Network()
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(exr.Shape(*split.shape)), False, False))
    net.add_inputs(eye, exr.Shape(*split.shape))
    net.add_output(eye)
    net.set_inputs(split.images[1])
    net.step()
    assert np.array_equal(net.layer_output(eye), split.images[1])


def test_apply_reward_to_one_layer():
    exr.reseed(2)
    net = exr.Network()
    inp = net.add_layer("in", exr.LayerSpec.dense(4, False, False))
    a = net.add_layer("a", exr.LayerSpec.dense(1, False, False))
    b = net.add_layer("b", exr.LayerSpec.dense(1, False, False))
    net.add_inputs(inp, 2)
    for out in (a, b):
        net.connect(inp, out)
        net.add_output(out)
    before = {layer: net.weights(layer, 0) for layer in (inp, a, b)}
    net.set_inputs([1.0, -1.0])
    net.step()
    net.apply_reward_to(a, 1.0, 0.1)
    assert not np.array_equal(net.weights(a, 0), before[a])
    assert np.array_equal(net.weights(b, 0), before[b])
    assert np.array_equal(net.weights(inp, 0), before[inp])
    net.freeze(b)
    net.apply_reward_to(b, 1.0, 0.1)  # frozen: nothing
    assert np.array_equal(net.weights(b, 0), before[b])


def test_build_info_and_threads():
    info = exr.build_info()
    assert set(info) == {"compiler", "build", "native", "openmp"}
    count = exr.threads()
    exr.set_threads(1)
    assert exr.threads() == 1 or not info["openmp"]
    exr.set_threads(count)
    with pytest.raises(ValueError):
        exr.set_threads(0)
