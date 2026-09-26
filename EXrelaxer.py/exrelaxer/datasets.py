"""Datasets prepared by NNtesting/datasets/fetch.py, as numpy arrays.

    from exrelaxer import datasets
    train = datasets.load("mnist", "train")          # images float32 N x C x H x W in 0..1
    train.images.shape, train.labels[:10], train.classes

fetch.py writes NAME/SPLIT_images.npy (uint8 N x C x H x W), SPLIT_labels.npy
(int32) and meta.json under the data directory: --data / data_dir=, else
$EXR_DATA, else NNtesting/data of the repository (found from the current
directory upwards). A dataset that is not there yet raises FileNotFoundError
with the fetch.py command to run, or is fetched first with fetch=True.
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Split:
    name: str                 # dataset name
    split: str                # "train", "test", "val" or "all"
    images: np.ndarray        # N x C x H x W: float32 in 0..1, or the stored uint8 with scale=False
    labels: np.ndarray        # int32, N, indexes into classes
    classes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.labels)

    @property
    def shape(self):
        """C x H x W of one image."""
        return tuple(self.images.shape[1:])


def find_nntesting(start=None):
    """The NNtesting directory of the repository containing `start` (default: the current directory), or None."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "NNtesting"
        if (candidate / "datasets" / "fetch.py").exists():
            return candidate
        if directory.name == "NNtesting" and (directory / "datasets" / "fetch.py").exists():
            return directory
    return None


def data_path(data_dir=None):
    """Where datasets are: `data_dir`, $EXR_DATA, or NNtesting/data of the enclosing repository."""
    if data_dir:
        return Path(data_dir)
    if os.environ.get("EXR_DATA"):
        return Path(os.environ["EXR_DATA"])
    nntesting = find_nntesting()
    if nntesting is None:
        raise FileNotFoundError("no data directory: pass data_dir=, set EXR_DATA, or run inside the repository")
    return nntesting / "data"


def available(data_dir=None):
    """Names of the prepared datasets."""
    root = data_path(data_dir)
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/meta.json"))


def fetch(name, data_dir=None):
    """Runs fetch.py get NAME (downloads and prepares the dataset)."""
    nntesting = find_nntesting()
    if nntesting is None:
        raise FileNotFoundError("fetch.py not found: run inside the repository")
    command = [sys.executable, str(nntesting / "datasets" / "fetch.py"), "--data", str(data_path(data_dir)),
               "get", name]
    subprocess.run(command, check=True)


def meta(name, data_dir=None):
    path = data_path(data_dir) / name / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset '{name}' is not prepared in {path.parent.parent}: "
                                f"run NNtesting/datasets/fetch.py get {name}")
    return json.loads(path.read_text())


def load(name, split="train", *, data_dir=None, scale=True, limit=None, fetch_missing=False):
    """One split of a prepared dataset.

    scale: images as float32 in 0..1 (networks clamp at +-10, so raw 0..255 is
    too strong); False keeps the stored uint8. limit: only the first `limit`
    images. fetch_missing: run fetch.py first when the dataset is not prepared.
    """
    root = data_path(data_dir)
    if fetch_missing and not (root / name / "meta.json").exists():
        fetch(name, root)
    info = meta(name, root)
    directory = root / name
    images_path = directory / f"{split}_images.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"dataset '{name}' has no split '{split}' (splits: {', '.join(info['splits'])})")
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(directory / f"{split}_labels.npy", mmap_mode="r")
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    images = images.astype(np.float32) / 255.0 if scale else np.array(images)
    return Split(name, split, images, np.array(labels, dtype=np.int32), list(info["classes"]), info)
