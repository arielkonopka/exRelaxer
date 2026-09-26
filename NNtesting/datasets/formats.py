"""Reading source formats and writing the common prepared format.

Prepared format, in one directory per dataset (see README.md):
  SPLIT_images.npy   uint8, N x C x H x W (channel-major, like exr::Shape)
  SPLIT_labels.npy   int32, N
  meta.json          name, classes, shape, splits, source, licence, file hashes
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def read_idx(path):
    """An IDX file (MNIST, Fashion-MNIST), optionally gzipped, as a numpy array."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    if data[0] != 0 or data[1] != 0:
        raise ValueError(f"{path}: not an IDX file")
    types = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}
    dtype, dims = np.dtype(types[data[2]]), data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(dims))
    array = np.frombuffer(data, dtype=dtype, offset=4 + 4 * dims)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {array.size} values, header says {shape}")
    return array.reshape(shape)


def read_cifar_binary(paths, label_bytes=1):
    """CIFAR binary batches: each record is the label byte(s), then 3 x 32 x 32 uint8."""
    records = np.concatenate([np.fromfile(p, dtype=np.uint8) for p in paths])
    record = label_bytes + 3 * 32 * 32
    if records.size % record:
        raise ValueError(f"{paths}: not whole CIFAR records")
    records = records.reshape(-1, record)
    return records[:, label_bytes:].reshape(-1, 3, 32, 32), records[:, label_bytes - 1].astype(np.int32)


def read_image_folder(root, classes=None, size=None, grey=False):
    """Images in class subdirectories of `root` (root/CLASS/image.png).

    `classes`: class names in label order (default: the sorted subdirectory
    names). `size`: (width, height) to resize to; default: the images' own
    size, which must then be the same for all. Returns (images N x C x H x W
    uint8, labels int32, classes).
    """
    from PIL import Image  # only needed for image datasets

    root = Path(root)
    if classes is None:
        classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    files, labels = [], []
    for label, name in enumerate(classes):
        found = sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        files += found
        labels += [label] * len(found)
    if not files:
        raise ValueError(f"{root}: no images in class subdirectories")
    images = None
    for i, path in enumerate(files):
        with Image.open(path) as img:
            img = img.convert("L" if grey else "RGB")
            if size is not None and img.size != tuple(size):
                img = img.resize(tuple(size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.uint8)
        pixels = pixels[None] if pixels.ndim == 2 else pixels.transpose(2, 0, 1)  # C x H x W
        if images is None:
            images = np.empty((len(files),) + pixels.shape, dtype=np.uint8)
        elif pixels.shape != images.shape[1:]:
            raise ValueError(f"{path}: {pixels.shape[2]} x {pixels.shape[1]} differs from the first image; "
                             f"pass a size to resize")
        images[i] = pixels
        if (i + 1) % 1000 == 0:
            print(f"  {root.name}: {i + 1} / {len(files)} images")
    return images, np.asarray(labels, dtype=np.int32), list(classes)


def write_prepared(dest, name, splits, classes, info):
    """Writes SPLIT_images.npy / SPLIT_labels.npy per split and meta.json.

    `splits`: {split: (images N x C x H x W uint8, labels int32)}.
    `info`: extra fields for meta.json (description, source, licence, files...).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shape = None
    counts = {}
    for split, (images, labels) in splits.items():
        images = np.ascontiguousarray(images, dtype=np.uint8)
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        if images.ndim != 4 or len(images) != len(labels):
            raise ValueError(f"{split}: images must be N x C x H x W with one label each")
        if shape is not None and images.shape[1:] != shape:
            raise ValueError(f"{split}: shape {images.shape[1:]} differs from {shape}")
        shape = images.shape[1:]
        np.save(dest / f"{split}_images.npy", images)
        np.save(dest / f"{split}_labels.npy", labels)
        counts[split] = int(len(images))
        print(f"  {split}: {len(images)} images, {shape[0]} x {shape[1]} x {shape[2]}")
    meta = {
        "name": name,
        "format": "exRelaxer prepared dataset 1: SPLIT_images.npy uint8 N x C x H x W, SPLIT_labels.npy int32",
        "classes": classes,
        "shape": {"channels": int(shape[0]), "height": int(shape[1]), "width": int(shape[2])},
        "splits": counts,
        "prepared": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **info,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
