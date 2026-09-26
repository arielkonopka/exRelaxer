#!/usr/bin/env python3
"""Fetches datasets for nntest experiments and prepares them in one format.

  fetch.py list                          the known datasets
  fetch.py get NAME... [options]         download (and prepare) datasets
  fetch.py kaggle OWNER/SLUG [options]   any Kaggle dataset; --image-root prepares an image folder
  fetch.py info NAME                     what a prepared dataset contains

Datasets go to NNtesting/data/NAME/ (or $EXR_DATA/NAME, or --data DIR/NAME):
  raw/                                   downloads and their extracted contents
  SPLIT_images.npy, SPLIT_labels.npy     prepared images (uint8 N x C x H x W) and labels
  meta.json                              classes, shape, splits, source, licence, hashes

Options for image datasets: --size N or WxH (resize), --grey / --colour.
Kaggle: credentials from KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json;
public datasets also download without them.

Needs numpy, and Pillow for image-folder datasets (see requirements.txt).
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download  # noqa: E402
import formats  # noqa: E402


@dataclass
class Dataset:
    name: str
    description: str
    source: str       # page to cite / read the terms on
    licence: str
    files: list = field(default_factory=list)  # (url, file name, sha256 or None)
    kaggle: str = ""  # OWNER/SLUG instead of files
    prepare: object = None  # prepare(dataset, raw_dir, dest, options) or None: download only
    kind: str = "image"
    classes: list = field(default_factory=list)  # label order, when the source files do not name them


# --- preparation --------------------------------------------------------------

def prepare_idx(ds, raw, dest, options):
    splits = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        images = formats.read_idx(raw / f"{prefix}-images-idx3-ubyte.gz")[:, None]  # N x 1 x 28 x 28
        labels = formats.read_idx(raw / f"{prefix}-labels-idx1-ubyte.gz").astype("int32")
        splits[split] = (images, labels)
    return splits, ds.classes


def prepare_cifar10(ds, raw, dest, options):
    root = download.extract(raw / "cifar-10-binary.tar.gz", raw / "extracted") / "cifar-10-batches-bin"
    classes = [c for c in (root / "batches.meta.txt").read_text().split() if c]
    train = formats.read_cifar_binary([root / f"data_batch_{i}.bin" for i in range(1, 6)])
    test = formats.read_cifar_binary([root / "test_batch.bin"])
    return {"train": train, "test": test}, classes


def find_split_root(root):
    """The directory holding train/ and test/ (or val/) subdirectories, else None."""
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        names = {d.name.lower() for d in candidate.iterdir() if d.is_dir()}
        if "train" in names and names & {"test", "val", "valid", "validation"}:
            return candidate
    return None


def prepare_image_folders(root, options, classes=None):
    """Splits from train/ + test/ (val/) subdirectories of `root`, or one split "all"."""
    size, grey = options.get("size"), options.get("grey", False)
    split_root = find_split_root(root)
    if split_root is None:
        images, labels, classes = formats.read_image_folder(root, classes, size, grey)
        return {"all": (images, labels)}, classes
    splits = {}
    for d in sorted(p for p in split_root.iterdir() if p.is_dir()):
        split = {"valid": "val", "validation": "val"}.get(d.name.lower(), d.name.lower())
        images, labels, found = formats.read_image_folder(d, classes, size, grey)
        classes = classes or found
        splits[split] = (images, labels)
    return splits, classes


def prepare_casting(ds, raw, dest, options):
    extracted = download.extract(raw / "archive.zip", raw / "extracted")
    roots = sorted(p for p in extracted.rglob("casting_data") if (p / "train").is_dir())
    if not roots:
        raise RuntimeError(f"{extracted}: no casting_data/train directory in the archive")
    options = {"grey": True, **options}
    return prepare_image_folders(roots[0], options, classes=["ok_front", "def_front"])


def extract_only(ds, raw, dest, options):
    for _, name, _ in ds.files:
        download.extract(raw / name, raw / "extracted")
    return None, None


# --- registry -------------------------------------------------------------------

# File name -> SHA-256, so a corrupt or changed download is detected.
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
FASHION_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}

DATASETS = {d.name: d for d in [
    Dataset(
        "mnist", "handwritten digits, 28 x 28 grey, 10 classes, 60 000 train / 10 000 test",
        "http://yann.lecun.com/exdb/mnist/ (mirror: storage.googleapis.com/cvdf-datasets)", "CC BY-SA 3.0",
        files=[(f"https://storage.googleapis.com/cvdf-datasets/mnist/{f}", f, h) for f, h in MNIST_FILES.items()],
        prepare=prepare_idx, classes=[str(d) for d in range(10)]),
    Dataset(
        "fashion_mnist", "clothing items, 28 x 28 grey, 10 classes, 60 000 train / 10 000 test",
        "https://github.com/zalandoresearch/fashion-mnist", "MIT",
        files=[(f"https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/{f}", f, h)
               for f, h in FASHION_MNIST_FILES.items()],
        prepare=prepare_idx, classes=["t-shirt/top", "trouser", "pullover", "dress", "coat",
                                      "sandal", "shirt", "sneaker", "bag", "ankle boot"]),
    Dataset(
        "cifar10", "small colour photos, 32 x 32 RGB, 10 classes, 50 000 train / 10 000 test",
        "https://www.cs.toronto.edu/~kriz/cifar.html", "no licence stated; cite Krizhevsky 2009",
        files=[("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "cifar-10-binary.tar.gz", None)],
        prepare=prepare_cifar10),
    Dataset(
        "casting", "industrial quality inspection: cast impellers, ok vs defective, 300 x 300 grey (Kaggle)",
        "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        "see the Kaggle page", kaggle="ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        prepare=prepare_casting),
    Dataset(
        "speech_commands", "one-second spoken words, 16 kHz WAV, 35 words (download only, for audio work)",
        "https://arxiv.org/abs/1804.03209", "CC BY 4.0",
        files=[("https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                "speech_commands_v0.02.tar.gz", None)],
        prepare=extract_only, kind="audio"),
    Dataset(
        "esc50", "environmental sounds, 5 s clips, 50 classes (download only, for audio work)",
        "https://github.com/karolpiczak/ESC-50", "CC BY-NC 3.0",
        files=[("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", "ESC-50-master.zip", None)],
        prepare=extract_only, kind="audio"),
]}


# --- commands --------------------------------------------------------------------

def data_root(args):
    if args.data:
        return Path(args.data)
    if os.environ.get("EXR_DATA"):
        return Path(os.environ["EXR_DATA"])
    return Path(__file__).resolve().parent.parent / "data"


def parse_size(text):
    if text is None:
        return None
    parts = text.lower().split("x")
    try:
        w, h = (int(parts[0]), int(parts[0])) if len(parts) == 1 else (int(parts[0]), int(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be N or WxH, got '{text}'")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("size must be positive")
    return (w, h)


def image_options(args):
    options = {}
    if args.size:
        options["size"] = args.size
    if args.grey is not None:
        options["grey"] = args.grey
    return options


def fetch(ds, root, options, force):
    dest = root / ds.name
    raw = dest / "raw"
    print(f"{ds.name}: {ds.description}")
    print(f"  source: {ds.source} (licence: {ds.licence})")
    hashes = {}
    if ds.kaggle:
        hashes["archive.zip"] = download.download_kaggle(ds.kaggle, raw / "archive.zip", force=force)
    for url, name, expected in ds.files:
        hashes[name] = download.download(url, raw / name, expected_sha256=expected, force=force)
    if ds.prepare is None:
        return
    splits, classes = ds.prepare(ds, raw, dest, options)
    if splits is None:
        print(f"  downloaded and extracted to {raw}")
        return
    formats.write_prepared(dest, ds.name, splits, classes, {
        "description": ds.description, "source": ds.source, "licence": ds.licence,
        "raw_files": hashes, "options": {k: list(v) if isinstance(v, tuple) else v for k, v in options.items()},
    })
    print(f"  prepared in {dest}")


def cmd_list(args):
    width = max(len(n) for n in DATASETS)
    for ds in DATASETS.values():
        print(f"{ds.name:<{width}}  {ds.kind:<5}  {ds.description}")
    print(f"\ndata directory: {data_root(args)}")


def cmd_get(args):
    root = data_root(args)
    failed = False
    for name in args.names:
        if name not in DATASETS:
            sys.exit(f"fetch.py: unknown dataset '{name}' (see fetch.py list)")
    for name in args.names:
        try:
            fetch(DATASETS[name], root, image_options(args), args.force)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed = True
    sys.exit(1 if failed else 0)


def cmd_kaggle(args):
    root = data_root(args)
    name = args.name or args.dataset.split("/")[1].replace("-", "_")
    dest = root / name
    raw = dest / "raw"
    try:
        digest = download.download_kaggle(args.dataset, raw / "archive.zip", force=args.force)
        extracted = download.extract(raw / "archive.zip", raw / "extracted")
        if not args.image_root:
            print(f"  extracted to {extracted}; add --image-root DIR (inside it) to prepare an image folder")
            return
        splits, classes = prepare_image_folders(extracted / args.image_root, image_options(args))
        formats.write_prepared(dest, name, splits, classes, {
            "description": f"Kaggle dataset {args.dataset}",
            "source": f"https://www.kaggle.com/datasets/{args.dataset}", "licence": "see the Kaggle page",
            "raw_files": {"archive.zip": digest}, "options": {"image_root": args.image_root, **{
                k: list(v) if isinstance(v, tuple) else v for k, v in image_options(args).items()}},
        })
        print(f"  prepared in {dest}")
    except (RuntimeError, ValueError, OSError) as e:
        sys.exit(f"FAILED: {e}")


def cmd_info(args):
    meta_path = data_root(args) / args.name / "meta.json"
    if not meta_path.exists():
        sys.exit(f"fetch.py: {args.name} is not prepared ({meta_path} missing)")
    meta = json.loads(meta_path.read_text())
    s = meta["shape"]
    print(f"{meta['name']}: {meta.get('description', '')}")
    print(f"  shape {s['channels']} x {s['height']} x {s['width']}, {len(meta['classes'])} classes: "
          f"{', '.join(meta['classes'][:12])}{' ...' if len(meta['classes']) > 12 else ''}")
    print(f"  splits: {', '.join(f'{k} {v}' for k, v in meta['splits'].items())}")
    print(f"  source: {meta.get('source')} (licence: {meta.get('licence')})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="data directory (default $EXR_DATA or NNtesting/data)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="the known datasets").set_defaults(func=cmd_list)

    def add_image_options(p):
        p.add_argument("--size", type=parse_size, help="resize images to N x N or W x H (WxH)")
        colour = p.add_mutually_exclusive_group()
        colour.add_argument("--grey", dest="grey", action="store_true", default=None, help="one grey channel")
        colour.add_argument("--colour", dest="grey", action="store_false", help="RGB channels")
        p.add_argument("--force", action="store_true", help="download again")

    get = sub.add_parser("get", help="download and prepare datasets")
    get.add_argument("names", nargs="+")
    add_image_options(get)
    get.set_defaults(func=cmd_get)

    kaggle = sub.add_parser("kaggle", help="any Kaggle dataset")
    kaggle.add_argument("dataset", help="OWNER/SLUG, as in the dataset's URL")
    kaggle.add_argument("--name", help="local name (default: the slug)")
    kaggle.add_argument("--image-root", help="directory inside the archive with class folders "
                                            "(or train/ and test/ folders of class folders)")
    add_image_options(kaggle)
    kaggle.set_defaults(func=cmd_kaggle)

    info = sub.add_parser("info", help="what a prepared dataset contains")
    info.add_argument("name")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress in order when piped or logged
    args.func(args)


if __name__ == "__main__":
    main()
