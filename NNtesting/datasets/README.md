# Datasets

`fetch.py` downloads public datasets for the experiments and prepares them
in one format. It needs Python 3.9+ with numpy, and Pillow for datasets made
of image files (`pip install -r NNtesting/requirements.txt`).

```bash
NNtesting/datasets/fetch.py list
NNtesting/datasets/fetch.py get mnist fashion_mnist
NNtesting/datasets/fetch.py get casting --size 150        # Kaggle; resize 300 x 300 -> 150 x 150
NNtesting/datasets/fetch.py kaggle OWNER/SLUG --image-root some/folder --size 64 --grey
NNtesting/datasets/fetch.py info mnist
```

| Name | Kind | Contents | Source (licence) |
|------|------|----------|------------------|
| `mnist` | image | handwritten digits, 28 × 28 grey, 10 classes, 60 000 / 10 000 | yann.lecun.com (CC BY-SA 3.0), via the Google mirror |
| `fashion_mnist` | image | clothing, 28 × 28 grey, 10 classes, 60 000 / 10 000 | Zalando Research on GitHub (MIT) |
| `cifar10` | image | photos, 32 × 32 RGB, 10 classes, 50 000 / 10 000 | University of Toronto (no licence stated; cite Krizhevsky 2009) |
| `casting` | image | industrial inspection: cast impellers, ok vs defective, 300 × 300 grey | Kaggle, ravirajsinh45 (see the dataset page) |
| `speech_commands` | audio | 1 s spoken words, 16 kHz WAV, 35 words; download and extract only | Google (CC BY 4.0) |
| `esc50` | audio | 5 s environmental sounds, 50 classes; download and extract only | K. Piczak on GitHub (CC BY-NC 3.0) |

Check each dataset's terms before using it beyond research.

## Where data goes

`NNtesting/data/NAME/` by default (ignored by git); `$EXR_DATA/NAME` when
`EXR_DATA` is set, or `--data DIR`:

```
NAME/
  raw/                  the downloads (kept; not fetched again unless --force)
  raw/extracted/        unpacked archives
  train_images.npy      prepared images
  train_labels.npy
  test_images.npy
  test_labels.npy
  meta.json
```

Downloads are written to a `.part` file and renamed when complete, and
their SHA-256 is recorded in `meta.json`.

## Prepared format

- `SPLIT_images.npy`: `uint8`, **N × C × H × W**, channel-major like
  `exr::Shape` (grey images have C = 1), pixel values 0…255. Networks
  should get them scaled to 0…1: neurons clamp at ±10.
- `SPLIT_labels.npy`: `int32`, N, indexes into `classes`.
- `meta.json`: `name`, `classes` (in label order), `shape`
  (`channels`, `height`, `width`), `splits` (image counts), `source`,
  `licence`, `raw_files` (SHA-256 per download), the options used, and the
  preparation time.

Splits are `train` and `test` (plus `val` when the source has one); an
image folder without train/test subfolders becomes one split, `all`.

## Kaggle

Public Kaggle datasets download without an account. For others, and for
datasets with rules (accept them on the dataset's page first), provide your
API token: `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`
(kaggle.com → Settings → API → Create New Token). No Kaggle package is
needed.

`fetch.py kaggle OWNER/SLUG` downloads and extracts any dataset (OWNER/SLUG
as in its URL). With `--image-root DIR` (a directory inside the archive)
it prepares an image folder: class subfolders, optionally under `train/`
and `test/` (or `val/`) folders. `--size N` or `--size WxH` resizes (with
Lanczos filtering); `--grey` or `--colour` sets the channels. Without
`--size`, all images must have the same size.

## Adding a dataset

Add a `Dataset(...)` to `DATASETS` in `fetch.py`: a name, a description, the
source page and licence, and either `files` (URL, file name, SHA-256) or
`kaggle` (OWNER/SLUG), plus a `prepare` function returning
`({split: (images, labels)}, classes)`. `formats.py` reads IDX (MNIST),
CIFAR binary and image folders.
