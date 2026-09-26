"""Downloading: plain URLs and Kaggle datasets, with progress and SHA-256.

Standard library only.
"""
import base64
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(name, done, total, started):
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = done / elapsed / 1e6
    if total:
        text = f"\r  {name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%), {rate:.1f} MB/s"
    else:
        text = f"\r  {name}: {done / 1e6:8.1f} MB, {rate:.1f} MB/s"
    sys.stderr.write(text)
    sys.stderr.flush()


def download(url, dest, *, headers=None, expected_sha256=None, force=False):
    """Downloads `url` to `dest` (a .part file renamed when complete).

    An existing `dest` is kept unless `force`; when `expected_sha256` is
    given, it is checked either way, and a mismatch raises ValueError.
    Returns the file's SHA-256.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  {dest.name}: already downloaded")
    else:
        part = dest.with_name(dest.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "exRelaxer-datasets/1.0", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done, started = 0, time.monotonic()
                for block in iter(lambda: response.read(CHUNK), b""):
                    out.write(block)
                    done += len(block)
                    _progress(dest.name, done, total, started)
            sys.stderr.write("\n")
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{url}: HTTP {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{url}: {e}") from e
        part.replace(dest)
    digest = sha256(dest)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{dest}: SHA-256 {digest}, expected {expected_sha256} "
                         f"(corrupt download or changed source; delete it and fetch again)")
    return digest


def kaggle_credentials():
    """(username, key) from KAGGLE_USERNAME / KAGGLE_KEY or kaggle.json, else None.

    kaggle.json is looked for in $KAGGLE_CONFIG_DIR, then ~/.kaggle (create it
    on kaggle.com: Settings -> API -> Create New Token).
    """
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if config.exists():
        data = json.loads(config.read_text())
        return data["username"], data["key"]
    return None


def download_kaggle(dataset, dest, *, force=False):
    """Downloads Kaggle dataset `owner/slug` as a zip to `dest`.

    Uses Kaggle credentials when available; public datasets also download
    without them. Datasets with rules must have them accepted on the
    dataset's page first.
    """
    if dataset.count("/") != 1:
        raise ValueError(f"Kaggle dataset must be OWNER/SLUG, got '{dataset}'")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    headers = {}
    credentials = kaggle_credentials()
    if credentials:
        token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    try:
        return download(url, dest, headers=headers, force=force)
    except RuntimeError as e:
        hint = ("" if credentials else
                " - no Kaggle credentials found: set KAGGLE_USERNAME and KAGGLE_KEY or create ~/.kaggle/kaggle.json;")
        raise RuntimeError(f"{e}{hint} datasets with rules must be accepted on "
                           f"https://www.kaggle.com/datasets/{dataset} first") from e


def extract(archive, dest):
    """Unpacks a .zip / .tar(.gz) archive into `dest` (once)."""
    archive, dest = Path(archive), Path(dest)
    marker = dest / ".extracted"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name}")
    kwargs = {"filter": "data"} if archive.suffix != ".zip" else {}  # tar: refuse links / absolute paths
    shutil.unpack_archive(str(archive), str(dest), **kwargs)
    marker.touch()
    return dest
