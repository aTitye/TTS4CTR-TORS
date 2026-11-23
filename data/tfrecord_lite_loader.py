"""Pure-python TFRecord loader — drop-in replacement for data/tfloader.py.

Removes the TensorFlow dependency from the train/eval pipeline: records are
parsed by the bit-exact pure-python parser in tools/tors_merge/tfrecord_lite.py
(verified against the official prediction bank with max|dp| = 1.2e-7).

Selection
---------
Set the environment variable ``DATALOADER=lite`` to use this loader, or rely on
the automatic fallback in utils/trainUtils.py when TensorFlow is not installed.

Behaviour parity with data/tfloader.py
--------------------------------------
* ``get_data(data_type, batch_size)`` yields ``(feature, label)`` where
  ``feature`` is an int64 tensor of shape ``(B, F)`` and ``label`` a float32
  tensor of shape ``(B, 1)`` (both on CPU; callers move them to device).
* ``data_type`` is a split *prefix* matched against ``{prefix}*`` files, so
  ``"val"`` picks up ``validation_*.tfrecord`` exactly like the tf.data glob.
* Files are consumed in sorted order and records in file order (the original
  tf.data pipeline did not shuffle either); the trailing partial batch is
  kept, matching ``tf.data.Dataset.batch()``'s default.

Speed
-----
If a cached ``{dataset}_{split}.npz`` exists under ``results/tors_merge/cache/``
(same format as produced by tools/tors_merge/build_cache.py), it is loaded to
memory instead of re-parsing the TFRecords (~40k rec/s single-threaded).
Build caches once with:

    python data/tfrecord_lite_loader.py avazu /path/to/avazu/record --split train

or by running training with ``LITE_BUILD_CACHE=1``.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.tfrecord_lite import iter_tfrecord, parse_example  # noqa: E402

# Shared with tools/tors_merge/build_cache.py; regenerable and git-ignored.
_DEFAULT_CACHE_DIR = os.path.join(
    _REPO_ROOT,
    os.environ.get("TTS4CTR_CACHE_DIR", os.path.join("outputs", "cache")),
)

# Split-prefix canonicalisation ({"val": "validation", ...}) for cache lookup.
_SPLIT_CANON = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


class LiteTFRecordLoader(object):
    """Base class; subclasses set DATASET and FIELDS."""

    DATASET = None
    FIELDS = None

    def __init__(self, tfrecord_path, cache_dir=None):
        self.tfrecord_path = tfrecord_path
        self.cache_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._arrays = {}  # memoised {split_canon: (X, y)} per loader instance

    # ------------------------------------------------------------------ #
    # public API — mirrors data/tfloader.py                              #
    # ------------------------------------------------------------------ #
    def get_data(self, data_type, batch_size=1):
        X, y = self._load_arrays(data_type)
        n = X.shape[0]
        for s in range(0, n, batch_size):
            xb = torch.from_numpy(X[s:s + batch_size].astype(np.int64))
            yb = torch.as_tensor(y[s:s + batch_size], dtype=torch.float32).unsqueeze(1)
            yield xb, yb

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #
    def _files(self, data_type):
        files = sorted(glob.glob(os.path.join(self.tfrecord_path, data_type + "*")))
        if not files:
            raise FileNotFoundError(
                "no tfrecord matches {} in {}".format(data_type + "*", self.tfrecord_path))
        return files

    def _iter_records(self, data_type):
        """Yield (feature int64[F], label float) in sorted-file/record order."""
        for fp in self._files(data_type):
            for payload in iter_tfrecord(fp):
                yield parse_example(payload, self.FIELDS)

    def _load_arrays(self, data_type):
        canon = _SPLIT_CANON.get(data_type, data_type)
        if canon in self._arrays:
            return self._arrays[canon]

        npz_path = os.path.join(self.cache_dir, "{}_{}.npz".format(self.DATASET, canon))
        if os.path.exists(npz_path):
            with np.load(npz_path) as d:
                X, y = d["X"], d["y"]
            print("[tfrecord_lite_loader] using cache {}".format(
                os.path.relpath(npz_path, _REPO_ROOT)))
        else:
            X, y = self._parse_and_maybe_cache(data_type, canon, npz_path)
        self._arrays[canon] = (X, y)
        return X, y

    def _parse_and_maybe_cache(self, data_type, canon, npz_path, write_cache=None):
        t0 = time.time()
        xs, ys, n = [], [], 0
        for feat, label in self._iter_records(data_type):
            xs.append(feat)
            ys.append(label)
            n += 1
            if n % 500000 == 0:
                print("[tfrecord_lite_loader] parsed {} records, {:.0f}s".format(
                    n, time.time() - t0))
        if n == 0:
            raise RuntimeError("no records parsed for {} / {}".format(self.DATASET, data_type))
        print("[tfrecord_lite_loader] streamed {} records from tfrecord in {:.0f}s".format(
            n, time.time() - t0))

        X = np.stack(xs).astype(np.int32)
        y = np.array(ys, dtype=np.float32)
        if write_cache is None:
            write_cache = os.environ.get("LITE_BUILD_CACHE", "").lower() in ("1", "true", "yes")
        if write_cache:
            self._write_cache(X, y, canon, npz_path)
        return X, y

    def _write_cache(self, X, y, canon, npz_path):
        files = self._files(canon)
        meta = {
            "dataset": self.DATASET,
            "split": canon,
            "files": [os.path.basename(f) for f in files],
            "file_md5": [hashlib.md5(open(f, "rb").read(1 << 20)).hexdigest() for f in files],
            "n": int(X.shape[0]),
            "fields": self.FIELDS,
            "max_feature_id": int(X.max()),
            "label_mean": float(y.mean()),
            "written_by": "data/tfrecord_lite_loader.py",
        }
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, X=X, y=y)
        with open(npz_path + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print("[tfrecord_lite_loader] wrote cache {}".format(
            os.path.relpath(npz_path, _REPO_ROOT)))


class CriteoLoader(LiteTFRecordLoader):
    DATASET = "criteo"
    FIELDS = 39


class Avazuloader(LiteTFRecordLoader):
    DATASET = "avazu"
    FIELDS = 24


class KDD12loader(LiteTFRecordLoader):
    DATASET = "kdd12"
    FIELDS = 11


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Build npz cache for the lite TFRecord loader")
    ap.add_argument("dataset", choices=["avazu", "criteo", "kdd12"])
    ap.add_argument("data_dir", help="directory containing {split}_*.tfrecord files")
    ap.add_argument("--split", default="train", help="train / validation / test")
    args = ap.parse_args()

    cls = {"avazu": Avazuloader, "criteo": CriteoLoader, "kdd12": KDD12loader}[args.dataset]
    loader = cls(args.data_dir)
    X, y = loader._parse_and_maybe_cache(
        args.split, args.split,
        os.path.join(loader.cache_dir, "{}_{}.npz".format(args.dataset, args.split)),
        write_cache=True)
    print("n={} fields={} max_feature_id={} label_mean={:.6f}".format(
        X.shape[0], X.shape[1], int(X.max()), float(y.mean())))


if __name__ == "__main__":
    _main()
