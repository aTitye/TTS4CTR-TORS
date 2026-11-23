import sys, os
import torch
import pickle
import numpy as np
import modules.models as models

# ---------------------------------------------------------------------------
# Data loader selection (see data/tfrecord_lite_loader.py):
#   DATALOADER=lite  -> pure-python TFRecord parser, no TensorFlow needed
#   DATALOADER=tf    -> original tf.data-based loader (requires TensorFlow)
#   DATALOADER=auto  (default) -> tf.data if importable, else lite fallback
# Both loaders expose the same CriteoLoader/Avazuloader/KDD12loader classes
# with an identical get_data(split_prefix, batch_size) interface.
# ---------------------------------------------------------------------------
_mode = os.environ.get("DATALOADER", "auto").lower()
if _mode == "lite":
    from data import tfrecord_lite_loader as tfloader
elif _mode == "tf":
    from data import tfloader
else:
    try:
        from data import tfloader
    except Exception:  # TensorFlow not installed (or broken CUDA build)
        print("[trainUtils] TensorFlow unavailable (DATALOADER=auto); falling back to "
              "the pure-python TFRecord loader. Set DATALOADER=lite to silence this.",
              file=sys.stderr)
        from data import tfrecord_lite_loader as tfloader


def getModel(model: str, opt):
    return models.build_model(model, opt)


class SyntheticTrajectoryLoader(object):
    """Small deterministic fixture for trajectory tests."""
    DATASET = "synthetic"
    FIELDS = 4

    _SPLIT_N = {"train": 80, "validation": 32, "val": 32, "test": 32}

    def __init__(self, path):
        self.path = str(path)

    def _load_arrays(self, data_type):
        n = self._SPLIT_N[data_type]
        # Different fixed seeds make each split identifiable while keeping the
        # fixture reproducible across processes and resume boundaries.
        seed = {"train": 20261001, "validation": 20261002, "val": 20261002, "test": 20261003}[data_type]
        rng = np.random.default_rng(seed)
        X = rng.integers(1, 97, size=(n, self.FIELDS), dtype=np.int64)
        y = np.arange(n, dtype=np.float32) % 2
        return X, y

    def get_data(self, data_type, batch_size=1):
        X, y = self._load_arrays(data_type)
        for s in range(0, len(X), batch_size):
            yield torch.from_numpy(X[s:s + batch_size].copy()), torch.as_tensor(
                y[s:s + batch_size], dtype=torch.float32
            ).unsqueeze(1)


def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))


def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def getDataLoader(dataset: str, path):
    dataset = dataset.lower()
    if dataset == 'criteo':
        return tfloader.CriteoLoader(path)
    elif dataset == 'avazu':
        return tfloader.Avazuloader(path)
    elif dataset == 'kdd12':
        return tfloader.KDD12loader(path)
    elif dataset == 'synthetic':
        return SyntheticTrajectoryLoader(path)


def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    offset_path = os.path.join(path + "/offset.pkl")
    with open(offset_path, 'rb') as fi:
        offset = pickle.load(fi)
    # return [i+1 for i in list(defaults.values())]
    return list(defaults.values()), list(offset.values())


def format_count(n):
    """Format a count with K/M/B suffix and 4 significant digits (e.g. 2.1025 M)."""
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return "{:.4f} {}".format(n / scale, suffix)
    return str(int(n))


def compute_model_complexity(model, input_tensor):
    """Compute (flops, macs, params) for one forward pass, PyTorch-native.

    Replaces the third-party `calflops` package:
      * FLOPs are counted by torch.utils.flop_counter.FlopCounterMode (built-in
        since torch 2.1), which intercepts aten ops via TorchDispatchMode.
        Matmul/conv-dominated ops are counted (1 multiply-add = 2 FLOPs);
        embedding lookups and element-wise ops are not, matching the previous
        calflops accounting used to produce logs/model_info/*_model.csv.
      * MACs = FLOPs / 2.
      * Params = sum(p.numel()).

    Returns raw ints/floats; use format_count() for human-readable strings.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    try:
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(input_tensor)
    finally:
        if was_training:
            model.train()
    flops = counter.get_total_flops()
    macs = flops / 2.0
    params = sum(p.numel() for p in model.parameters())
    return flops, macs, params
