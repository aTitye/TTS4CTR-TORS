import torch
import argparse
import logging
import os, sys
from pathlib import Path
import hashlib
import json
import math
import random
import numpy as np
from sklearn import metrics
from utils import trainUtils
import time

parser = argparse.ArgumentParser(description="optfs trainer")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model", type=str, help="specify model")

# dataset information
parser.add_argument("--feature", type=int, help="feature number", required=True)
parser.add_argument("--field", type=int, help="field number", required=True)
parser.add_argument("--data_dir", type=str, help="data directory", required=True)

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=3e-5)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-3)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", help="mlp batch normalization")

# wukong specific parameters
parser.add_argument("--num_layers", type=int, default=3, help="number of interaction layers in Wukong")
parser.add_argument("--cross", type=int, default=3, help="number of cross layers in DCN/DCNv2")
parser.add_argument("--num_embed_lcb", type=int, default=16, help="number of embedding in Linear Compress Block in Wukong")
parser.add_argument(
    "--num_embed_fmb", type=int, default=16, help="number of embedding in Factorization Machine Block in Wukong"
)
parser.add_argument("--rank_fmb", type=int, default=24, help="rank in Wukong")
parser.add_argument("--fm_mlp_dims", type=int, nargs='+', default=[512, 512], help="sub-mlp within the FM Block in Wukong")

# Rankmixer specific parameters
parser.add_argument("--num_L", type=int, default=3, help="number of layers in RankMixer")
parser.add_argument("--expansion_rate", type=int, default=2, help="expansion rate in RankMixer")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")
parser.add_argument("--seed", type=int, default=2021, help="random seed")

# SwanLab configuration
parser.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")
parser.add_argument("--swanlab_project", type=str, default="Seed_TTS", help="SwanLab project name")
parser.add_argument("--swanlab_api_key", type=str, default=os.environ.get("SWANLAB_API_KEY", ""), help="SwanLab API key")
parser.add_argument("--tune_mode", action="store_true", help="Enable hyperparameter tuning mode (no checkpoint saving)")
parser.add_argument("--experiment_name", type=str, default=None, help="Custom experiment name for SwanLab")

# Trajectory-merge experiment flags (docs/paper/TORS_theory_traj_merge.md §4, all default off)
parser.add_argument("--trajectory_protocol", action="store_true", help="collect validation-only trajectory checkpoints; disables all in-training test evaluation")
parser.add_argument("--save_every_epoch", action="store_true", help="save a checkpoint at the end of every training epoch to {save_dir}.ep{e}")
parser.add_argument("--reset_emb_per_epoch", action="store_true", help="MEIA-style: re-initialize the shared embedding table at each epoch boundary (except the first); backbone and its optimizer state carry over")
parser.add_argument("--no_early_stop", action="store_true", help="disable val-AUC early stopping (needed to collect full overfitting trajectories)")
parser.add_argument("--trajectory_resume", type=Path, help="resume a trajectory run from its resume_last.pt package")
# t3-local-v1 (docs/analysis/theory_3_local_merge_generalization_plan_20261019.md §3):
# configurable first-epoch snapshot fractions. Default (empty) keeps the historical
# traj_v2.1 behaviour (0.90/0.95). Fraction 1.00 aliases the epoch_end snapshot.
parser.add_argument("--first_epoch_snapshot_fractions", type=float, nargs="*", default=[],
                    help="t3-local-v1: first-epoch snapshot fractions in (0,1]; e.g. 0.60 0.80 0.90 0.95 1.00; 1.00 aliases epoch_end; default keeps historical 0.90/0.95")

args = parser.parse_args()

my_seed = args.seed
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
# os.environ['NUMEXPR_NUM_THREADS'] = '8'
# os.environ['NUMEXPR_MAX_THREADS'] = '8'

os.environ['OMP_NUM_THREADS'] = '4'  # 控制 OpenMP
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # 控制 TensorFlow 算子内并行
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # 控制 TensorFlow 图级并行
os.environ['NUMEXPR_NUM_THREADS'] = '4'  # 与 OMP 一致
os.environ['NUMEXPR_MAX_THREADS'] = '4'  # 与 OMP 一致

# Enable XLA persistent cache to avoid repeated JIT compilation
# This caches compiled CUDA kernels to disk for reuse across runs
# XLA_CACHE_DIR = '/root/tf_xla_cache'
# os.makedirs(XLA_CACHE_DIR, exist_ok=True)
# os.environ[
#     'TF_XLA_FLAGS'
# ] = f'--tf_xla_enable_xla_devices --tf_xla_persistent_cache_directory={XLA_CACHE_DIR} --tf_xla_persistent_cache_read_only=0'

# Initialize SwanLab
if args.use_swanlab:
    import swanlab
    os.environ["SWANLAB_API_KEY"] = args.swanlab_api_key
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"{args.dataset}_{args.model}_seed{args.seed}"
    swanlab.init(project=args.swanlab_project, experiment_name=experiment_name, config=vars(args))

if args.trajectory_protocol:
    if args.tune_mode:
        parser.error("--trajectory_protocol cannot be combined with --tune_mode")
    if not args.save_every_epoch:
        parser.error("--trajectory_protocol requires --save_every_epoch")
    if not args.no_early_stop:
        parser.error("--trajectory_protocol requires --no_early_stop")
    if args.save_dir is None:
        parser.error("--trajectory_protocol requires --save_dir")
    if args.trajectory_resume is not None and not args.trajectory_resume.exists():
        parser.error("--trajectory_resume points to a missing checkpoint package")
    if args.first_epoch_snapshot_fractions:
        if not args.trajectory_protocol:
            parser.error("--first_epoch_snapshot_fractions requires --trajectory_protocol")
        fr = args.first_epoch_snapshot_fractions
        if any(not (0.0 < f <= 1.0) for f in fr):
            parser.error("--first_epoch_snapshot_fractions entries must lie in (0, 1]")
        if len(set(fr)) != len(fr):
            parser.error("--first_epoch_snapshot_fractions entries must be unique")
        if sorted(fr) != list(fr):
            parser.error("--first_epoch_snapshot_fractions must be given in ascending order")


class Trainer(object):

    def __init__(self, opt, tune_mode=False):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.tune_mode = tune_mode
        self.opt = opt  # Store opt for model profiling
        if not tune_mode:
            os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = trainUtils.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        # Trajectory checkpoint options.
        self.save_every_epoch = opt.get("save_every_epoch", False)
        self.reset_emb_per_epoch = opt.get("reset_emb_per_epoch", False)
        self.no_early_stop = opt.get("no_early_stop", False)
        self.trajectory_protocol = opt.get("trajectory_protocol", False)
        self.trajectory_resume = opt.get("trajectory_resume")
        self.first_epoch_snapshot_fractions = list(opt.get("first_epoch_snapshot_fractions") or [])
        # Custom first-epoch fractions use the t3-local-v1 protocol.
        self.trajectory_protocol_version = (
            "t3-local-v1" if self.first_epoch_snapshot_fractions else "traj_v2.1"
        )

        if self.trajectory_protocol:
            if self.tune_mode:
                raise ValueError("trajectory_protocol cannot be combined with tune_mode")
            if not self.save_every_epoch or not self.no_early_stop:
                raise ValueError("trajectory_protocol requires save_every_epoch and no_early_stop")
            self._trajectory_init_bookkeeping()

        self.best_epoch_idx = -1
        self.best_auc = 0.0
        self._resume_completed_epochs = 0
        if self.trajectory_protocol and self.trajectory_resume is not None:
            self._trajectory_load_resume()

        # Compute and log model statistics
        self.compute_model_stats()

    def compute_model_stats(self):
        """Compute and log model FLOPs, MACs, and parameters (PyTorch-native)."""
        model_opt = self.opt["model_opt"]
        cal_bs = 1  # Use a smaller batch size for profiling
        # Create a dummy input tensor for profiling
        input_tensor = torch.randint(0, model_opt["feat_num"], (cal_bs, model_opt["field_num"])).long().to(self.device)

        flops, macs, params = trainUtils.compute_model_complexity(self.network, input_tensor)
        flops = trainUtils.format_count(flops)
        macs = trainUtils.format_count(macs)
        params = trainUtils.format_count(params)

        print(f"\n{'='*60}")
        print(f"Model: {self.opt['model']}, Batch Size: {cal_bs}")
        print(f"Input Shape: ({cal_bs}, {model_opt['field_num']})")
        print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")
        print(f"{'='*60}\n")

        # Log model stats to SwanLab as an ECharts table
        if self.opt.get("use_swanlab", False):
            headers = ["Metric", "Value"]
            rows = [
                ["Model", self.opt['model']],
                ["Batch Size", str(cal_bs)],
                ["Input Shape", f"({cal_bs}, {model_opt['field_num']})"],
                ["FLOPs", flops],
                ["MACs", macs],
                ["Params", params],
            ]
            table = swanlab.echarts.Table()
            table.add(headers, rows)
            swanlab.log({"Model Summary": table})

    def _trajectory_init_bookkeeping(self):
        """Create the traj_v2 run contract without touching legacy behaviour."""
        self.run_root = Path(self.model_dir).parent
        self.init_path = self.run_root / "init.pt"
        self.resume_path = self.run_root / "resume_last.pt"
        self.metrics_path = self.run_root / "trajectory_metrics.jsonl"
        self.manifest_path = self.run_root / "run_manifest.json"

        self.trajectory_manifest = {
            "protocol_version": self.trajectory_protocol_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": self._trajectory_config_record(),
            "config_signature": self._trajectory_config_signature(),
            "code_fingerprints": self._trajectory_code_fingerprints(),
            "events": [],
            "snapshots": [],
            "best_epoch_idx": None,
        }

        if self.trajectory_resume is None:
            protected = [self.manifest_path, self.init_path, self.metrics_path, self.resume_path]
            protected += sorted(self.run_root.glob(Path(self.model_dir).name + ".ep*"))
            existing = [str(p) for p in protected if p.exists()]
            if existing:
                raise FileExistsError(
                    "trajectory output already exists; use a new save_dir. Existing: " + ", ".join(existing)
                )

    @staticmethod
    def _trajectory_code_fingerprints():
        files = [
            Path(__file__).resolve(),
            Path("utils/trainUtils.py").resolve(),
            Path("data/tfrecord_lite_loader.py").resolve(),
        ]
        return {
            str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.exists()
        }

    def _trajectory_config_record(self):
        keys = [
            "dataset", "model", "data_dir", "lr", "l2", "bsize", "optimizer",
            "model_opt", "seed", "trajectory_protocol", "save_every_epoch",
            "reset_emb_per_epoch", "no_early_stop", "first_epoch_snapshot_fractions",
        ]
        return {k: self.opt.get(k) for k in keys}

    def _trajectory_config_signature(self):
        payload = json.dumps(self._trajectory_config_record(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _tensor_sha256(tensor):
        h = hashlib.sha256()
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _large_embedding_parameter(self):
        feat_num = self.opt["model_opt"]["feat_num"]
        matches = [
            (name, param) for name, param in self.network.named_parameters()
            if name.endswith("embedding.embedding") and param.dim() == 2 and param.shape[0] == feat_num
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one large feature embedding, found {}: {}".format(
                    len(matches), [name for name, _ in matches]
                )
            )
        return matches[0]

    def _state_fingerprint(self, exclude_parameter=None):
        h = hashlib.sha256()
        excluded = set() if exclude_parameter is None else {id(exclude_parameter)}
        for name, param in self.network.named_parameters():
            if id(param) not in excluded:
                h.update(name.encode())
                h.update(self._tensor_sha256(param).encode())
        for name, buf in self.network.named_buffers():
            h.update(name.encode())
            h.update(self._tensor_sha256(buf).encode())
        return h.hexdigest()

    def _optimizer_fingerprint(self, exclude_parameter=None):
        h = hashlib.sha256()
        excluded = set() if exclude_parameter is None else {id(exclude_parameter)}
        for group_idx, group in enumerate(self.optim.param_groups):
            h.update("group:{}:lr:{}:weight_decay:{}:".format(
                group_idx, group.get("lr"), group.get("weight_decay")
            ).encode())
            for param in group["params"]:
                if id(param) in excluded:
                    continue
                state = self.optim.state.get(param, {})
                for state_key in sorted(state, key=str):
                    value = state[state_key]
                    if torch.is_tensor(value):
                        h.update(str(state_key).encode())
                        h.update(self._tensor_sha256(value).encode())
                    else:
                        h.update(repr((str(state_key), value)).encode())
        return h.hexdigest()

    def reset_embedding(self, epoch_idx):
        """Reset only the large feature table; leave backbone and its Adam state intact."""
        name, param = self._large_embedding_parameter()
        before_param_id = id(param)
        before_embedding = self._tensor_sha256(param)
        before_backbone = self._state_fingerprint(exclude_parameter=param)
        before_optimizer = self._optimizer_fingerprint(exclude_parameter=param)

        seed_sequence = np.random.SeedSequence([int(self.opt["seed"]), int(epoch_idx), 20261006])
        seed_int = int.from_bytes(seed_sequence.generate_state(2, dtype=np.uint32).tobytes(), "little")
        seed_int &= (1 << 63) - 1
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed_int)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(param.data, gain=1.0, generator=generator)
        self.optim.state[param] = {}

        after_param = dict(self.network.named_parameters())[name]
        event = {
            "event": "reset_embedding",
            "epoch_idx": int(epoch_idx),
            "epoch_number": int(epoch_idx + 1),
            "parameter": name,
            "shape": list(param.shape),
            "parameter_identity_preserved": id(after_param) == before_param_id,
            "reset_seed_int": int(seed_int),
            "generator_device": str(param.device),
            "torch_version": torch.__version__,
            "embedding_before_sha256": before_embedding,
            "embedding_after_sha256": self._tensor_sha256(param),
            "backbone_before_sha256": before_backbone,
            "backbone_after_sha256": self._state_fingerprint(exclude_parameter=param),
            "backbone_optimizer_before_sha256": before_optimizer,
            "backbone_optimizer_after_sha256": self._optimizer_fingerprint(exclude_parameter=param),
            "embedding_optimizer_state_keys_after": sorted(self.optim.state[param], key=str),
        }
        self._append_manifest_event(event)
        return event

    def _append_manifest_event(self, event):
        self.trajectory_manifest["events"].append(event)
        self._write_manifest()

    def _write_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.trajectory_manifest, f, indent=2, sort_keys=True, default=str)

    def _append_trajectory_metric(self, record):
        record = dict(record)
        record.setdefault("wall_time", time.time())
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _trajectory_train_count(self):
        if not hasattr(self.dataloader, "_load_arrays"):
            raise RuntimeError(
                "trajectory_protocol requires the deterministic lite loader (set DATALOADER=lite)"
            )
        X, _ = self.dataloader._load_arrays("train")
        return int(X.shape[0])

    def _trajectory_data_identity(self):
        identity = {
            "loader_class": type(self.dataloader).__name__,
            "data_dir": str(self.opt["data_dir"]),
            "train_samples": self._trajectory_train_count(),
        }
        dataset = getattr(self.dataloader, "DATASET", None)
        cache_dir = getattr(self.dataloader, "cache_dir", None)
        if dataset and cache_dir:
            metadata_path = Path(cache_dir) / f"{dataset}_train.npz.json"
            array_path = Path(cache_dir) / f"{dataset}_train.npz"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    identity["cache_metadata"] = json.load(f)
            if array_path.exists():
                identity["cache_array_bytes"] = int(array_path.stat().st_size)
        return identity

    @staticmethod
    def _trajectory_file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _save_trajectory_weights(
        self, filename, epoch_idx, step, kind, evaluate_after=False, kind_aliases=None
    ):
        path = self.run_root / filename
        if path.exists():
            raise FileExistsError("trajectory checkpoint already exists: {}".format(path))
        torch.save(self.network.state_dict(), path)
        j = getattr(self, "_trajectory_epoch_batches", None)
        n = getattr(self, "_trajectory_train_samples", None)
        if j is not None and n is not None:
            sample_count = int(min(step * self.bs, n)) if step < j else int(n)
            global_step = int(epoch_idx) * int(j) + int(step)
        else:
            sample_count = None
            global_step = None
        record = {
            "path": filename,
            "epoch_idx": int(epoch_idx),
            "epoch_number": int(epoch_idx + 1),
            "step": int(step),
            "kind": kind,
            "kind_aliases": list(kind_aliases or []),
            "sample_count": sample_count,
            "global_step": global_step,
            "bytes": int(path.stat().st_size),
            "sha256": self._trajectory_file_sha256(path),
        }
        self.trajectory_manifest["snapshots"].append(record)
        self._write_manifest()
        if evaluate_after:
            auc, loss, infer_time = self.evaluate("val")
            self._append_trajectory_metric({
                "event": "validation",
                "split": "validation",
                "epoch_idx": int(epoch_idx),
                "epoch_number": int(epoch_idx + 1),
                "step": int(step),
                "kind": kind,
                "auc": float(auc),
                "logloss": float(loss),
                "infer_time_s": float(infer_time),
            })
        return path

    def _trajectory_save_resume(self, epoch_idx):
        package = {
            "protocol_version": self.trajectory_protocol_version,
            "config_signature": self.trajectory_manifest["config_signature"],
            "completed_epochs": int(epoch_idx + 1),
            "best_auc": float(self.best_auc),
            "best_epoch_idx": int(self.best_epoch_idx),
            "model": self.network.state_dict(),
            "optimizer": self.optim.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(package, self.resume_path)

    def _trajectory_load_resume(self):
        package = torch.load(self.trajectory_resume, map_location="cpu", weights_only=False)
        if package.get("protocol_version") != self.trajectory_protocol_version:
            raise ValueError("trajectory resume protocol version mismatch")
        if package.get("config_signature") != self.trajectory_manifest["config_signature"]:
            raise ValueError("trajectory resume config signature mismatch")
        if not self.manifest_path.exists():
            raise FileNotFoundError("cannot resume without the original run_manifest.json")
        with open(self.manifest_path) as f:
            self.trajectory_manifest = json.load(f)
        self.network.load_state_dict(package["model"])
        self.network.to(self.device)
        self.optim.load_state_dict(package["optimizer"])
        rng = package["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        self._resume_completed_epochs = int(package["completed_epochs"])
        self.best_auc = float(package["best_auc"])
        self.best_epoch_idx = int(package["best_epoch_idx"])
        self._append_manifest_event({
            "event": "resume",
            "source": str(self.trajectory_resume),
            "completed_epochs_before": self._resume_completed_epochs,
        })

    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        #print(logit.shape)
        logloss = self.criterion(logit, label)
        regloss = self.network.reg()
        loss = regloss + logloss
        loss.backward()
        self.optim.step()
        return logloss.item()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def train(self, epochs):
        cur_auc = 0.0
        early_stop = False
        if self.trajectory_protocol:
            if self._resume_completed_epochs == 0:
                if self.init_path.exists():
                    raise FileExistsError("trajectory init.pt already exists")
                torch.save(self.network.state_dict(), self.init_path)
                self.trajectory_manifest["snapshots"].append({
                    "path": "init.pt",
                    "epoch_idx": None,
                    "epoch_number": None,
                    "step": 0,
                    "kind": "init",
                    "bytes": int(self.init_path.stat().st_size),
                    "sha256": self._trajectory_file_sha256(self.init_path),
                })
                self.trajectory_manifest["data_identity"] = self._trajectory_data_identity()
                self._write_manifest()
            else:
                self.trajectory_manifest.setdefault("data_identity", self._trajectory_data_identity())
                self._write_manifest()

            cur_auc = self.best_auc
            start_epoch = self._resume_completed_epochs
            total_train_samples = self._trajectory_train_count()
            batches_per_epoch = math.ceil(total_train_samples / self.bs)
            self._trajectory_train_samples = total_train_samples
            self._trajectory_epoch_batches = batches_per_epoch
            if self._resume_completed_epochs == 0:
                snapshot_plan = {
                    "batches_per_epoch": int(batches_per_epoch),
                    "train_samples": int(total_train_samples),
                    "batch_size": int(self.bs),
                    "first_epoch_snapshot_fractions": list(self.first_epoch_snapshot_fractions),
                    "fraction_step_map": {
                        "{}".format(f): int(math.ceil(f * batches_per_epoch))
                        for f in self.first_epoch_snapshot_fractions
                    },
                }
                self.trajectory_manifest["snapshot_plan"] = snapshot_plan
                self._write_manifest()
        else:
            start_epoch = 0
            batches_per_epoch = None

        for epoch_idx in range(start_epoch, int(epochs)):
            if self.trajectory_protocol and self.reset_emb_per_epoch and epoch_idx > 0:
                self.reset_embedding(epoch_idx)
            epoch_start = time.time()
            train_loss = 0.0
            step = 0
            step_targets = {}
            if self.trajectory_protocol:
                def _add_step_target(target, kind):
                    step_targets.setdefault(int(target), []).append(kind)

                if epoch_idx == 0:
                    if self.first_epoch_snapshot_fractions:
                        # t3-local-v1: fractions map to step=ceil(f*J); 1.00 is the
                        # epoch_end snapshot itself (aliased there, not re-saved).
                        for fraction in self.first_epoch_snapshot_fractions:
                            if fraction < 1.0:
                                _add_step_target(
                                    math.ceil(fraction * batches_per_epoch),
                                    "first_epoch_{}pct".format(int(round(fraction * 100))),
                                )
                    else:
                        _add_step_target(math.ceil(0.90 * batches_per_epoch), "first_epoch_90pct")
                        _add_step_target(math.ceil(0.95 * batches_per_epoch), "first_epoch_95pct")
                elif epoch_idx > 0:
                    _add_step_target(math.ceil(0.01 * batches_per_epoch), "epoch_start_1pct")
                    _add_step_target(math.ceil(0.05 * batches_per_epoch), "epoch_start_5pct")
                    _add_step_target(math.ceil(0.10 * batches_per_epoch), "epoch_start_10pct")

            # training loop
            for feature, label in self.dataloader.get_data("train", batch_size=self.bs):
                train_loss += self.train_on_batch(label, feature)
                step += 1
                if step in step_targets and step != batches_per_epoch:
                    event_kinds = step_targets.pop(step)
                    kind = "+".join(event_kinds)
                    if epoch_idx == 0:
                        filename = "{}.ep0.step{}".format(Path(self.model_dir).name, step)
                        self._save_trajectory_weights(
                            filename, epoch_idx=0, step=step,
                            kind=kind, evaluate_after=True,
                            kind_aliases=event_kinds[1:] if len(event_kinds) > 1 else None,
                        )
                    else:
                        auc, loss, infer_time = self.evaluate("val")
                        self._append_trajectory_metric({
                            "event": "validation",
                            "split": "validation",
                            "epoch_idx": int(epoch_idx),
                            "epoch_number": int(epoch_idx + 1),
                            "step": int(step),
                            "kind": kind,
                            "kind_aliases": event_kinds[1:] if len(event_kinds) > 1 else [],
                            "auc": float(auc),
                            "logloss": float(loss),
                            "infer_time_s": float(infer_time),
                        })

            # avoid division by zero when no batches returned
            if step == 0:
                print(f"No training batches returned for epoch {epoch_idx}, skipping epoch.")
                continue

            # snapshot the end-of-epoch weights BEFORE any early-stop restore logic
            if self.trajectory_protocol:
                filename = "{}.ep{}".format(Path(self.model_dir).name, epoch_idx)
                epoch_end_aliases = None
                if (
                    epoch_idx == 0
                    and self.first_epoch_snapshot_fractions
                    and 1.0 in self.first_epoch_snapshot_fractions
                ):
                    # t3-local-v1: fraction 1.00 aliases the epoch_end snapshot
                    # (dedup: never saved twice under a different file).
                    epoch_end_aliases = ["first_epoch_100pct"]
                    if step in step_targets:
                        # a fraction < 1.0 whose ceil collided with J also lands here
                        epoch_end_aliases = epoch_end_aliases + list(step_targets.pop(step))
                self._save_trajectory_weights(
                    filename, epoch_idx=epoch_idx, step=step, kind="epoch_end",
                    evaluate_after=False, kind_aliases=epoch_end_aliases,
                )
            elif self.save_every_epoch:
                torch.save(self.network.state_dict(), f"{self.model_dir}.ep{epoch_idx}")

            epoch_train_time = time.time() - epoch_start
            train_loss /= step

            # validation (evaluate returns auc, loss, infer_time)
            val_auc, val_loss, val_infer_time = self.evaluate("val")
            if self.trajectory_protocol:
                self._append_trajectory_metric({
                    "event": "validation",
                    "split": "validation",
                    "epoch_idx": int(epoch_idx),
                    "epoch_number": int(epoch_idx + 1),
                    "step": int(step),
                    "kind": "epoch_end",
                    "auc": float(val_auc),
                    "logloss": float(val_loss),
                    "infer_time_s": float(val_infer_time),
                })

            print(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Train Time:{ttrain:.3f}s | Val AUC:{val_auc:.6f} | Val Loss:{val_loss:.6f} | Val Inf Time:{vtime:.3f}s]"
                .format(
                    epoch=epoch_idx,
                    loss=train_loss,
                    ttrain=epoch_train_time,
                    val_auc=val_auc,
                    val_loss=val_loss,
                    vtime=val_infer_time
                )
            )

            # Log metrics to SwanLab
            if self.opt.get("use_swanlab", False):
                swanlab.log(
                    {
                        "epoch": epoch_idx,
                        "train_loss": train_loss,
                        "train_time": epoch_train_time,
                        "val_auc": val_auc,
                        "val_loss": val_loss,
                        "val_infer_time": val_infer_time,
                    }
                )

            early_stop = False
            if val_auc > cur_auc:
                cur_auc = val_auc
                if self.trajectory_protocol:
                    self.best_epoch_idx = epoch_idx
                    self.best_auc = val_auc
                    self.trajectory_manifest["best_epoch_idx"] = int(epoch_idx)
                    self._write_manifest()
                    torch.save(self.network.state_dict(), self.model_dir)
                elif self.tune_mode:
                    # In tune mode, save state dict in memory instead of disk
                    self.best_state_dict = {k: v.clone() for k, v in self.network.state_dict().items()}
                else:
                    torch.save(self.network.state_dict(), self.model_dir)
            else:
                if self.no_early_stop:
                    # trajectory collection mode: keep training through val dips
                    # (best checkpoint on disk stays at the last improvement)
                    continue
                # restore best model and evaluate on test
                if self.tune_mode:
                    self.network.load_state_dict(self.best_state_dict)
                else:
                    self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                te_auc, te_loss, te_infer_time = self.evaluate("test")

                print(
                    "Early stop at epoch {epoch:d} | Test AUC: {te_auc:.6f} | Test Loss: {te_loss:.6f} | Test Inf Time: {tinf:.3f}s"
                    .format(epoch=epoch_idx, te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time)
                )
                if not self.tune_mode:
                    self.log_dir = str(self.model_dir)[:-3] + ".log"
                    with open(self.log_dir, 'a') as f:
                        print(
                            f"Test AUC: {te_auc:.6f}, Test Loss: {te_loss:.6f}, Test Inf Time: {te_infer_time:.3f}s", file=f
                        )

                # Log test results as table
                if self.opt.get("use_swanlab", False):
                    test_headers = ["Metric", "Value"]
                    test_rows = [
                        ["Test AUC", f"{te_auc:.6f}"],
                        ["Test Loss", f"{te_loss:.6f}"],
                        ["Test Inference Time", f"{te_infer_time:.3f}s"],
                        ["Early Stop Epoch", str(epoch_idx)],
                    ]
                    test_table = swanlab.echarts.Table()
                    test_table.add(test_headers, test_rows)
                    swanlab.log({"Test Results": test_table})
                break

            if self.trajectory_protocol:
                self._trajectory_save_resume(epoch_idx)

        if not early_stop:
            if self.trajectory_protocol:
                self._append_manifest_event({
                    "event": "training_complete",
                    "completed_epochs": int(epochs),
                    "best_epoch_idx": int(self.best_epoch_idx),
                    "test_evaluations_during_training": 0,
                })
                return
            te_auc, te_loss, te_infer_time = self.evaluate("test")
            print(
                "Final Test AUC:{te_auc:.6f}, Test Loss:{te_loss:.6f}, Test Inf Time:{tinf:.3f}s".format(
                    te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time
                )
            )
            if not self.tune_mode:
                self.log_dir = str(self.model_dir)[:-3] + ".log"
                with open(self.log_dir, 'a') as f:
                    print(f"Test AUC: {te_auc:.6f}, Test Loss: {te_loss:.6f}, Test Inf Time: {te_infer_time:.3f}s", file=f)

            # Log test results as table
            if self.opt.get("use_swanlab", False):
                test_headers = ["Metric", "Value"]
                test_rows = [
                    ["Test AUC", f"{te_auc:.6f}"],
                    ["Test Loss", f"{te_loss:.6f}"],
                    ["Test Inference Time", f"{te_infer_time:.3f}s"],
                    ["Early Stop", "False"],
                ]
                test_table = swanlab.echarts.Table()
                test_table.add(test_headers, test_rows)
                swanlab.log({"Test Results": test_table})

    def evaluate(self, on: str):
        preds, trues = [], []
        infer_time = 0.0
        total_samples = 0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs):
            t0 = time.time()
            pred = self.eval_on_batch(feature)
            infer_time += time.time() - t0
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
            total_samples += label.shape[0]
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, infer_time


def main():
    model_opt = {
        "latent_dim": args.dim,
        "feat_num": args.feature,
        "field_num": args.field,
        "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn,
        "mlp_dims": args.mlp_dims,
        "cross": args.cross,
        # Wukong specific parameters
        "num_layers": args.num_layers,
        "num_embed_lcb": args.num_embed_lcb,
        "num_embed_fmb": args.num_embed_fmb,
        "rank_fmb": args.rank_fmb,
        "fm_mlp_dims": args.fm_mlp_dims,
        # RankMixer specific parameters
        "num_L": args.num_L,
        "expansion_rate": args.expansion_rate,
    }
    opt = {
        "model_opt": model_opt,
        "dataset": args.dataset,
        "model": args.model,
        "lr": args.lr,
        "l2": args.l2,
        "bsize": args.bsize,
        "epoch": args.max_epoch,
        "optimizer": args.optim,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "cuda": args.cuda,
        "use_swanlab": args.use_swanlab,
        "seed": args.seed,
        "trajectory_protocol": args.trajectory_protocol,
        "trajectory_resume": args.trajectory_resume,
        "save_every_epoch": args.save_every_epoch,
        "reset_emb_per_epoch": args.reset_emb_per_epoch,
        "no_early_stop": args.no_early_stop,
        "first_epoch_snapshot_fractions": args.first_epoch_snapshot_fractions,
    }

    # Print configuration in a readable format
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for key, value in opt.items():
        if key == 'model_opt':
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    trainer = Trainer(opt, tune_mode=args.tune_mode)
    trainer.train(args.max_epoch)
    # trainer.calibrate(args.max_epoch)


if __name__ == "__main__":
    """
    python trainer.py Criteo DeepFM --feature
    """
    main()
    if args.use_swanlab:
        try:
            swanlab.finish()
        except:
            pass
    sys.exit(0)
