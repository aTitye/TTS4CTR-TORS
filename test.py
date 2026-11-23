import torch
import argparse
import logging
import os, sys
import time
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
import modules.models as models
import configs as configs

parser = argparse.ArgumentParser(description="optfs tester")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model", type=str, help="specify model or model combo")

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
parser.add_argument(
    "--model_dim",
    action="append",
    default=[],
    help="override latent dim per sub-model as model=dim",
)

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
parser.add_argument("--num_L", type=int, default=3, help="number of layers in the model")
parser.add_argument("--expansion_rate", type=int, default=2, help="expansion rate for the model")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")
parser.add_argument("--seed", type=int, default=2021, help="random seed")
parser.add_argument("--exp_name", type=str, default="naive", help="experiment name (e.g. naive, distilled)")
parser.add_argument(
    "--eval_splits",
    nargs="+",
    choices=("test", "validation"),
    default=("test",),
    help="Splits to evaluate. Use validation for selection and reserve test for final reporting.",
)
parser.add_argument(
    "--save_logits",
    action="store_true",
    help="Save raw logits alongside probabilities for pooling analyses.",
)

args = parser.parse_args()

my_seed = args.seed
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '7'
os.environ['NUMEXPR_MAX_THREADS'] = '7'


class Tester(object):

    def __init__(self, opt):
        self.opt = opt
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])

        if "modellist" in opt and opt["modellist"]:
            print("Testing group model:", opt["modellist"])
            self.network = models.GroupModel(opt["model_opt"], opt["modellist"]).to(self.device)
        else:
            self.network = trainUtils.getModel(opt["model"], opt["model_opt"]).to(self.device)

        # Skip optimizer creation to avoid initializing compile/Triton during inference.
        self.criterion = None
        self.optim = None

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

    def eval_on_batch(self, data, return_logits=False):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            if return_logits:
                return prob, logit.detach().cpu().numpy()
        return prob

    def evaluate(self, on: str, save_logits=False):
        preds, trues, logits = [], [], []
        infer_time = 0.0
        sample_count = 0
        # for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 5):
            t0 = time.time()
            result = self.eval_on_batch(feature, return_logits=save_logits)
            if save_logits:
                pred, logit = result
                logits.append(logit)
            else:
                pred = result
            infer_time += time.time() - t0
            batch_size = feature.shape[0]
            sample_count += batch_size
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        per_sample_ms = (infer_time / sample_count * 1000) if sample_count else float('nan')
        y_logit = np.concatenate(logits).astype("float64") if save_logits else None
        return auc, loss, y_pred, y_true, y_logit, infer_time, per_sample_ms

    def prediction_locations(self, split: str):
        """Return shared labels and per-checkpoint prediction paths for a split."""
        model_path = Path(self.model_dir).resolve()
        dataset_name = self.opt["dataset"]
        dataset_root = next((parent for parent in model_path.parents if parent.name == dataset_name), None)
        if dataset_root is None:
            dataset_root = model_path.parent

        if split == "test":
            true_path = dataset_root / f"{dataset_name}_full_true.npy"
            prediction_name = f"full_pred_seed{self.opt.get('seed', 'unknown')}.npy"
            logit_name = f"full_logit_seed{self.opt.get('seed', 'unknown')}.npy"
        else:
            true_path = dataset_root / f"{dataset_name}_{split}_full_true.npy"
            prediction_name = f"full_pred_{split}_seed{self.opt.get('seed', 'unknown')}.npy"
            logit_name = f"full_logit_{split}_seed{self.opt.get('seed', 'unknown')}.npy"

        prediction_dir = model_path.parent / self.opt.get("exp_name", "naive")
        prediction_dir.mkdir(parents=True, exist_ok=True)
        return true_path, prediction_dir / prediction_name, prediction_dir / logit_name

    def save_evaluation(self, split: str, result):
        auc, loss, full_pred, full_true, full_logit, infer_time, per_sample_ms = result
        true_path, prediction_path, logit_path = self.prediction_locations(split)

        if not true_path.exists():
            np.save(true_path, full_true)
            print(f"Saved {split} ground truth to: {true_path}")
        else:
            existing_true = np.load(true_path, mmap_mode="r")
            if existing_true.shape != full_true.shape:
                raise ValueError(
                    f"Existing labels at {true_path} have shape {existing_true.shape}, "
                    f"but {split} evaluation produced {full_true.shape}."
                )

        np.save(prediction_path, full_pred)
        if full_logit is not None:
            np.save(logit_path, full_logit)

        message = (
            f"{split.title()} AUC: {auc:.6f}, {split.title()} Loss: {loss:.6f}, "
            f"Time: {infer_time:.3f}s, Per-sample: {per_sample_ms:.3f} ms"
        )
        print(message)
        print(f"Saved {split} predictions to: {prediction_path}")
        if full_logit is not None:
            print(f"Saved {split} logits to: {logit_path}")

        log_file = str(self.model_dir).replace(".pt", ".log")
        with open(log_file, "a") as file:
            print(message, file=file)
            print(f"Saved {split} predictions to: {prediction_path}", file=file)
            if full_logit is not None:
                print(f"Saved {split} logits to: {logit_path}", file=file)

    def test(self):
        print("Testing...")
        # Load model
        print(f"Loading model from {self.model_dir}")
        self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.network.eval()
        for split in self.opt["eval_splits"]:
            self.save_evaluation(split, self.evaluate(split, save_logits=self.opt["save_logits"]))
        print("Evaluation completed successfully.")


def main():
    model_dim_overrides = {}
    for override in args.model_dim:
        try:
            name, value = override.split('=')
            name = name.strip().lower()
            value = int(value)
            model_dim_overrides[name] = value
        except ValueError:
            print(f"Warning: Invalid model_dim format '{override}'. Expected 'model=dim'. Skipping.")
            continue

    modellist = None
    # Check if it is a group model
    if args.model in configs.GROUP_CONFIGS:
        modellist = configs.GROUP_CONFIGS[args.model]
    elif "+" in args.model:
        modellist = args.model.split("+")

    # Auto-fill per-model dims for group models to mirror trainer behavior
    if modellist:
        dataset_base_cfg = configs.BASE_MODEL_CONFIGS.get(args.dataset, {})
        unique_models = dict.fromkeys([m.lower() for m in modellist])

        for model_name in unique_models:
            if model_name in model_dim_overrides:
                continue
            base_dim = dataset_base_cfg.get(model_name, {}).get("dim")
            if base_dim is not None:
                model_dim_overrides[model_name] = base_dim

        for model_name in unique_models:
            model_dim_overrides.setdefault(model_name, args.dim)

    # For group models, rely on per-model dims; fall back to args.dim otherwise.
    if modellist:
        first_model_key = next(iter(model_dim_overrides)) if model_dim_overrides else None
        global_dim = model_dim_overrides.get(first_model_key) if first_model_key else args.dim
    else:
        global_dim = args.dim

    model_opt = {
        "latent_dim": global_dim,
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
        # Rankmixer specific parameters
        "num_L": args.num_L,
        "expansion_rate": args.expansion_rate,
        "model_dims": model_dim_overrides,
    }

    opt = {
        "model_opt": model_opt,
        "dataset": args.dataset,
        "model": args.model,
        "modellist": modellist,
        "lr": args.lr,
        "l2": args.l2,
        "bsize": args.bsize,
        "epoch": args.max_epoch,
        "optimizer": args.optim,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "cuda": args.cuda,
        "seed": args.seed,
        "exp_name": args.exp_name,
        "eval_splits": args.eval_splits,
        "save_logits": args.save_logits,
    }
    print("=" * 60 + "\n")

    # Print configuration in a readable format
    print("\n" + "=" * 60)
    print("Test Configuration")
    print("=" * 60)
    for key, value in opt.items():
        if key == 'model_opt':
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    tester = Tester(opt)
    tester.test()


if __name__ == "__main__":
    main()
