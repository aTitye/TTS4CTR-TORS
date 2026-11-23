import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['NUMEXPR_MAX_THREADS'] = '4'
# XLA_CACHE_DIR = '/root/tf_xla_cache'
# os.makedirs(XLA_CACHE_DIR, exist_ok=True)
# os.environ['TF_XLA_FLAGS'] = f'--tf_xla_enable_xla_devices --tf_xla_persistent_cache_directory={XLA_CACHE_DIR} --tf_xla_persistent_cache_read_only=0'

import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
import modules.models as models
import time
# import swanlab
import configs as configs

parser = argparse.ArgumentParser(description="optfs trainer group")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model_combo", type=str, help="specify a model combo key, backbone, or a plus-separated list")

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
parser.add_argument("--skip_test", action="store_true",
                    help="save the validation-best checkpoint without reading test")
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
parser.add_argument("--num_embed_lcb", type=int, default=16, help="number of embedding in Linear Compress Block in Wukong")
parser.add_argument(
    "--num_embed_fmb", type=int, default=16, help="number of embedding in Factorization Machine Block in Wukong"
)
parser.add_argument("--rank_fmb", type=int, default=24, help="rank in Wukong")
parser.add_argument("--fm_mlp_dims", type=int, nargs='+', default=[512, 512], help="sub-mlp within the FM Block in Wukong")

# Rankmixer specific parameters
parser.add_argument("--num_L", type=int, default=3, help="number of layers in RankMixer")
parser.add_argument("--expansion_rate", type=int, default=2, help="expansion rate in RankMixer")
parser.add_argument(
    "--aux_branch_weight",
    type=float,
    default=0.0,
    help="branch-level auxiliary supervision; 0 uses the group objective.",
)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")
parser.add_argument("--seed", type=int, default=2021, help="random seed")

# SwanLab configuration
parser.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")
parser.add_argument("--swanlab_project", type=str, default="Group_TTS", help="SwanLab project name")
parser.add_argument("--swanlab_api_key", type=str, default=os.environ.get("SWANLAB_API_KEY", ""), help="SwanLab API key")
parser.add_argument("--experiment_name", type=str, default=None, help="Custom experiment name for SwanLab")

args = parser.parse_args()

my_seed = args.seed
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
# Initialize SwanLab
if args.use_swanlab:
    import swanlab
    os.environ["SWANLAB_API_KEY"] = args.swanlab_api_key
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"{args.dataset}_{args.model_combo}_seed{args.seed}"
    swanlab.init(project=args.swanlab_project, experiment_name=experiment_name, config=vars(args))


class Trainer(object):

    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.opt = opt

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])

        modellist = opt["modellist"]
        print("Training group model:", modellist)
        # print("Model options:", opt["model_opt"])

        self.network = models.GroupModel(opt["model_opt"], modellist).to(self.device)
        # print(self.network)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptim(self.network, opt["optimizer"], self.lr, self.l2)

        # Compute and log model statistics
        self.compute_model_stats()

    def compute_model_stats(self):
        """Compute and log model FLOPs, MACs, and parameters."""
        model_opt = self.opt["model_opt"]
        cal_bs = 1  # Use a smaller batch size for profiling
        # Create a dummy input tensor for profiling
        input_tensor = torch.randint(0, model_opt["feat_num"], (cal_bs, model_opt["field_num"])).long().to(self.device)

        try:
            flops, macs, params = trainUtils.compute_model_complexity(self.network, input_tensor)
            flops = trainUtils.format_count(flops)
            macs = trainUtils.format_count(macs)
            params = trainUtils.format_count(params)

            print(f"\n{'='*60}")
            print(f"Model Combo: {self.opt['model_combo']}, Batch Size: {cal_bs}")
            print(f"Input Shape: ({cal_bs}, {model_opt['field_num']})")
            print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")
            print(f"{'='*60}\n")

            # Log model stats to SwanLab as an ECharts table
            if self.opt.get("use_swanlab", False):
                headers = ["Metric", "Value"]
                rows = [
                    ["Model Combo", self.opt['model_combo']],
                    ["Batch Size", str(cal_bs)],
                    ["Input Shape", f"({cal_bs}, {model_opt['field_num']})"],
                    ["FLOPs", flops],
                    ["MACs", macs],
                    ["Params", params],
                ]
                table = swanlab.echarts.Table()
                table.add(headers, rows)
                swanlab.log({"Model Summary": table})
        except Exception as e:
            print(f"Failed to compute model stats: {e}")

    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        branch_weight = float(self.opt.get("aux_branch_weight", 0.0))
        if branch_weight > 0 and isinstance(self.network, models.GroupModel):
            branch_logits = self.network(data, return_branch_logits=True)
            group_logit = branch_logits.mean(dim=1)
        else:
            group_logit = self.network(data)
            branch_logits = None
        logloss = self.criterion(group_logit, label)
        if branch_logits is not None:
            branch_loss = self.criterion(branch_logits.reshape(-1, 1), label.expand(-1, branch_logits.shape[1]).reshape(-1, 1))
            logloss = logloss + branch_weight * branch_loss
        regloss = self.network.reg()
        # print(logloss, regloss)
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
        step = 0
        cur_auc = 0.0
        early_stop = False
        for epoch_idx in range(int(epochs)):
            epoch_start = time.time()
            train_loss = .0
            step = 0
            for feature, label in self.dataloader.get_data("train", batch_size=self.bs):
                train_loss += self.train_on_batch(label, feature)
                step += 1

            if step == 0:
                print(f"No training batches returned for epoch {epoch_idx}, skipping epoch.")
                continue

            train_loss /= step
            epoch_train_time = time.time() - epoch_start
            val_auc, val_loss, val_infer_time = self.evaluate("val")
            print(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Train Time:{ttrain:.3f}s | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f} | Val Inf Time:{vtime:.3f}s]"
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

            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                if self.opt.get("skip_test", False):
                    print(f"Early stop at epoch {epoch_idx}; test evaluation skipped.")
                    break
                te_auc, te_loss, te_infer_time = self.evaluate("test")

                self.log_dir = str(self.model_dir)[:-3] + ".log"
                print(
                    "Early stop at epoch {epoch:d}|Test AUC: {te_auc:.6f}, Test Loss:{te_loss:.6f} | Test Inf Time:{tinf:.3f}s"
                    .format(epoch=epoch_idx, te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time)
                )
                with open(self.log_dir, 'a') as f:
                    print(f"Test AUC: {te_auc:.6f}, Test Loss: {te_loss:.6f}, Test Inf Time: {te_infer_time:.3f}s", file=f)

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
        if not early_stop and not self.opt.get("skip_test", False):
            te_auc, te_loss, te_infer_time = self.evaluate("test")
            print(
                "Final Test AUC:{te_auc:.6f}, Test Loss:{te_loss:.6f}, Test Inf Time:{tinf:.3f}s".format(
                    te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time
                )
            )
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
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs):
            t0 = time.time()
            pred = self.eval_on_batch(feature)
            infer_time += time.time() - t0
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, infer_time


def main():
    model_dim_overrides = {}
    for override in args.model_dim:
        try:
            name, value = override.split("=", 1)
            name = name.strip().lower()
            value = int(value.strip())
        except ValueError:
            raise ValueError(f"Invalid --model_dim entry '{override}'. Use format model=dim")
        model_dim_overrides[name] = value

    model_opt = {
        "latent_dim": args.dim,
        "feat_num": args.feature,
        "field_num": args.field,
        "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn,
        "mlp_dims": args.mlp_dims,
        # Wukong specific parameters
        "num_layers": args.num_layers,
        "num_embed_lcb": args.num_embed_lcb,
        "num_embed_fmb": args.num_embed_fmb,
        "rank_fmb": args.rank_fmb,
        "fm_mlp_dims": args.fm_mlp_dims,
        # RankMixer specific parameters
        "num_L": args.num_L,
        "expansion_rate": args.expansion_rate,
        "model_dims": model_dim_overrides,
    }

    # Resolve model combo
    if args.model_combo in configs.GROUP_CONFIGS:
        modellist = configs.GROUP_CONFIGS[args.model_combo]
    else:
        # Fallback to old behavior if it looks like a list of models separated by +
        if "+" in args.model_combo:
            modellist = args.model_combo.split("+")
        else:
            raise ValueError(f"Model combo '{args.model_combo}' not found in configs and not a valid '+' separated list.")

    opt = {
        "model_opt": model_opt,
        "dataset": args.dataset,
        "model_combo": args.model_combo,
        "modellist": modellist,
        "lr": args.lr,
        "l2": args.l2,
        "bsize": args.bsize,
        "epoch": args.max_epoch,
        "optimizer": args.optim,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "cuda": args.cuda,
        "use_swanlab": args.use_swanlab,
        "aux_branch_weight": args.aux_branch_weight,
        "skip_test": args.skip_test,
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

    trainer = Trainer(opt)
    trainer.train(args.max_epoch)
    # trainer.calibrate(args.max_epoch)


if __name__ == "__main__":
    """
    python train_group.py avazu wukong+rankmixer
    """
    main()
    if args.use_swanlab:
        try:
            swanlab.finish()
        except:
            pass
    sys.exit(0)
