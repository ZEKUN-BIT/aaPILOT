"""Fine-tune LigandMPNN on an audited, cluster-separated enzyme dataset."""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from data_integrity import seed_everything, validate_disjoint_jsonl
from model_utils import ProteinMPNN, loss_nll, loss_smoothed
from featurization import featurize_ligand_mpnn
from cluster_balanced_sampler import ClusterBalancedSampler


MODELS = [
    "ligandmpnn_v_32_005_25", "ligandmpnn_v_32_010_25",
    "ligandmpnn_v_32_020_25", "ligandmpnn_v_32_030_25",
]


class JsonlDataset(Dataset):
    def __init__(self, path, max_length=2400):
        self.rows = []
        self.dropped = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if len(row["seq"]) <= max_length:
                        self.rows.append(row)
                    else:
                        self.dropped += 1
        if self.dropped:
            print(f"{path}: dropped {self.dropped} rows longer than {max_length}")
        if not self.rows:
            raise ValueError(f"No usable records in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_featurize(rows, atom_context_num):
    """Featurize one batch on CPU; runs inside DataLoader workers, so CPU cost
    overlaps with GPU compute via prefetching."""
    return featurize_ligand_mpnn(rows, "cpu", atom_context_num)


def to_device(feat, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in feat.items()}


class LengthSortedBatchSampler(Sampler):
    """Pack the balanced sampler's epoch draws into length-sorted batches.

    Sorting changes only the ORDER of the same multiset of draws, so the
    target->cluster->member balance is preserved while same-length rows are
    batched together (minimal padding waste, large speedup vs batch_size=1).
    """

    def __init__(self, base_sampler, lengths, batch_size, seed=42):
        super().__init__()
        self.base = base_sampler
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.base.set_epoch(epoch)

    def __iter__(self):
        indices = list(self.base)
        indices.sort(key=lambda i: self.lengths[i])
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return (len(self.base) + self.batch_size - 1) // self.batch_size


def model_from_checkpoint(checkpoint, atom_context_num):
    state = checkpoint.get("model_state_dict", checkpoint)
    hidden = checkpoint.get("hidden_dim", 128)
    layers = checkpoint.get("num_encoder_layers", 3)
    edges = checkpoint.get("num_edges", 25)
    model = ProteinMPNN(num_letters=21, node_features=hidden, edge_features=hidden,
                        hidden_dim=hidden, num_encoder_layers=layers,
                        num_decoder_layers=layers, k_neighbors=edges,
                        augment_eps=0.0, dropout=0.1, model_type="ligand_mpnn",
                        atom_context_num=atom_context_num)
    model.load_state_dict(state)
    return model, hidden, layers, edges


def run_model(model_name, loaders, checkpoint_dir, output_dir, device, args):
    ckpt_path = Path(checkpoint_dir) / f"{model_name}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, hidden, layers, edges = model_from_checkpoint(checkpoint, args.atom_context_num)
    model.to(device)
    noise = float(model_name.split("_")[3]) / 100.0
    for name, parameter in model.named_parameters():
        if args.freeze_mode == "decoder_head":
            parameter.requires_grad = name in ("W_out.weight", "W_out.bias") or \
                f"decoder_layers.{layers - 1}." in name
        else:  # "current": every layer's MLP output projection + last decoder layer + head
            parameter.requires_grad = "W_out" in name or f"decoder_layers.{layers - 1}" in name
    if args.compile_model:
        model = torch.compile(model, fullgraph=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: trainable {trainable / 1e6:.2f}M params (freeze_mode={args.freeze_mode}, ema={args.use_ema})")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad} if args.use_ema else None
    best = float("inf")
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total = 0.0; optimizer.zero_grad(set_to_none=True)
        sampler = getattr(loaders["train"], "batch_sampler", None) or loaders["train"].sampler
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(loaders["train"], desc=f"{model_name} epoch {epoch + 1}")):
            feat = to_device(batch, device)
            valid = feat["mask"] * feat["chain_M"]
            randn = torch.randn_like(feat["mask"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                log_probs = model(feat["X"] + torch.randn_like(feat["X"]) * noise * feat["mask"][:, :, None, None],
                                  feat["S"], feat["mask"], feat["chain_M"], feat["residue_idx"],
                                  feat["chain_encoding_all"], randn, Y=feat["Y"], Y_t=feat["Y_t"], Y_m=feat["Y_m"])
                _, loss = loss_smoothed(feat["S"], log_probs, valid, weight=0.1)
                scaled = loss / args.accumulation_steps
            scaler.scale(scaled).backward()
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    decay = args.ema_decay
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                ema[n].mul_(decay).add_(p.detach(), alpha=1 - decay)
            total += loss.item()
        # evaluate with EMA weights
        if ema is not None:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(ema[n])
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                feat = to_device(batch, device)
                valid = feat["mask"] * feat["chain_M"]
                log_probs = model(feat["X"], feat["S"], feat["mask"], feat["chain_M"], feat["residue_idx"],
                                  feat["chain_encoding_all"], torch.randn_like(feat["mask"]),
                                  Y=feat["Y"], Y_t=feat["Y_t"], Y_m=feat["Y_m"])
                _, loss = loss_nll(feat["S"], log_probs, valid); val_total += loss.item()
        train_loss, val_loss = total / len(loaders["train"]), val_total / len(loaders["val"])
        print(f"{model_name} epoch={epoch + 1} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            state = model.state_dict()
            if ema is not None:
                state = {k: ema.get(k, v) for k, v in state.items()}
            torch.save({"epoch": epoch, "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best, "atom_context_num": args.atom_context_num, "num_edges": edges,
                        "hidden_dim": hidden, "num_encoder_layers": layers, "noise_level": noise}, model_dir / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--checkpoint-dir", default="../model_params")
    ap.add_argument("--output-dir", default="training_runs")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accumulation-steps", type=int, default=8); ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--atom-context-num", type=int, default=25); ap.add_argument("--max-length", type=int, default=2400)
    ap.add_argument("--val-max-samples", type=int, default=512, help="cap on val batches per epoch (0 = full val)")
    ap.add_argument("--samples-per-epoch", type=int, default=6000,
                    help="training draws per epoch for the cluster-balanced sampler (0 = one draw per row); "
                         "the target->cluster->member balance is preserved regardless")
    ap.add_argument("--use-cluster-balanced", action="store_true", default=True, help="cluster-balanced sampling (recommended)")
    ap.add_argument("--freeze-mode", choices=("current", "decoder_head"), default="current",
                    help="current: every layer's W_out projection + last decoder layer + head; "
                         "decoder_head: only final head + last decoder layer")
    ap.add_argument("--use-ema", action="store_true", default=True, help="exponential moving average of trainable weights")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--compile-model", action="store_true", default=False, help="torch.compile the model (test on GPU)")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for CPU featurization prefetch")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    data = Path(args.data_dir); paths = {s: data / f"{s}.jsonl" for s in ("train", "val", "test")}
    if any(not p.is_file() for p in paths.values()): raise FileNotFoundError("Missing split JSONL")
    metadata = validate_disjoint_jsonl({k: str(v) for k, v in paths.items()})
    print(json.dumps({k: {"rows": v["rows"], "ligand_rows": v["ligand_rows"]} for k, v in metadata.items()}))
    train_ds = JsonlDataset(paths["train"], args.max_length)
    val_ds = JsonlDataset(paths["val"], args.max_length)
    if args.val_max_samples and args.val_max_samples < len(val_ds.rows):
        rng = np.random.default_rng(args.seed)
        val_ds.rows = [val_ds.rows[i] for i in
                       rng.choice(len(val_ds.rows), size=args.val_max_samples, replace=False)]
        print(f"val subset: {len(val_ds.rows)} rows")
    from functools import partial
    collate = partial(collate_featurize, atom_context_num=args.atom_context_num)
    train_sampler = ClusterBalancedSampler(train_ds.rows,
                                           num_samples=args.samples_per_epoch or None,
                                           seed=args.seed) if args.use_cluster_balanced else None
    if train_sampler is not None and args.batch_size > 1:
        train_sampler = LengthSortedBatchSampler(
            train_sampler, [len(r["seq"]) for r in train_ds.rows],
            args.batch_size, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                                  num_workers=args.num_workers, prefetch_factor=4,
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    elif train_sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, collate_fn=collate,
                                  num_workers=args.num_workers, prefetch_factor=4,
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                                  num_workers=args.num_workers, prefetch_factor=4,
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers, prefetch_factor=4, pin_memory=True)
    loaders = {"train": train_loader, "val": val_loader}
    run = Path(args.output_dir) / datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps({**vars(args), "device": str(device)}, indent=2), encoding="utf-8")
    for model_name in args.models: run_model(model_name, loaders, args.checkpoint_dir, run, device, args)


if __name__ == "__main__": main()
