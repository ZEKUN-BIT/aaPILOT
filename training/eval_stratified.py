"""Stratified evaluation of fine-tuned LigandMPNN checkpoints against the
vanilla baseline on the pathway test split.

Strata: structure_state (experimental_holo / transferred_pose /
predicted_no_ligand) x target (thrA_AK / thrB / thrC). Metrics: overall
sequence recovery and pocket recovery (residues with CA within the binding-site
cutoff of ligand atoms), reported with paired bootstrap 95% CIs (numpy only,
so it runs in the diffdock env without pandas/seaborn).
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_integrity import seed_everything
from model_utils import loss_nll
from train import model_from_checkpoint

SEED = 42
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def featurize(batch, device, atom_context_num=25, binding_cutoff=5.0):
    B = len(batch)
    lengths = [len(b["seq"]) for b in batch]
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3]); S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32); chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.ones([B, L_max], dtype=np.int32)
    Y = np.zeros([B, L_max, atom_context_num, 3])
    Y_t = np.zeros([B, L_max, atom_context_num], dtype=np.int32)
    Y_m = np.zeros([B, L_max, atom_context_num], dtype=np.float32)
    is_binding_site = np.zeros([B, L_max], dtype=bool)
    for i, b in enumerate(batch):
        seq = b["seq"]; l_seq = len(seq)
        mask[i, :l_seq] = 1.0; chain_M[i, :l_seq] = 1.0
        chain_keys = [k.replace("seq_chain_", "") for k in b if k.startswith("seq_chain_")]
        global_idx = 0; all_ca = []
        for c_idx, chain_id in enumerate(chain_keys):
            c_seq = b[f"seq_chain_{chain_id}"]; c_len = len(c_seq)
            for j, aa in enumerate(c_seq):
                S[i, global_idx + j] = ALPHABET.index(aa) if aa in ALPHABET else 20
            chain_encoding_all[i, global_idx:global_idx + c_len] = c_idx + 1
            residue_idx[i, global_idx:global_idx + c_len] = 100 * c_idx + np.arange(c_len)
            c_coords = b[f"coords_chain_{chain_id}"]
            for k, atom in enumerate(("N", "CA", "C", "O")):
                X[i, global_idx:global_idx + c_len, k, :] = c_coords[f"{atom}_chain_{chain_id}"]
            all_ca.extend(c_coords[f"CA_chain_{chain_id}"])
            global_idx += c_len
        if b.get("ligand_coords"):
            l_coords = np.asarray(b["ligand_coords"], dtype=float)
            l_types = np.asarray(b["ligand_types"], dtype=np.int64)
            p_ca = np.asarray(all_ca, dtype=float)
            dists = np.linalg.norm(p_ca[:, None, :] - l_coords[None, :, :], axis=-1)
            is_binding_site[i, :l_seq] = dists.min(axis=1) < binding_cutoff
            for r_idx in range(l_seq):
                k = min(len(l_coords), atom_context_num)
                top = np.argsort(dists[r_idx])[:k]
                Y[i, r_idx, :k, :] = l_coords[top]
                Y_t[i, r_idx, :k] = l_types[top]
                Y_m[i, r_idx, :k] = 1.0
    return {
        "X": torch.from_numpy(X).float().to(device), "S": torch.from_numpy(S).long().to(device),
        "mask": torch.from_numpy(mask).float().to(device), "chain_M": torch.from_numpy(chain_M).float().to(device),
        "residue_idx": torch.from_numpy(residue_idx).long().to(device),
        "chain_encoding_all": torch.from_numpy(chain_encoding_all).long().to(device),
        "Y": torch.from_numpy(Y).float().to(device), "Y_t": torch.from_numpy(Y_t).long().to(device),
        "Y_m": torch.from_numpy(Y_m).float().to(device),
        "is_binding_site": torch.from_numpy(is_binding_site).to(device),
    }


def evaluate_pair(baseline, finetuned, loader, device, atom_context_num, seed):
    """Evaluate both models with identical features and decoding orders."""
    rows = []
    baseline.eval()
    finetuned.eval()
    seed_everything(seed)
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            feat = featurize(batch, device, atom_context_num)
            decoding_order = torch.randn_like(feat["mask"])
            outputs = []
            for model in (baseline, finetuned):
                log_probs = model(feat["X"], feat["S"], feat["mask"], feat["chain_M"],
                                  feat["residue_idx"], feat["chain_encoding_all"],
                                  decoding_order,
                                  Y=feat["Y"], Y_t=feat["Y_t"], Y_m=feat["Y_m"])
                pred = torch.argmax(log_probs, dim=-1)
                loss, _ = loss_nll(feat["S"], log_probs, feat["mask"])
                outputs.append((pred, loss))
            for i, b in enumerate(batch):
                valid = feat["mask"][i] * feat["chain_M"][i]
                n = int(valid.sum().item())
                bs = feat["is_binding_site"][i]
                n_bs = int((bs * valid).sum().item())
                metrics = []
                for pred, loss in outputs:
                    rec = float((((pred[i] == feat["S"][i]) * valid).sum() / n).item()) if n else np.nan
                    nll = float((loss[i] * valid).sum().item() / n) if n else np.nan
                    bs_rec = float((((pred[i] == feat["S"][i]) * bs * valid).sum() / n_bs).item()) if n_bs else np.nan
                    metrics.append((rec, bs_rec, float(np.exp(nll)) if not np.isnan(nll) else np.nan))
                van, ft = metrics
                rows.append({
                    "name": b.get("name"), "target": b.get("target", ""),
                    "structure_state": b.get("structure_state", ""),
                    "cluster_id": b.get("cluster_id", ""),
                    "has_ligand": bool(b.get("ligand_coords")),
                    "seed": seed, "n_res": n,
                    "recovery": ft[0], "recovery_vanilla": van[0],
                    "bs_recovery": ft[1], "bs_recovery_vanilla": van[1],
                    "perplexity": ft[2], "perplexity_vanilla": van[2],
                })
    return rows


def bootstrap_delta(van, ft, rng, samples=10000, groups=None):
    d = np.asarray(ft, dtype=float) - np.asarray(van, dtype=float)
    if groups is not None:
        grouped = defaultdict(list)
        for group, value in zip(groups, d):
            grouped[group].append(value)
        d = np.asarray([np.mean(values) for values in grouped.values()])
    idx = rng.integers(0, len(d), size=(samples, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)), len(d))


def summarize(rows, model_name, seed, rng, samples):
    summary = []
    strata = [("all", lambda r: True)]
    for state in ("experimental_holo", "transferred_pose", "predicted_no_ligand"):
        strata.append((f"state={state}", lambda r, s=state: r["structure_state"] == s))
    for target in ("thrA_AK", "thrB", "thrC"):
        strata.append((f"target={target}", lambda r, t=target: r["target"] == t))
    for state in ("transferred_pose", "predicted_no_ligand"):
        for target in ("thrA_AK", "thrB", "thrC"):
            strata.append((f"state={state},target={target}",
                           lambda r, s=state, t=target: r["structure_state"] == s and r["target"] == t))
    for label, predicate in strata:
        subset = [r for r in rows if predicate(r) and not np.isnan(r["recovery"])]
        for metric, key in (("recovery", "recovery"), ("pocket_recovery", "bs_recovery")):
            paired = [r for r in subset if not np.isnan(r[key]) and not np.isnan(r[f"{key}_vanilla"])]
            if len(paired) < 2:
                continue
            for unit, groups in (("row", None), ("cluster", [r["cluster_id"] for r in paired])):
                delta, lo, hi, n_units = bootstrap_delta(
                    [r[f"{key}_vanilla"] for r in paired], [r[key] for r in paired],
                    rng, samples, groups=groups)
                summary.append({
                    "model": model_name, "seed": seed, "bootstrap_unit": unit,
                    "stratum": label, "metric": metric, "n_rows": len(paired),
                    "n_units": n_units,
                    "vanilla_mean": float(np.mean([r[f"{key}_vanilla"] for r in paired])),
                    "finetuned_mean": float(np.mean([r[key] for r in paired])),
                    "mean_delta": delta, "ci95_low": lo, "ci95_high": hi,
                })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--finetuned", required=True, help="path to best.pt")
    ap.add_argument("--baseline-dir", default="../model_params")
    ap.add_argument("--models", nargs="+", default=["ligandmpnn_v_32_005_25"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--atom-context-num", type=int, default=25)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.test_jsonl) as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    rng = np.random.default_rng(SEED)

    summary_rows = []
    for model_name in args.models:
        baseline_path = Path(args.baseline_dir) / f"{model_name}.pt"
        if not baseline_path.is_file():
            print(f"skip {model_name}: baseline missing"); continue
        base_ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
        base_model, *_ = model_from_checkpoint(base_ckpt, args.atom_context_num)
        base_model.to(device)
        ft_ckpt = torch.load(args.finetuned, map_location=device, weights_only=False)
        ft_model, *_ = model_from_checkpoint(ft_ckpt, args.atom_context_num)
        ft_model.to(device)

        ft_rows = []
        for seed in args.seeds:
            seed_rows = evaluate_pair(base_model, ft_model, loader, device, args.atom_context_num, seed)
            ft_rows.extend(seed_rows)
            summary_rows.extend(summarize(seed_rows, model_name, seed, rng, args.bootstrap_samples))
        fieldnames = ["name", "target", "structure_state", "cluster_id", "has_ligand", "seed", "n_res",
                      "recovery", "recovery_vanilla", "bs_recovery", "bs_recovery_vanilla",
                      "perplexity", "perplexity_vanilla"]
        with open(out / f"{model_name}_per_row.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ft_rows:
                writer.writerow({k: r.get(k) for k in fieldnames})

        del base_model, ft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(out / "stratified_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "bootstrap_unit", "stratum", "metric",
                                               "n_rows", "n_units", "vanilla_mean", "finetuned_mean",
                                               "mean_delta", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        if (row["bootstrap_unit"] == "cluster" and row["metric"] == "recovery" and
                row["stratum"] in ("all", "state=experimental_holo", "state=transferred_pose",
                                   "state=predicted_no_ligand")):
            print(f'seed={row["seed"]} {row["stratum"]:35s} n={row["n_units"]:3d} clusters '
                  f'vanilla={row["vanilla_mean"]:.4f} '
                  f'ft={row["finetuned_mean"]:.4f} delta={row["mean_delta"]:+.4f} '
                  f'[{row["ci95_low"]:+.4f},{row["ci95_high"]:+.4f}]')
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
