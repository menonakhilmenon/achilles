#!/usr/bin/env python
"""Train per-layer expert predictors on OLMoE traces and compare against the
untrained gate-ahead baseline.

For lookahead D: predict top-8 experts of layer l+D from the hidden state at
layer l. Two learned models per (l, D):
  linear : 2048 -> 64  (a probe; same shape as the router itself)
  mlp    : 2048 -> 512 -> 64
Trained with BCE against soft targets (router softmax probs at l+D), which
distills margin information rather than hard top-8.

Split: by prompt (80/20) to avoid within-generation leakage.
Output: bench/results/predictor.json + docs/assets/olmoe-predictor.png
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import analyze_olmoe as A

ROOT = A.ROOT
DELTAS = [1, 4, 8]
MS = [8, 16, 32]
EPOCHS = 6
DEV = "cpu"


def load_all():
    prompts = A.load_prompts()
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(prompts))
    n_test = max(len(prompts) // 5, 1)
    test_ids = set(idx[:n_test].tolist())
    train, test = [], []
    for i, p in enumerate(prompts):
        hidden = np.load(p["dir"] / "decode_hidden.npy").astype(np.float32)
        (test if i in test_ids else train).append(
            {"hidden": hidden, "logits": p["logits"].astype(np.float32), "tops": p["tops"]})
    return train, test


def recall_at(scores, actual, m):
    order = np.argsort(-scores, axis=1)[:, :m]
    found = (order[:, :, None] == actual[:, None, :]).any(axis=1)
    return float(found.mean())


def main():
    torch.manual_seed(0)
    torch.set_num_threads(6)
    train, test = load_all()
    L = train[0]["logits"].shape[1]
    E = train[0]["logits"].shape[2]
    print(f"train {len(train)} / test {len(test)} prompts, L={L} E={E}")

    gates = A.load_gates(L)
    results = {"gate_ahead": {}, "linear": {}, "mlp": {}}

    for D in DELTAS:
        ga_r = {m: [] for m in MS}
        li_r = {m: [] for m in MS}
        ml_r = {m: [] for m in MS}
        for l in range(L - D):
            Xtr = torch.from_numpy(np.concatenate([p["hidden"][:, l] for p in train]))
            Ptr = torch.from_numpy(
                np.concatenate([torch.softmax(torch.from_numpy(p["logits"][:, l + D]), -1).numpy()
                                for p in train]))
            Xte = np.concatenate([p["hidden"][:, l] for p in test])
            Yte = np.concatenate([p["tops"][:, l + D] for p in test])

            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5

            lin = nn.Linear(Xtr.shape[1], E)
            with torch.no_grad():  # warm start at the gate-ahead solution
                lin.weight.copy_(torch.from_numpy(gates[l + D]))
                lin.bias.zero_()
            mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(),
                                nn.Linear(512, E))

            for kind, model, use_norm in (("linear", lin, False), ("mlp", mlp, True)):
                opt = torch.optim.Adam(model.parameters(), lr=3e-4)
                for _ in range(EPOCHS):
                    perm = torch.randperm(Xtr.shape[0])
                    for i in range(0, len(perm), 4096):
                        b = perm[i:i + 4096]
                        x = (Xtr[b] - mu) / sd if use_norm else Xtr[b]
                        opt.zero_grad()
                        # KL distillation against the true router distribution
                        loss = -(Ptr[b] * torch.log_softmax(model(x), -1)).sum(-1).mean()
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    xte = torch.from_numpy(Xte)
                    if use_norm:
                        xte = (xte - mu) / sd
                    scores = model(xte).numpy()
                tgt = li_r if kind == "linear" else ml_r
                for m in MS:
                    tgt[m].append(recall_at(scores, Yte, m))

            ga_scores = Xte @ gates[l + D].T
            for m in MS:
                ga_r[m].append(recall_at(ga_scores, Yte, m))
            print(f"D={D} l={l}: ga16={ga_r[16][-1]:.3f} lin16={li_r[16][-1]:.3f} "
                  f"mlp16={ml_r[16][-1]:.3f}")

        for name, r in (("gate_ahead", ga_r), ("linear", li_r), ("mlp", ml_r)):
            results[name][str(D)] = {str(m): float(np.mean(r[m])) for m in MS}

    Path(ROOT / "bench/results/predictor.json").write_text(json.dumps(results, indent=1))

    # chart: recall@16 vs delta, three methods (fixed categorical order)
    fig, ax = A.newfig()
    methods = [("gate_ahead", "gate-ahead (no training)"), ("linear", "linear probe"),
               ("mlp", "MLP 512")]
    for i, (key, label) in enumerate(methods):
        ys = [results[key][str(D)]["16"] for D in DELTAS]
        ax.plot(DELTAS, ys, color=A.CAT[i], linewidth=2, marker="o", markersize=5)
        ax.annotate(label, (DELTAS[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=A.CAT[i], fontsize=10, va="center")
    ax.set_xlabel("lookahead Δ (layers ahead)")
    ax.set_ylabel("recall of actual top-8, prefetching 16")
    ax.set_title("Trained predictors vs. gate-ahead baseline (held-out prompts)")
    ax.set_xticks(DELTAS)
    ax.set_xlim(right=DELTAS[-1] * 1.55)
    ax.set_ylim(0.5, 1.02)
    A.savefig(fig, "olmoe-predictor.png")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
