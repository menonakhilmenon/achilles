#!/usr/bin/env python
"""Analyze OLMoE routing traces: skew, temporal locality, cache policies,
cross-layer (gate-ahead) predictability, router margins.

Reads traces/olmoe/<domain>/<id>/{decode_logits,decode_hidden,prefill_logits}.npy
Writes docs/assets/*.png, bench/results/olmoe-analysis.json, and prints a summary.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "traces/olmoe"
ASSETS = ROOT / "docs/assets"
K = 8  # experts per token

# ---- chart chrome (reference palette, light mode) ---------------------------
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SEQ3 = ["#86b6ef", "#2a78d6", "#104281"]  # sequential blue 250/450/650

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def newfig():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    return fig, ax


def savefig(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ASSETS / name, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote docs/assets/{name}")


def load_prompts():
    out = []
    for meta_path in sorted(TRACES.glob("*/*/meta.json")):
        d = meta_path.parent
        logits = np.load(d / "decode_logits.npy")
        if logits.shape[0] < 32:
            continue
        out.append({
            "dir": d,
            "domain": json.loads(meta_path.read_text())["domain"],
            "logits": logits,           # (T, L, E) fp16
            "tops": np.argsort(-logits.astype(np.float32), axis=-1)[..., :K],  # (T,L,8)
        })
    return out


# ---- 1. popularity skew ------------------------------------------------------
def analyze_skew(prompts, L, E):
    counts = np.zeros((L, E))
    for p in prompts:
        for l in range(L):
            np.add.at(counts[l], p["tops"][:, l, :].ravel(), 1)
    share = np.sort(counts, axis=1)[:, ::-1] / counts.sum(axis=1, keepdims=True)
    cum = np.cumsum(share, axis=1)  # (L, E)

    fig, ax = newfig()
    med = np.median(cum, axis=0)
    lo, hi = np.percentile(cum, 25, axis=0), np.percentile(cum, 75, axis=0)
    x = np.arange(1, E + 1)
    ax.fill_between(x, lo, hi, color=CAT[0], alpha=0.18, linewidth=0)
    ax.plot(x, med, color=CAT[0], linewidth=2)
    for n in (8, 16, 32):
        ax.annotate(f"top {n}: {med[n-1]:.0%}", (n, med[n - 1]),
                    textcoords="offset points", xytext=(8, -12), color=INK, fontsize=10)
        ax.scatter([n], [med[n - 1]], s=24, color=CAT[0], zorder=3)
    ax.set_xlabel(f"experts, most-popular first (of {E})")
    ax.set_ylabel("cumulative share of activations")
    ax.set_title("Expert popularity is long-tailed (median layer, IQR band)")
    ax.set_ylim(0, 1.02)
    savefig(fig, "olmoe-skew.png")
    return {f"top{n}_coverage_median": float(np.median(cum[:, n - 1])) for n in (8, 16, 32)}


# ---- 2. temporal locality ----------------------------------------------------
def analyze_reuse(prompts, L):
    windows = [1, 2, 4, 8, 16, 32]
    reuse = defaultdict(list)  # w -> per-layer means
    for w in windows:
        per_layer = np.zeros(L)
        n = 0
        for p in prompts:
            tops = p["tops"]  # (T,L,8)
            T = tops.shape[0]
            if T <= w:
                continue
            for l in range(L):
                hits = 0
                total = 0
                hist = [set(tops[t, l]) for t in range(T)]
                for t in range(w, T):
                    window_union = set().union(*hist[t - w:t])
                    hits += len(hist[t] & window_union)
                    total += K
                per_layer[l] += hits / total
            n += 1
        reuse[w] = per_layer / n
    fig, ax = newfig()
    med = [np.median(reuse[w]) for w in windows]
    lo = [np.percentile(reuse[w], 25) for w in windows]
    hi = [np.percentile(reuse[w], 75) for w in windows]
    ax.fill_between(windows, lo, hi, color=CAT[0], alpha=0.18, linewidth=0)
    ax.plot(windows, med, color=CAT[0], linewidth=2, marker="o", markersize=5)
    for w, m in zip(windows, med):
        ax.annotate(f"{m:.0%}", (w, m), textcoords="offset points", xytext=(0, 9),
                    color=INK, fontsize=10, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, [str(w) for w in windows])
    ax.set_xlabel("window: previous w tokens")
    ax.set_ylabel("share of current token's experts already used")
    ax.set_title("Temporal locality: expert reuse vs. lookback window")
    ax.set_ylim(0, 1.02)
    savefig(fig, "olmoe-reuse.png")
    return {f"reuse_w{w}_median": float(np.median(reuse[w])) for w in windows}


# ---- 3. cache policy simulation ---------------------------------------------
def sim_cache(seq, capacity, policy, static_order=None):
    """seq: (T,8) expert ids; static_order: experts by global popularity desc."""
    hits = total = 0
    if policy == "static":
        top = set(static_order[:capacity])
        for t in range(seq.shape[0]):
            hits += sum(1 for e in seq[t] if int(e) in top)
            total += K
        return hits / total
    pinned = set()
    cap = capacity
    if policy == "hybrid":
        pinned = set(static_order[: capacity // 2])
        cap = capacity - len(pinned)
    lru = {}  # expert -> last-use clock
    score = defaultdict(float)  # lfu decayed frequency
    clock = 0
    for t in range(seq.shape[0]):
        clock += 1
        if policy == "lfu":
            for e in score:
                score[e] *= 0.98
        for e in seq[t]:
            e = int(e)
            total += 1
            score[e] += 1
            if e in pinned:
                hits += 1
                continue
            if e in lru:
                hits += 1
            else:
                if len(lru) >= cap:
                    if policy == "lfu":
                        victim = min(lru, key=lambda x: score[x])
                    else:
                        victim = min(lru, key=lambda x: lru[x])
                    del lru[victim]
                lru[e] = clock  # miss -> load into cache
            lru[e] = clock
    return hits / total


def analyze_cache(prompts, L, E):
    caps = [8, 12, 16, 24, 32, 48]
    counts = np.zeros((L, E))
    for p in prompts:
        for l in range(L):
            np.add.at(counts[l], p["tops"][:, l, :].ravel(), 1)
    policies = ["lru", "lfu", "static", "hybrid"]
    names = {"lru": "LRU", "lfu": "LFU (decayed)", "static": "static top-C", "hybrid": "pinned+LRU"}
    static_order = [np.argsort(-counts[l]).tolist() for l in range(L)]
    results = {pol: [] for pol in policies}
    for cap in caps:
        for pol in policies:
            rates = []
            for p in prompts:
                for l in range(L):
                    rates.append(sim_cache(p["tops"][:, l, :], cap, pol, static_order[l]))
            results[pol].append(float(np.mean(rates)))
    fig, ax = newfig()
    xs = [c / E for c in caps]
    for i, pol in enumerate(policies):
        ax.plot(xs, results[pol], color=CAT[i], linewidth=2, marker="o", markersize=4)
    # direct labels at line ends, nudged apart to avoid collisions
    ends = sorted(((results[p][-1], i, p) for i, p in enumerate(policies)), reverse=True)
    y_prev = 2.0
    for y, i, pol in ends:
        y_lab = min(y, y_prev - 0.055)
        ax.annotate(names[pol], (xs[-1], y_lab), textcoords="offset points",
                    xytext=(8, 0), color=CAT[i], fontsize=10, va="center")
        y_prev = y_lab
    ax.set_xlabel(f"cache capacity as fraction of experts per layer (E={E})")
    ax.set_ylabel("hit rate")
    ax.set_title("Cache policy hit rates (per-layer caches, per-session reset)")
    ax.set_xlim(right=max(xs) * 1.35)
    ax.set_ylim(0, 1.02)
    savefig(fig, "olmoe-cache.png")
    return {"capacities_frac": xs, **{f"hit_{p}": results[p] for p in policies}}


# ---- 4. gate-ahead predictability ---------------------------------------------
def load_gates(L):
    gates = {}
    for f in sorted((ROOT / "models/olmoe-1b-7b-instruct").glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                if ".mlp.gate.weight" in key:
                    l = int(key.split(".")[2])
                    gates[l] = sf.get_tensor(key).float().numpy()  # (E, H)
    assert len(gates) == L, f"found {len(gates)} gates"
    return gates


def analyze_gate_ahead(prompts, L, gates):
    deltas = list(range(0, 9))
    ms = [8, 16, 32]
    recall = {m: {d: [] for d in deltas} for m in ms}
    for p in prompts:
        hidden = np.load(p["dir"] / "decode_hidden.npy").astype(np.float32)  # (T,L,H)
        tops = p["tops"]
        for d in deltas:
            for l in range(0, L - d):
                pred = hidden[:, l, :] @ gates[l + d].T  # (T,E)
                actual = tops[:, l + d, :]  # (T,8)
                order = np.argsort(-pred, axis=1)
                for m in ms:
                    pm = order[:, :m]  # (T,m)
                    # each actual expert present anywhere in the predicted top-m?
                    found = (pm[:, :, None] == actual[:, None, :]).any(axis=1)  # (T,8)
                    recall[m][d].append(float(found.mean()))
    fig, ax = newfig()
    for i, m in enumerate(ms):
        ys = [float(np.mean(recall[m][d])) for d in deltas]
        ax.plot(deltas, ys, color=SEQ3[i], linewidth=2, marker="o", markersize=4)
        ax.annotate(f"prefetch {m}", (deltas[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), color=SEQ3[i], fontsize=10, va="center")
    ax.set_xlabel("lookahead Δ (apply layer l+Δ's router to layer l's hidden state)")
    ax.set_ylabel("recall of actual top-8")
    ax.set_title("Gate-ahead predictability vs. lookahead (no training)")
    ax.set_xlim(right=deltas[-1] * 1.28)
    ax.set_ylim(0, 1.02)
    savefig(fig, "olmoe-gate-ahead.png")
    return {f"recall_m{m}": {str(d): float(np.mean(recall[m][d])) for d in deltas} for m in ms}


# ---- 5. router margins ---------------------------------------------------------
def analyze_margins(prompts):
    gaps = []
    for p in prompts:
        probs = np.exp(p["logits"].astype(np.float32))
        probs /= probs.sum(-1, keepdims=True)
        srt = np.sort(probs, axis=-1)[..., ::-1]
        gaps.append((srt[..., K - 1] - srt[..., K]).ravel())
    gaps = np.concatenate(gaps)
    fig, ax = newfig()
    ax.hist(np.clip(gaps, 1e-5, 1), bins=np.logspace(-5, 0, 60), color=CAT[0])
    ax.set_xscale("log")
    ax.set_xlabel("router probability gap between 8th and 9th expert")
    ax.set_ylabel("count (token × layer)")
    ax.set_title("Routing margins: how contested is the last expert slot?")
    savefig(fig, "olmoe-margins.png")
    return {
        "frac_gap_below_0.001": float((gaps < 1e-3).mean()),
        "frac_gap_below_0.01": float((gaps < 1e-2).mean()),
        "median_gap": float(np.median(gaps)),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list: skew,reuse,cache,margins,gate_ahead")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    prompts = load_prompts()
    if not prompts:
        print("no traces found")
        return
    T_total = sum(p["tops"].shape[0] for p in prompts)
    L, E = prompts[0]["logits"].shape[1:]
    print(f"{len(prompts)} prompts, {T_total} decode tokens, L={L} E={E}")

    res_path = ROOT / "bench/results/olmoe-analysis.json"
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    out.update({"prompts": len(prompts), "decode_tokens": int(T_total),
                "layers": int(L), "experts": int(E)})
    sections = {
        "skew": lambda: analyze_skew(prompts, L, E),
        "reuse": lambda: analyze_reuse(prompts, L),
        "cache": lambda: analyze_cache(prompts, L, E),
        "margins": lambda: analyze_margins(prompts),
        "gate_ahead": lambda: analyze_gate_ahead(prompts, L, load_gates(L)),
    }
    for name, fn in sections.items():
        if only and name not in only:
            continue
        print(f"{name}...")
        out[name] = fn()

    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(out, indent=1))
    print("saved", res_path)


if __name__ == "__main__":
    main()
