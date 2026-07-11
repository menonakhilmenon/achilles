#!/usr/bin/env python
"""Analyze Qwen3-30B-A3B expert-ID traces (from trace-moe): skew, locality,
cache policies, margins. No hidden states -> no gate-ahead section.

Reads traces/qwen3-30b/<domain>-<id>.decode_topk.npy (+ .decode_probs.npy).
Writes docs/assets/qwen3-30b-*.png and bench/results/qwen3-30b-analysis.json.
"""
import json
from pathlib import Path

import numpy as np

import analyze_olmoe as A

A.PREFIX = "qwen3-30b"
ROOT = A.ROOT
TRACES = ROOT / "traces/qwen3-30b"


def load_prompts():
    out = []
    for f in sorted(TRACES.glob("*.decode_topk.npy")):
        tops = np.load(f).astype(np.int64)
        if tops.shape[0] < 32:
            continue
        p = {"dir": f.parent, "domain": f.name.split("-")[0], "tops": tops}
        probs_f = f.with_name(f.name.replace("decode_topk", "decode_probs"))
        if probs_f.exists():
            p["probs"] = np.load(probs_f)  # (T, L, E) fp16, already normalized
        out.append(p)
    return out


def analyze_margins(prompts):
    gaps = []
    for p in prompts:
        if "probs" not in p:
            continue
        pr = p["probs"].astype(np.float32)
        srt = np.sort(pr, axis=-1)[..., ::-1]
        gaps.append((srt[..., A.K - 1] - srt[..., A.K]).ravel())
    if not gaps:
        return {}
    gaps = np.concatenate(gaps)
    fig, ax = A.newfig()
    ax.hist(np.clip(gaps, 1e-5, 1), bins=np.logspace(-5, 0, 60), color=A.CAT[0])
    ax.set_xscale("log")
    ax.set_xlabel("router probability gap between 8th and 9th expert")
    ax.set_ylabel("count (token × layer)")
    ax.set_title("Routing margins: how contested is the last expert slot?")
    A.savefig(fig, f"{A.PREFIX}-margins.png")
    return {
        "frac_gap_below_0.001": float((gaps < 1e-3).mean()),
        "frac_gap_below_0.01": float((gaps < 1e-2).mean()),
        "median_gap": float(np.median(gaps)),
    }


def main():
    prompts = load_prompts()
    if not prompts:
        raise SystemExit("no qwen traces found")
    T_total = sum(p["tops"].shape[0] for p in prompts)
    L, E = prompts[0]["tops"].shape[1], 128
    print(f"{len(prompts)} prompts, {T_total} decode tokens, L={L} E={E}")

    out = {"prompts": len(prompts), "decode_tokens": int(T_total), "layers": L, "experts": E}
    print("skew...");    out["skew"] = A.analyze_skew(prompts, L, E)
    print("reuse...");   out["reuse"] = A.analyze_reuse(prompts, L)
    print("cache...");   out["cache"] = A.analyze_cache(prompts, L, E)
    print("margins..."); out["margins"] = analyze_margins(prompts)

    path = ROOT / "bench/results/qwen3-30b-analysis.json"
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
