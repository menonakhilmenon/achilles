#!/usr/bin/env python
"""Does the decode expert working set vary by prompt? (validation for a
prompt-conditioned usage-prior predictor)

Inputs: traces/qwen-prompts/pNN.bin (arena --dump; T records during decode).

A) Divergence: per-layer top-K working-set Jaccard, cross-prompt vs
   split-half self-consistency, plus usage-vector cosines. If self >> cross,
   prompt-specific structure exists worth predicting.
B) Oracle eviction sim (gap/pop policy on each prompt's stream):
   - LRU
   - GDLRU online EWMA (deployed --policy reuse)
   - GDLRU + global prior (leave-one-out mean usage, online updates continue)
   - GDLRU + oracle prior (this prompt's true usage, frozen) <- upper bound
   Report overall and second-half hit rates (short streams are
   compulsory-miss dominated early).
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_EXPERT = 128
TOPK_JACCARD = 32


def parse_stream(path):
    """Returns (decode_stream, prefill_usage_vector_or_None)."""
    data = path.read_bytes()
    off = 0
    stream = []
    pre = None
    while off < len(data):
        tag = data[off:off + 1]
        l, n = struct.unpack_from("<ii", data, off + 1)
        off += 9
        if tag == b"H":
            off += 4 * n
        elif tag == b"T":
            ids = struct.unpack_from(f"<{n}i", data, off)
            off += 4 * n
            for e in ids:
                stream.append(l * N_EXPERT + e)
        elif tag == b"P":
            ids = struct.unpack_from(f"<{n}i", data, off)
            off += 4 * n
            if pre is None:
                pre = np.zeros(64 * N_EXPERT)
            for e in ids:
                if 0 <= e < N_EXPERT:
                    pre[l * N_EXPERT + e] += 1
        else:
            raise ValueError(f"bad tag in {path}")
    return stream, pre


def usage_matrix(stream, n_layer):
    u = np.zeros(n_layer * N_EXPERT)
    for k in stream:
        u[k] += 1
    return u


def jaccard_topk(u1, u2, n_layer, k=TOPK_JACCARD):
    js = []
    for l in range(n_layer):
        a = set(np.argsort(u1[l * N_EXPERT:(l + 1) * N_EXPERT])[-k:])
        b = set(np.argsort(u2[l * N_EXPERT:(l + 1) * N_EXPERT])[-k:])
        js.append(len(a & b) / len(a | b))
    return float(np.mean(js))


def sim_lru(stream, cap):
    from collections import OrderedDict
    c = OrderedDict()
    h1 = h2 = 0
    half = len(stream) // 2
    for i, k in enumerate(stream):
        if k in c:
            if i < half:
                h1 += 1
            else:
                h2 += 1
            c.move_to_end(k)
        else:
            if len(c) >= cap:
                c.popitem(last=False)
            c[k] = True
    return (h1 + h2) / len(stream), h2 / (len(stream) - half)


def sim_gdlru(stream, cap, n_layer, prior=None, frozen=False, alpha=0.02):
    visits = np.zeros(n_layer)
    key_arr = np.full(cap, -1, dtype=np.int64)
    layer_arr = np.zeros(cap, dtype=np.int64)
    last_arr = np.zeros(cap)
    p_arr = np.full(cap, 1e-4)
    pup_arr = np.zeros(cap)
    slot_of = {}
    free = list(range(cap))
    h1 = h2 = 0
    half = len(stream) // 2
    prev = None
    for i, k in enumerate(stream):
        l = k // N_EXPERT
        if l != prev:
            visits[l] += 1
            prev = l
        s = slot_of.get(k)
        if s is not None:
            if i < half:
                h1 += 1
            else:
                h2 += 1
        else:
            if free:
                s = free.pop()
            else:
                gap = visits[layer_arr] - last_arr
                pd = p_arr if frozen else p_arr * (1.0 - alpha) ** (visits[layer_arr] - pup_arr)
                score = gap / np.maximum(pd, 1e-4)
                s = int(np.argmax(score))
                del slot_of[int(key_arr[s])]
            slot_of[k] = s
            key_arr[s] = k
            layer_arr[s] = l
            p_arr[s] = prior[k] if prior is not None else 1e-4
            pup_arr[s] = visits[l] - 1
        if not frozen:
            gp = visits[l] - 1 - pup_arr[s]
            p_arr[s] = (1.0 - alpha) ** (gp + 1) * p_arr[s] + alpha
            pup_arr[s] = visits[l]
        last_arr[s] = visits[l]
    return (h1 + h2) / len(stream), h2 / (len(stream) - half)


def main():
    files = sorted((ROOT / "traces/qwen-prompts").glob("p*.bin"))
    parsed = {f.stem: parse_stream(f) for f in files}
    streams = {k: v[0] for k, v in parsed.items() if len(v[0]) > 10000}
    prefills = {k: parsed[k][1] for k in streams}
    names = sorted(streams)
    n_layer = max(k // N_EXPERT for v in streams.values() for k in v) + 1
    print(f"{len(names)} prompts, n_layer={n_layer}, "
          f"activations/prompt ~{np.mean([len(v) for v in streams.values()]):.0f}")

    U = {p: usage_matrix(streams[p], n_layer) for p in names}
    # A) divergence
    cross = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross.append(jaccard_topk(U[names[i]], U[names[j]], n_layer))
    selfs = []
    coss_self, coss_cross = [], []
    for p in names:
        st = streams[p]
        h = len(st) // 2
        u1 = usage_matrix(st[:h], n_layer)
        u2 = usage_matrix(st[h:], n_layer)
        selfs.append(jaccard_topk(u1, u2, n_layer))
        coss_self.append(float(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-9)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = U[names[i]], U[names[j]]
            coss_cross.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    print(f"\nA) top-{TOPK_JACCARD} working-set Jaccard: "
          f"cross-prompt {np.mean(cross):.3f}  vs  split-half self {np.mean(selfs):.3f}")
    print(f"   usage cosine:                 cross-prompt {np.mean(coss_cross):.3f}  "
          f"vs  split-half self {np.mean(coss_self):.3f}")
    if all(prefills[p] is not None for p in names):
        cos_pre = []
        for p in names:
            a = prefills[p][:n_layer * N_EXPERT]
            b = U[p]
            cos_pre.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        print(f"   prefill-usage -> decode-usage cosine: {np.mean(cos_pre):.3f} "
              f"(free zero-parameter prior)")

    # B) oracle sims
    total = n_layer * N_EXPERT
    for frac in (0.15, 0.25, 0.31):
        cap = int(total * frac)
        rows = {"LRU": [], "GDLRU online": [], "GDLRU +global": [],
                "GDLRU +prefill": [], "GDLRU +oracle": []}
        for p in names:
            st = streams[p]
            n_tok = len(st) / (n_layer * 8)
            loo = np.mean([U[q] for q in names if q != p], axis=0) / n_tok
            oracle = U[p] / n_tok
            rows["LRU"].append(sim_lru(st, cap))
            rows["GDLRU online"].append(sim_gdlru(st, cap, n_layer))
            rows["GDLRU +global"].append(sim_gdlru(st, cap, n_layer, prior=np.clip(loo, 1e-4, 1)))
            if prefills[p] is not None:
                pv = prefills[p][:n_layer * N_EXPERT]
                n_pre = pv.sum() / (n_layer * 8)
                pre = pv / max(n_pre, 1.0)
                rows["GDLRU +prefill"].append(sim_gdlru(st, cap, n_layer, prior=np.clip(pre, 1e-4, 1)))
            rows["GDLRU +oracle"].append(sim_gdlru(st, cap, n_layer, prior=np.clip(oracle, 1e-4, 1), frozen=True))
        print(f"\nB) cap {cap} slots ({frac:.0%}):   overall / 2nd-half hit")
        for name, vals in rows.items():
            if not vals:
                continue
            o = np.mean([v[0] for v in vals])
            h2 = np.mean([v[1] for v in vals])
            print(f"   {name:14s} {o:.4f} / {h2:.4f}")


if __name__ == "__main__":
    main()
