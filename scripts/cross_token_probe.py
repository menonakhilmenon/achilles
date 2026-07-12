#!/usr/bin/env python
"""Cross-token expert predictability on GLM-5.2: can token t's hidden state
predict token t+1's expert needs across ALL layers?

If recall is decent, the arena can issue a whole-token fetch plan at token
start -> continuous SSD streaming instead of per-layer bursts.

Source layer: a mid/late layer of token t (sweep a few). Targets: every MoE
layer of token t+1. Linear probes, prompt-level holdout. Also reports the
zero-training baseline: "predict token t+1 uses exactly token t's experts"
(the reuse heuristic the LRU cache already exploits).
"""
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SRC_LAYERS = [20, 45, 74]   # candidate source layers within token t
EPOCHS = 20
E, H = 256, 6144


def parse_dump(path):
    data = path.read_bytes()
    off = 0
    passes, cur, last_l = [], {}, 1 << 30
    while off < len(data):
        tag = data[off:off + 1]
        l, n = struct.unpack_from("<ii", data, off + 1)
        off += 9
        if tag == b"H":
            arr = np.frombuffer(data, "<f4", n, off).copy()
            off += 4 * n
            if l < last_l and cur:
                passes.append(cur)
                cur = {}
            last_l = l
            cur.setdefault(l, [None, None])[0] = arr
        else:
            ids = np.frombuffer(data, "<i4", n, off).copy()
            off += 4 * n
            cur.setdefault(l, [None, None])[1] = ids
    if cur:
        passes.append(cur)
    return passes


def main():
    torch.set_num_threads(12)
    dumps = [parse_dump(d) for d in sorted((ROOT / "traces/glm52").glob("dump*.bin"))]
    print(f"dumps: {[len(d) for d in dumps]} passes")

    # pairs (t, t+1) within each dump; holdout = last dump
    def pairs(dump_list):
        out = []
        for dp in dump_list:
            for a, b in zip(dp, dp[1:]):
                out.append((a, b))
        return out

    train_pairs = pairs(dumps[:-1])
    test_pairs = pairs(dumps[-1:])
    print(f"{len(train_pairs)} train / {len(test_pairs)} test pairs")

    moe_layers = sorted(l for l in dumps[0][0] if dumps[0][0][l][1] is not None)

    # zero-training baseline: token t's experts as the prediction for t+1
    reuse = []
    for a, b in test_pairs:
        per = []
        for l in moe_layers:
            if a.get(l, [None, None])[1] is None or b.get(l, [None, None])[1] is None:
                continue
            sa, sb = set(a[l][1].tolist()), set(b[l][1].tolist())
            per.append(len(sa & sb) / len(sb))
        reuse.append(np.mean(per))
    print(f"baseline (t's experts as plan): recall {np.mean(reuse):.3f}")

    for src in SRC_LAYERS:
        # train one shared probe per target layer: h[t, src] -> topk[t+1, tgt]
        Xtr = [a[src][0] for a, b in train_pairs if src in a and a[src][0] is not None]
        if len(Xtr) < 400:
            print(f"src={src}: insufficient hidden samples, skipping")
            continue
        rec8, rec16 = [], []
        for tgt in moe_layers[::6]:  # sample every 6th layer for speed
            X, Y, Xte, Yte = [], [], [], []
            for pp, dst in ((train_pairs, (X, Y)), (test_pairs, (Xte, Yte))):
                for a, b in pp:
                    ha = a.get(src, [None, None])[0]
                    yb = b.get(tgt, [None, None])[1]
                    if ha is None or yb is None:
                        continue
                    dst[0].append(ha)
                    dst[1].append(yb)
            if len(X) < 400 or len(Xte) < 50:
                continue
            Xt = torch.from_numpy(np.stack(X))
            Yt = torch.zeros(len(Y), E)
            for i, ids in enumerate(Y):
                Yt[i, ids] = 1.0
            lin = nn.Linear(H, E)
            opt = torch.optim.Adam(lin.parameters(), lr=3e-4)
            for _ in range(EPOCHS):
                perm = torch.randperm(len(Xt))
                for i in range(0, len(perm), 512):
                    bidx = perm[i:i + 512]
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt[bidx]), Yt[bidx])
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                sc = lin(torch.from_numpy(np.stack(Xte))).numpy()
            order = np.argsort(-sc, 1)
            for m, acc in ((8, rec8), (16, rec16)):
                top = order[:, :m]
                hits = sum(len(set(top[i]) & set(ids.tolist())) for i, ids in enumerate(Yte))
                acc.append(hits / (len(Yte) * 8))
        print(f"src layer {src}: trained cross-token recall@8 {np.mean(rec8):.3f}, "
              f"recall@16 {np.mean(rec16):.3f} (over {len(rec8)} target layers)")


if __name__ == "__main__":
    main()
