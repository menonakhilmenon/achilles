#!/usr/bin/env python
"""Does adding the previous token's expert trace as a feature improve
within-token lookahead probes? (hidden-only vs hidden + prev-token multihot)

For target layer tgt and lookahead DELTA: predict topk[t, tgt] from
  A: h[t, tgt-DELTA]                          (current probes)
  B: h[t, tgt-DELTA] ++ multihot(topk[t-1, tgt])   (trace-enriched)
"""
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DELTA = 8
E, H = 256, 6144
EPOCHS = 20


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


def recall_at(scores, Y, m):
    order = np.argsort(-scores, 1)[:, :m]
    hits = sum(len(set(order[i]) & set(ids.tolist())) for i, ids in enumerate(Y))
    return hits / (len(Y) * 8)


def train_eval(Xtr, Ytr, Xte, Yte, dim):
    Xt = torch.from_numpy(np.stack(Xtr))
    Yt = torch.zeros(len(Ytr), E)
    for i, ids in enumerate(Ytr):
        Yt[i, ids] = 1.0
    lin = nn.Linear(dim, E)
    opt = torch.optim.Adam(lin.parameters(), lr=3e-4)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        sc = lin(torch.from_numpy(np.stack(Xte))).numpy()
    return recall_at(sc, Yte, 8), recall_at(sc, Yte, 16)


def main():
    torch.set_num_threads(12)
    dumps = [parse_dump(d) for d in sorted((ROOT / "traces/glm52").glob("dump*.bin"))]
    tr_dumps, te_dumps = dumps[:-1], dumps[-1:]
    moe = sorted(l for l in dumps[0][2] if dumps[0][2][l][1] is not None)
    targets = [l for l in moe if l - DELTA in dumps[0][2]][10::12][:5]
    print(f"targets: {targets}, delta={DELTA}")

    for tgt in targets:
        src = tgt - DELTA
        XA, XB, Y, XAte, XBte, Yte = [], [], [], [], [], []
        for group, (xa, xb, yy) in ((tr_dumps, (XA, XB, Y)), (te_dumps, (XAte, XBte, Yte))):
            for dp in group:
                for prev, curp in zip(dp, dp[1:]):
                    h = curp.get(src, [None, None])[0]
                    y = curp.get(tgt, [None, None])[1]
                    py = prev.get(tgt, [None, None])[1]
                    if h is None or y is None or py is None:
                        continue
                    mh = np.zeros(E, np.float32)
                    mh[py] = 1.0
                    xa.append(h)
                    xb.append(np.concatenate([h, mh]))
                    yy.append(y)
        if len(Y) < 400:
            print(f"tgt {tgt}: insufficient")
            continue
        a8, a16 = train_eval(XA, Y, XAte, Yte, H)
        b8, b16 = train_eval(XB, Y, XBte, Yte, H + E)
        print(f"tgt {tgt}: hidden-only r@8 {a8:.3f} r@16 {a16:.3f}  |  "
              f"+prev-trace r@8 {b8:.3f} r@16 {b16:.3f}")


if __name__ == "__main__":
    main()
