#!/usr/bin/env python
"""Train GLM-5.2 gate-ahead probes from arena --dump traces.

Dump format (per decode forward pass, interleaved per layer):
  'H' i32 l i32 n  f32[n]   -- router-input hidden state at layer l
  'T' i32 l i32 k  i32[k]   -- actual top-k expert ids at layer l

Trains, for each source layer l with l+DELTA a MoE layer, a linear probe
W (E x H): hidden at l -> top-8 experts at l+DELTA. Warm-started from the
real router of l+DELTA (gate-ahead baseline), fine-tuned with BCE on the
actual routing (which bakes in sigmoid+bias/noaux effects).

Output blob: i32 n_layer, n_expert, n_embd, then n_layer * E * H f32
(zeros for layers without a trained probe). Loaded via achilles-arena --probe.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DELTA = 3
EPOCHS = 30
HOLDOUT = 0.15


def parse_dump(path):
    data = path.read_bytes()
    off = 0
    passes = []          # list of dict: l -> (hidden, ids)
    cur = {}
    last_l = 1 << 30
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
        elif tag == b"T":
            ids = np.frombuffer(data, "<i4", n, off).copy()
            off += 4 * n
            cur.setdefault(l, [None, None])[1] = ids
        else:
            raise ValueError(f"bad tag {tag} at {off - 9}")
    if cur:
        passes.append(cur)
    return passes


def load_routers():
    """Router weights for warm start, from shard GGUFs via arena-identical logic."""
    sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
    from gguf import GGUFReader
    routers, biases = {}, {}
    for f in sorted((ROOT / "models/glm52-gguf/UD-Q2_K_XL").glob("*.gguf")):
        r = GGUFReader(str(f))
        for t in r.tensors:
            if t.name.endswith("ffn_gate_inp.weight"):
                l = int(t.name.split(".")[1])
                routers[l] = np.array(t.data, dtype=np.float32).reshape(t.shape[1], t.shape[0]) \
                    if len(t.shape) == 2 else np.array(t.data, dtype=np.float32)
    return routers


def main():
    dumps = sorted((ROOT / "traces/glm52").glob("dump*.bin"))
    passes = []
    for d in dumps:
        passes.extend(parse_dump(d))
    print(f"{len(passes)} forward passes from {len(dumps)} dumps")

    # assemble per source layer: X = hidden[l], Y = ids[l+DELTA]
    layers = sorted({l for p in passes for l in p})
    L_max = max(layers) + 1
    E, H = 256, 6144
    routers = load_routers()
    print(f"routers loaded: {len(routers)}")

    blob = np.zeros((L_max, E, H), np.float32)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(passes))
    n_te = max(int(len(passes) * HOLDOUT), 1)
    te_set = set(order[:n_te].tolist())

    trained = 0
    ga_all, pr_all = [], []
    for l in layers:
        tgt = l + DELTA
        X, Y, Xte, Yte = [], [], [], []
        for pi, p in enumerate(passes):
            if l not in p or tgt not in p:
                continue
            h, _ = p[l]
            _, ids = p[tgt]
            if h is None or ids is None:
                continue
            (Xte if pi in te_set else X).append(h)
            (Yte if pi in te_set else Y).append(ids)
        if len(X) < 200 or tgt not in routers or len(Xte) < 20:
            continue
        Xt = torch.from_numpy(np.stack(X))
        Yt = torch.zeros(len(Y), E)
        for i, ids in enumerate(Y):
            Yt[i, ids] = 1.0
        W0 = routers[tgt]
        if W0.size != E * H:
            continue
        lin = nn.Linear(H, E)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W0.reshape(E, H)))
            lin.bias.zero_()
        opt = torch.optim.Adam(lin.parameters(), lr=2e-4)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
        # held-out recall@fetch vs gate-ahead warm start
        Xv = torch.from_numpy(np.stack(Xte))
        with torch.no_grad():
            pr = lin(Xv).numpy()
        ga = np.stack(Xte) @ W0.reshape(E, H).T
        m = 8
        def recall(scores):
            top = np.argsort(-scores, 1)[:, :m]
            hits = 0
            for i, ids in enumerate(Yte):
                hits += len(set(top[i]) & set(ids.tolist()))
            return hits / (len(Yte) * len(Yte[0]))
        r_pr, r_ga = recall(pr), recall(ga)
        ga_all.append(r_ga); pr_all.append(r_pr)
        if r_pr < r_ga - 0.01:
            continue  # probe regressed on holdout; layer keeps no probe
        with torch.no_grad():
            blob[l] = lin.weight.numpy()
        trained += 1
        if l % 10 == 0:
            print(f"l={l}: gate-ahead {r_ga:.3f} -> probe {r_pr:.3f} ({len(X)} samples)")

    print(f"\ntrained {trained} probes; mean recall@8 delta={DELTA}: "
          f"gate-ahead {np.mean(ga_all):.3f} -> probe {np.mean(pr_all):.3f}")
    out = ROOT / "models/glm52-probes-d3.bin"
    with open(out, "wb") as f:
        f.write(struct.pack("<iii", L_max, E, H))
        f.write(blob.tobytes())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
