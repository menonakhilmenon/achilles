#!/usr/bin/env python
"""Paging simulator: replay routing traces against a two-tier (RAM <- SSD)
CPU-inference model and predict decode throughput.

Timeline model per decode token, layer by layer:
  compute stream : layer_time = (dense_bytes_per_layer + k*expert_bytes) / ram_bw
                   (CPU decode is memory-bound; resident-weight reads at ram_bw)
  I/O stream     : FIFO queue served at ssd_bw with per-op latency ssd_lat;
                   runs concurrently with compute (DMA), but degrades ram_bw by
                   `interference` while busy.
  prefetch       : at layer l-DELTA the predictor emits top-m candidates for
                   layer l; those absent from RAM are enqueued.
  demand miss    : needed expert absent at layer start -> stall until loaded.
  cache          : global expert cache (layer,expert) keys, capacity in bytes,
                   LRU or hybrid (half pinned to globally-hot experts).

Predictor emulation: each actually-needed expert is independently in the
predicted set with probability `recall`; the prefetch bandwidth cost includes
overshoot: m predicted per layer cost m loads' worth of queue occupancy for
the ones not cached.

Usage:
  sim_paging.py --traces traces/olmoe --expert-mib 6.5 --dense-mib 40 \
      --ram-gb 2 --ram-bw 55e9 --ssd-bw 6.5e9 --recall 0.85 --delta 4 --m 12
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


class LRU:
    def __init__(self, capacity, pinned=()):
        self.cap = max(capacity - len(pinned), 0)
        self.pinned = set(pinned)
        self.d = OrderedDict()

    def __contains__(self, k):
        return k in self.pinned or k in self.d

    def touch(self, k):
        if k in self.d:
            self.d.move_to_end(k)

    def insert(self, k):
        if k in self.pinned:
            return
        if k in self.d:
            self.d.move_to_end(k)
            return
        while len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[k] = True


def load_traces(root):
    """Yield (T, L, k) decode top-k arrays per prompt (OLMoE or parsed Qwen)."""
    seqs = []
    for f in sorted(Path(root).glob("**/decode_logits.npy")):
        a = np.load(f).astype(np.float32)
        if a.shape[0] < 32:
            continue
        seqs.append(np.argsort(-a, axis=-1)[..., :8].astype(np.int32))
    for f in sorted(Path(root).glob("**/*.decode_topk.npy")):
        a = np.load(f)
        if a.shape[0] >= 32:
            seqs.append(a.astype(np.int32))
    return seqs


def simulate(seq, args, rng, pop_order):
    T, L, k = seq.shape
    expert_b = args.expert_mib * 2**20
    dense_b = args.dense_mib * 2**20
    cap = int(args.ram_gb * 2**30 / expert_b)
    pinned = []
    if args.policy == "hybrid":
        pinned = pop_order[: cap // 2]
    cache = LRU(cap, pinned)
    # warm start: fill with pinned + most popular (cold-start measured separately)
    for key in pop_order[cap // 2 : cap]:
        cache.insert(key)

    io_free_at = 0.0          # when the SSD queue drains
    inflight = {}             # (l,e) -> completion time
    t_clock = 0.0
    stall_time = load_bytes = 0.0
    hits = need = 0
    tok_times = []

    def enqueue(key, now):
        nonlocal io_free_at, load_bytes
        start = max(io_free_at, now)
        done = start + args.ssd_lat + expert_b / args.ssd_bw
        io_free_at = done
        inflight[key] = done
        load_bytes += expert_b
        return done

    for t in range(T):
        tok_start = t_clock
        for l in range(L):
            now = t_clock
            # 1. prefetch for layer l+delta
            fl = l + args.delta
            if fl < L:
                actual = seq[t, fl]
                predicted = [e for e in actual if rng.random() < args.recall]
                # overshoot: predictor emits m total; extras are wasted loads
                extras = max(args.m - len(actual), 0)
                for e in predicted:
                    key = (fl, int(e))
                    if key not in cache and key not in inflight:
                        enqueue(key, now)
                for _ in range(extras):
                    if rng.random() < args.waste:  # wasted prefetches that queue
                        io_free_at = max(io_free_at, now) + expert_b / args.ssd_bw
                        load_bytes += expert_b

            # 2. land any completed loads
            for key, done in list(inflight.items()):
                if done <= now:
                    cache.insert(key)
                    del inflight[key]

            # 3. demand misses for this layer
            for e in seq[t, l]:
                key = (l, int(e))
                need += 1
                if key in cache:
                    hits += 1
                    cache.touch(key)
                    continue
                done = inflight.pop(key, None) or enqueue(key, now)
                if done > now:
                    stall_time += done - now
                    now = done
                cache.insert(key)
                inflight.pop(key, None)

            # 4. compute for this layer (slower while SSD is busy)
            bw = args.ram_bw * (args.interference if io_free_at > now else 1.0)
            now += (dense_b + k * expert_b) / bw
            t_clock = now
        tok_times.append(t_clock - tok_start)

    tok_times = np.array(tok_times)
    return {
        "tok_s": 1.0 / tok_times.mean(),
        "p50_ms": float(np.percentile(tok_times, 50) * 1e3),
        "p99_ms": float(np.percentile(tok_times, 99) * 1e3),
        "hit_rate": hits / need,
        "stall_frac": stall_time / t_clock if t_clock else 0,
        "ssd_GB_per_tok": load_bytes / T / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces/olmoe")
    ap.add_argument("--expert-mib", type=float, required=True)
    ap.add_argument("--dense-mib", type=float, required=True,
                    help="resident non-expert bytes read per layer per token")
    ap.add_argument("--ram-gb", type=float, required=True, help="expert cache size")
    ap.add_argument("--ram-bw", type=float, default=55e9)
    ap.add_argument("--ssd-bw", type=float, default=6.5e9)
    ap.add_argument("--ssd-lat", type=float, default=0.0002)
    ap.add_argument("--interference", type=float, default=0.7,
                    help="ram_bw multiplier while SSD queue is busy (measured)")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--delta", type=int, default=4)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--waste", type=float, default=0.5,
                    help="fraction of overshoot prefetches that actually queue")
    ap.add_argument("--policy", choices=["lru", "hybrid"], default="hybrid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs = load_traces(args.traces)
    if not seqs:
        raise SystemExit("no traces found")
    L = seqs[0].shape[1]
    counts = {}
    for s in seqs:
        for l in range(L):
            ids, n = np.unique(s[:, l, :], return_counts=True)
            for e, c in zip(ids, n):
                counts[(l, int(e))] = counts.get((l, int(e)), 0) + int(c)
    pop_order = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    rng = np.random.default_rng(args.seed)
    results = [simulate(s, args, rng, pop_order) for s in seqs]
    agg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    agg["prompts"] = len(results)
    agg["config"] = {k: v for k, v in vars(args).items() if k != "traces"}
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
