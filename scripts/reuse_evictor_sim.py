#!/usr/bin/env python
"""Learned/structured reuse-distance eviction vs LRU and Belady (GLM-5.2 streams).

Policy idea (phase + geometric): an expert (l, e) can only be used when layer l
runs — once per pass (token/prefill sweep). Estimated next use in layer-steps:

    est_next_use(l, e) = next_visit(l) + n_layer * (1 - p) / p

where next_visit is the absolute layer-step when layer l next comes around and
p is an EWMA of "activated when its layer ran" (lazily decayed). Evict the max.
Belady with p perfectly known and within-token knowledge is the upper bound.

Variants:
  PHASE      p ignored (p=1): evict the expert whose layer just ran (max cyclic dist)
  REUSE      the full formula with EWMA alpha
Run at multiple caps; stream from traces/glm52 dumps (arena --dump T records).
"""
import heapq
import struct
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_EXPERT = 256


def parse_topk(path, limit=None):
    data = path.read_bytes()
    off = 0
    stream = []
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
            if limit and len(stream) >= limit:
                return stream
        else:
            raise ValueError("bad tag")
    return stream


def layers_of(stream):
    return sorted({k // N_EXPERT for k in stream})


def sim_lru(stream, cap, **_):
    c = OrderedDict()
    hits = 0
    for k in stream:
        if k in c:
            hits += 1
            c.move_to_end(k)
        else:
            if len(c) >= cap:
                c.popitem(last=False)
            c[k] = True
    return hits / len(stream)


def sim_belady(stream, cap, **_):
    nxt = defaultdict(list)
    for i in range(len(stream) - 1, -1, -1):
        nxt[stream[i]].append(i)
    INF = 1 << 60
    c = set()
    cur_nu = {}
    heap = []
    hits = 0
    for k in stream:
        nxt[k].pop()
        nu = nxt[k][-1] if nxt[k] else INF
        if k in c:
            hits += 1
        else:
            if len(c) >= cap:
                while True:
                    negnu, vk = heapq.heappop(heap)
                    if vk in c and cur_nu.get(vk) == -negnu:
                        c.remove(vk)
                        del cur_nu[vk]
                        break
            c.add(k)
        cur_nu[k] = nu
        heapq.heappush(heap, (-nu, k))
    return hits / len(stream)


def sim_plan_lru(stream, cap, **_):
    """Deployed arena policy, idealized: keys with a known future use in the
    CURRENT pass are protected (rolling plan); otherwise LRU."""
    # precompute pass boundaries: a pass restarts when layer index decreases
    pass_id = []
    cur = 0
    prev_l = 1 << 30
    for k in stream:
        l = k // N_EXPERT
        if l < prev_l:
            cur += 1
        prev_l = l
        pass_id.append(cur)
    # next use index per position
    nxt = defaultdict(list)
    for i in range(len(stream) - 1, -1, -1):
        nxt[stream[i]].append(i)
    c = OrderedDict()
    hits = 0
    for i, k in enumerate(stream):
        nxt[k].pop()
        if k in c:
            hits += 1
            c.move_to_end(k)
        else:
            if len(c) >= cap:
                victim = None
                for cand in c:   # LRU order; skip current-pass-planned
                    nu = nxt[cand][-1] if nxt[cand] else None
                    if nu is not None and pass_id[nu] == pass_id[i]:
                        continue
                    victim = cand
                    break
                if victim is None:
                    victim = next(iter(c))
                del c[victim]
            c[k] = True
    return hits / len(stream)


def sim_markov(stream, cap, alpha=0.05, plan=False):
    """est_next_use = d + (1 - q1) * L / p   (lazy max-heap).

    q1 = EWMA of 'activated again at the very next visit of its layer'
    p  = EWMA of 'activated when its layer ran' (lazily decayed)
    With plan=True, keys with a known future use in the current pass are
    protected (matches the arena's plan_hint) via est = now.
    """
    layers = layers_of(stream)
    lpos = {l: i for i, l in enumerate(layers)}
    L = len(layers)
    p = {}
    q1 = {}
    lastact = {}     # key -> layer-visit index of last activation
    visits = defaultdict(int)
    pass_id = []
    cur = 0
    prev_l = 1 << 30
    for k in stream:
        l = k // N_EXPERT
        if l < prev_l:
            cur += 1
        prev_l = l
        pass_id.append(cur)
    nxt = defaultdict(list)
    if plan:
        for i in range(len(stream) - 1, -1, -1):
            nxt[stream[i]].append(i)
    c = set()
    cur_est = {}
    heap = []
    hits = 0
    t = 0
    prev = None
    INF = 1 << 60

    def est_next_use(k, i):
        l = k // N_EXPERT
        if plan:
            nu = nxt[k][-1] if nxt[k] else None
            if nu is not None and pass_id[nu] == pass_id[i]:
                return t  # planned this token: evict last
        d = (lpos[l] - (t % L)) % L
        if d == 0:
            d = L
        ew, at = p.get(k, (0.0, visits[l]))
        gap = visits[l] - at
        if gap > 0:
            ew *= (1.0 - alpha) ** gap
        ew = max(ew, 1e-4)
        q = q1.get(k, 0.3)
        return t + d + (1.0 - q) * L / ew

    for i, k in enumerate(stream):
        l = k // N_EXPERT
        if l != prev:
            t += 1
            visits[l] += 1
            prev = l
        if plan:
            nxt[k].pop()
        # p update (activation at this visit)
        ew, at = p.get(k, (0.0, visits[l] - 1))
        gap = visits[l] - 1 - at
        if gap > 0:
            ew *= (1.0 - alpha) ** gap
        p[k] = ((1.0 - alpha) * ew + alpha, visits[l])
        # q1 update: was the gap since last activation exactly 1 visit?
        la = lastact.get(k)
        if la is not None:
            reused_next = 1.0 if visits[l] - la == 1 else 0.0
            q1[k] = (1.0 - alpha) * q1.get(k, 0.3) + alpha * reused_next
        lastact[k] = visits[l]
        if k in c:
            hits += 1
        else:
            if len(c) >= cap:
                while True:
                    negest, vk = heapq.heappop(heap)
                    if vk in c and cur_est.get(vk) == -negest:
                        c.remove(vk)
                        del cur_est[vk]
                        break
            c.add(k)
        e = est_next_use(k, i)
        cur_est[k] = e
        heapq.heappush(heap, (-e, k))
    return hits / len(stream)


def sim_gdlru(stream, cap, alpha=0.02, pexp=1.0, plan=False):
    """Evict max (visit-gap / p^pexp): LRU stretched by inverse popularity.
    Hazard analysis (scripts/reuse_hazard.py): h(g) monotone decreasing and
    ~proportional to popularity at every gap -> expected wait ~ g / p.
    Linear victim scan vectorized with numpy. plan=True: protect keys with a
    known future use in the current pass (arena plan_hint idealization)."""
    import numpy as np
    layers = layers_of(stream)
    nl = max(layers) + 1
    visits = np.zeros(nl, dtype=np.float64)
    # slot arrays
    key_arr = np.full(cap, -1, dtype=np.int64)
    layer_arr = np.zeros(cap, dtype=np.int64)
    last_arr = np.zeros(cap, dtype=np.float64)     # visits[l] at last activation
    p_arr = np.full(cap, 1e-4, dtype=np.float64)   # EWMA popularity
    pup_arr = np.zeros(cap, dtype=np.float64)      # visits[l] at last p update
    plan_arr = np.zeros(cap, dtype=np.int64)       # pass id protected through
    slot_of = {}
    free = list(range(cap))
    pass_id = []
    cur = 0
    prev_l = 1 << 30
    for k in stream:
        l = k // N_EXPERT
        if l < prev_l:
            cur += 1
        prev_l = l
        pass_id.append(cur)
    nxt = defaultdict(list)
    if plan:
        for i in range(len(stream) - 1, -1, -1):
            nxt[stream[i]].append(i)
    hits = 0
    prev = None
    for i, k in enumerate(stream):
        l = k // N_EXPERT
        if l != prev:
            visits[l] += 1
            prev = l
        if plan:
            nxt[k].pop()
        s = slot_of.get(k)
        if s is not None:
            hits += 1
        else:
            if free:
                s = free.pop()
            else:
                gap = visits[layer_arr] - last_arr
                # lazily decayed popularity
                pd = p_arr * (1.0 - alpha) ** (visits[layer_arr] - pup_arr)
                score = gap / np.maximum(pd, 1e-4) ** pexp
                if plan:
                    score[plan_arr >= pass_id[i]] = -1.0
                s = int(np.argmax(score))
                del slot_of[int(key_arr[s])]
            slot_of[k] = s
            key_arr[s] = k
            layer_arr[s] = l
            p_arr[s] = 1e-4
            pup_arr[s] = visits[l] - 1
        # activation updates
        gap_p = visits[l] - 1 - pup_arr[s]
        p_arr[s] = (1.0 - alpha) ** (gap_p + 1) * p_arr[s] + alpha
        pup_arr[s] = visits[l]
        last_arr[s] = visits[l]
        if plan:
            nu = nxt[k][-1] if nxt[k] else None
            plan_arr[s] = pass_id[nu] if (nu is not None and pass_id[nu] == pass_id[i]) else 0
    return hits / len(stream)


def sim_reuse(stream, cap, alpha=0.05, use_p=True):
    """Phase + geometric-EWMA eviction with a lazy max-heap.

    Time unit = layer-steps: t advances by 1 whenever the stream moves to a
    new (pass, layer). p(l,e) is EWMA over *visits of layer l* — decayed
    lazily by the number of visits since last update.
    """
    layers = layers_of(stream)
    lpos = {l: i for i, l in enumerate(layers)}
    L = len(layers)
    p = {}          # key -> (ewma, visits_of_layer_at_update)
    visits = defaultdict(int)   # layer -> visit count so far
    c = set()
    cur_est = {}
    heap = []       # (-est, key), lazy
    hits = 0
    t = 0           # absolute layer-step
    prev = None     # (layer)
    INF = 1 << 60

    def est_next_use(k):
        l = k // N_EXPERT
        # next visit of layer l in layer-steps from now (cyclic over L)
        d = (lpos[l] - (t % L)) % L
        if d == 0:
            d = L
        if not use_p:
            return t + d
        ew, at = p.get(k, (0.0, visits[l]))
        gap = visits[l] - at
        if gap > 0:
            ew *= (1.0 - alpha) ** gap
        ew = max(ew, 1e-4)
        return t + d + L * (1.0 - ew) / ew

    for k in stream:
        l = k // N_EXPERT
        if l != prev:
            # new (pass, layer) visit
            t += 1
            visits[l] += 1
            prev = l
        # EWMA update for the activated expert (activation on this visit)
        if use_p:
            ew, at = p.get(k, (0.0, visits[l] - 1))
            gap = visits[l] - 1 - at
            if gap > 0:
                ew *= (1.0 - alpha) ** gap
            p[k] = ((1.0 - alpha) * ew + alpha, visits[l])
        if k in c:
            hits += 1
        else:
            if len(c) >= cap:
                while True:
                    negest, vk = heapq.heappop(heap)
                    if vk in c and cur_est.get(vk) == -negest:
                        c.remove(vk)
                        del cur_est[vk]
                        break
            c.add(k)
        e = est_next_use(k)
        cur_est[k] = e
        heapq.heappush(heap, (-e, k))
    return hits / len(stream)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    stream = []
    for d in sorted((ROOT / "traces/glm52").glob("dump*.bin")):
        stream.extend(parse_topk(d, limit))
    uniq = len(set(stream))
    print(f"{len(stream)} activations, {uniq} unique experts")
    for cap in (2626, 3675):   # 30 GiB and 42 GiB at 11.7 MiB/expert
        print(f"--- cap {cap} slots ({cap/uniq:.0%} of touched) ---")
        results = [
            ("LRU", sim_lru(stream, cap)),
            ("PLAN-LRU", sim_plan_lru(stream, cap)),
            ("GDLRU p^1", sim_gdlru(stream, cap, pexp=1.0)),
            ("GDLRU p^.5", sim_gdlru(stream, cap, pexp=0.5)),
            ("GDLRU a=.005", sim_gdlru(stream, cap, alpha=0.005)),
            ("PLAN-GDLRU", sim_gdlru(stream, cap, pexp=1.0, plan=True)),
            ("Belady", sim_belady(stream, cap)),
        ]
        for name, r in results:
            print(f"{name:16s} hit rate: {r:.4f}")


if __name__ == "__main__":
    main()
