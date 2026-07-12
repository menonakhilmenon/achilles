#!/usr/bin/env python
"""Reuse-hazard analysis on GLM-5.2 routing streams.

h(g) = P(expert activated at this visit of its layer | last activation was
g visits ago). If h is monotone decreasing in g, LRU is the optimal
gap-only policy and headroom must come from per-expert rate conditioning.
Also reports h(g) split by expert popularity tercile, and the reuse-distance
distribution Belady actually sees.
"""
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_EXPERT = 256


def parse_topk(path):
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
            stream.append((l, ids))
        else:
            raise ValueError("bad tag")
    return stream


def main():
    recs = []
    for d in sorted((ROOT / "traces/glm52").glob("dump*.bin")):
        recs.extend(parse_topk(d))
    # visits[l] counts how many times layer l has run; per-key last activation visit
    visits = defaultdict(int)
    lastact = {}
    act_count = defaultdict(int)
    # hazard numerator/denominator per gap bucket: on each activation with gap g,
    # it contributes: misses at gaps 1..g-1 (denominator only) and a hit at g.
    GAPS = 64
    num = [0] * (GAPS + 1)
    den = [0] * (GAPS + 1)
    for l, ids in recs:
        visits[l] += 1
        v = visits[l]
        for e in ids:
            k = l * N_EXPERT + e
            act_count[k] += 1
            la = lastact.get(k)
            if la is not None:
                g = v - la
                gb = min(g, GAPS)
                num[gb] += 1
                for gg in range(1, gb):
                    den[gg] += 1
                den[gb] += 1
            lastact[k] = v
    total_visits = sum(visits.values()) / len(visits)
    print(f"{len(recs)} layer-visits, mean visits/layer {total_visits:.0f}")
    print("g   h(g)=P(reuse at gap g | survived to g)   n")
    for g in range(1, 33):
        if den[g]:
            print(f"{g:3d}  {num[g]/den[g]:.4f}   {den[g]}")
    # popularity terciles
    pops = sorted(act_count.values())
    t1 = pops[len(pops) // 3]
    t2 = pops[2 * len(pops) // 3]
    print(f"\npopularity terciles: <= {t1}, <= {t2}, rest")
    num3 = [[0] * (GAPS + 1) for _ in range(3)]
    den3 = [[0] * (GAPS + 1) for _ in range(3)]
    visits.clear()
    lastact.clear()
    for l, ids in recs:
        visits[l] += 1
        v = visits[l]
        for e in ids:
            k = l * N_EXPERT + e
            c = act_count[k]
            tier = 0 if c <= t1 else (1 if c <= t2 else 2)
            la = lastact.get(k)
            if la is not None:
                g = v - la
                gb = min(g, GAPS)
                num3[tier][gb] += 1
                for gg in range(1, gb):
                    den3[tier][gg] += 1
                den3[tier][gb] += 1
            lastact[k] = v
    print("g    h_low     h_mid     h_high")
    for g in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        row = f"{g:3d}"
        for tier in range(3):
            row += f"  {num3[tier][g]/den3[tier][g]:.4f}" if den3[tier][g] else "      -"
        print(row)


if __name__ == "__main__":
    main()
