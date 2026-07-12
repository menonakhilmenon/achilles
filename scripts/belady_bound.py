#!/usr/bin/env python
"""Belady (optimal eviction) bound vs LFU/LRU on GLM-5.2 routing streams.

Parses arena --dump 'T' records into a single activation stream of
(layer, expert) keys and replays it through caches at the real budget
(experts of equal cost; capacity in expert slots). Answers: how much hit
rate is left on the table by LFU — i.e., is a smarter eviction policy
worth building?
"""
import heapq
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_SLOTS = 3675  # 42 GiB / 11.7 MiB


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
            for e in ids:
                stream.append(l * 256 + e)
        else:
            raise ValueError("bad tag")
    return stream


def sim_lru(stream, cap):
    from collections import OrderedDict
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


def sim_lfu(stream, cap):
    c = {}
    score = defaultdict(float)
    hits = 0
    for k in stream:
        score[k] += 1
        if k in c:
            hits += 1
        else:
            if len(c) >= cap:
                victim = min(c, key=lambda x: score[x])
                del c[victim]
            c[k] = True
    return hits / len(stream)


def sim_belady(stream, cap):
    nxt = defaultdict(list)
    for i in range(len(stream) - 1, -1, -1):
        nxt[stream[i]].append(i)
    INF = 1 << 60
    c = set()
    cur_nu = {}   # authoritative next-use per cached key
    heap = []     # lazy (-next_use, key); validated against cur_nu on pop
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


def main():
    stream = []
    for d in sorted((ROOT / "traces/glm52").glob("dump*.bin")):
        stream.extend(parse_topk(d))
    uniq = len(set(stream))
    print(f"{len(stream)} activations, {uniq} unique experts, cap {CAP_SLOTS} "
          f"({CAP_SLOTS/uniq:.0%} of touched)")
    for name, fn in (("LRU", sim_lru), ("LFU", sim_lfu), ("Belady", sim_belady)):
        r = fn(stream, CAP_SLOTS)
        print(f"{name:8s} hit rate: {r:.4f}")


if __name__ == "__main__":
    main()
