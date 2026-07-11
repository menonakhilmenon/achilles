#!/usr/bin/env python
"""Parse trace-moe binary output into npy arrays.

Records:
  TOPK  (0x544F504B): i32 step, layer, n_tokens, k     + k*n_tokens i32
  PROB  (0x50524F42): i32 step, layer, n_tokens, n_exp + n_exp*n_tokens f32

Writes alongside the .bin:
  decode_topk.npy   (T, L, k) i32   -- steps with n_tokens==1
  decode_probs.npy  (T, L, E) f16   -- if PROB records present
  prefill_topk.npy  (P, L, k) i32   -- step-0 batch
"""
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TOPK, PROB = 0x544F504B, 0x50524F42


def parse(path: Path):
    data = path.read_bytes()
    off = 0
    topk = defaultdict(dict)   # step -> layer -> (k, n_tokens) ids
    probs = defaultdict(dict)
    while off + 20 <= len(data):
        magic, step, layer, n_tokens, n = struct.unpack_from("<IiiiI", data, off)
        off += 20
        if magic == TOPK:
            arr = np.frombuffer(data, "<i4", n * n_tokens, off).reshape(n_tokens, n)
            off += 4 * n * n_tokens
            topk[step][layer] = arr
        elif magic == PROB:
            arr = np.frombuffer(data, "<f4", n * n_tokens, off).reshape(n_tokens, n)
            off += 4 * n * n_tokens
            probs[step][layer] = arr
        else:
            raise ValueError(f"bad magic {magic:#x} at {off - 20}")
    return topk, probs


def main():
    for arg in sys.argv[1:]:
        path = Path(arg)
        topk, probs = parse(path)
        layers = sorted(topk[0])
        L = len(layers)

        # llama.cpp computes only the last prompt token through the final layer
        # during prefill; pad missing leading rows with -1 (invalid expert id)
        P = max(a.shape[0] for a in topk[0].values())
        def pad(a):
            if a.shape[0] == P:
                return a
            fill = np.full((P - a.shape[0], a.shape[1]), -1, a.dtype)
            return np.concatenate([fill, a])
        prefill = np.stack([pad(topk[0][l]) for l in layers], axis=1)  # (P, L, k)
        dec_steps = sorted(s for s in topk if s > 0)
        decode = (np.stack([np.stack([topk[s][l][0] for l in layers]) for s in dec_steps])
                  if dec_steps else np.zeros((0, L, 8), np.int32))
        np.save(path.parent / (path.stem + ".prefill_topk.npy"), prefill)
        np.save(path.parent / (path.stem + ".decode_topk.npy"), decode)
        if probs:
            psteps = sorted(probs)
            E = next(iter(probs[psteps[0]].values())).shape[1]
            dp = np.stack([
                np.stack([probs[s][l][0] if l in probs[s] else np.zeros(E, np.float32)
                          for l in layers]) for s in psteps
            ]).astype(np.float16)
            np.save(path.parent / (path.stem + ".decode_probs.npy"), dp)
        print(f"{path.name}: prefill={prefill.shape} decode={decode.shape} layers={L}")


if __name__ == "__main__":
    main()
