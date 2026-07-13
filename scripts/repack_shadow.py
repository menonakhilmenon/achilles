#!/usr/bin/env python
"""Expert-major shadow repack: rewrite MoE expert tensors contiguously.

For each (layer, expert): gate/up/down slices packed adjacently in a shadow
file. Each slice's shadow offset is padded so that
    shadow_off % 4096 == original_file_off % 4096
which preserves the arena's O_DIRECT address/offset congruence (tensor mmap
addresses are page-aligned to file offsets).

Outputs:
  <out>.bin  — packed expert bytes
  <out>.idx  — binary index:
      magic 'ASHD', u32 version=1, u32 n_layer, u32 n_expert,
      then per (layer, expert, proj[3]): u64 shadow_off, u64 len
      (len==0 for missing projections/layers)

Usage: repack_shadow.py <first-shard.gguf> <out-prefix>
"""
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

PROJS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
CHUNK = 64 << 20


def shard_list(p: str):
    if "-of-" not in p:
        return [p]
    stem, tail = p.rsplit("-of-", 1)
    count = int(tail.split(".")[0])
    base = stem.rsplit("-", 1)[0]
    return [f"{base}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def main():
    first, outp = sys.argv[1], sys.argv[2]
    slices = {}   # (layer, proj_idx) -> (path, file_off, expert_len, n_expert)
    n_layer = 0
    n_expert = 0
    for path in shard_list(first):
        r = GGUFReader(path)
        # data section base: offset of first tensor is relative to data start
        for t in r.tensors:
            parts = t.name.split(".")
            if len(parts) < 3 or parts[0] != "blk":
                continue
            proj = parts[2].replace(".weight", "")
            if proj not in PROJS:
                continue
            l = int(parts[1])
            pi = PROJS.index(proj)
            ne = int(t.shape[-1])          # experts are the last dim
            nbytes = int(t.n_bytes)
            per = nbytes // ne
            assert per * ne == nbytes, f"{t.name}: {nbytes} not divisible by {ne}"
            slices[(l, pi)] = (path, int(t.data_offset), per, ne)
            n_layer = max(n_layer, l + 1)
            n_expert = max(n_expert, ne)
    print(f"{len(slices)} expert tensors, n_layer={n_layer}, n_expert={n_expert}")

    # Layout: each slice is stored inside its ORIGINAL 4K-aligned window
    # (align_down(src) .. align_up(src+per)), windows placed at 4K boundaries.
    # The arena's O_DIRECT reads align down/up and rely on overshoot bytes
    # matching what lives at the neighbouring addresses — copying the whole
    # original window preserves that (plain congruence padding does NOT:
    # overshoot would splat pad bytes into adjacent tensors' memory).
    idx = {}
    win = {}   # (l,e,pi) -> (window_lo_src, window_len)
    off = 0
    total = 0
    for l in range(n_layer):
        for e in range(n_expert):
            for pi in range(3):
                s = slices.get((l, pi))
                if s is None or e >= s[3]:
                    idx[(l, e, pi)] = (0, 0)
                    continue
                _, foff, per, _ = s
                src = foff + e * per
                wlo = src & ~4095
                whi = (src + per + 4095) & ~4095
                idx[(l, e, pi)] = (off + (src - wlo), per)
                win[(l, e, pi)] = (wlo, whi - wlo)
                off += whi - wlo
                total += per
    print(f"shadow size: {off/2**30:.1f} GiB (payload {total/2**30:.1f})")

    fouts = {}
    t0 = time.time()
    written = 0
    with open(outp + ".bin", "wb") as out:
        out.truncate(off)
        for l in range(n_layer):
            for pi in range(3):
                s = slices.get((l, pi))
                if s is None:
                    continue
                path, foff, per, ne = s
                if path not in fouts:
                    fouts[path] = open(path, "rb")
                f = fouts[path]
                fsize = Path(path).stat().st_size
                for e in range(ne):
                    so, ln = idx[(l, e, pi)]
                    if ln == 0:
                        continue
                    wlo, wlen = win[(l, e, pi)]
                    f.seek(wlo)
                    out.seek(so - ((foff + e * per) - wlo))
                    # the aligned-up window can extend past EOF on the file's
                    # last tensor: read what exists, zero-fill the tail
                    left = min(wlen, max(0, fsize - wlo))
                    tail = wlen - left
                    while left:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            raise IOError(f"short read at {wlo} ({path})")
                        out.write(buf)
                        left -= len(buf)
                    if tail:
                        out.write(b"\0" * tail)
                    written += ln
            if l % 8 == 0:
                el = time.time() - t0
                print(f"layer {l}/{n_layer}  {written/2**30:.1f} GiB  "
                      f"{written/2**30/max(el,1e-9):.2f} GiB/s", flush=True)
    with open(outp + ".idx", "wb") as ix:
        ix.write(b"ASHD" + struct.pack("<III", 1, n_layer, n_expert))
        for l in range(n_layer):
            for e in range(n_expert):
                for pi in range(3):
                    so, ln = idx[(l, e, pi)]
                    ix.write(struct.pack("<QQ", so, ln))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {outp}.bin/.idx")


if __name__ == "__main__":
    main()
