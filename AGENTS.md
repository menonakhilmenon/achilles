# AGENTS.md — operating this repo

A runbook for AI agents (and humans) to set up and drive the achilles stack on a
fresh machine. This repo runs a frontier-scale MoE model on a 64 GB desktop by
using **RAM as a cache tier over SSD-resident experts** (VRAM holds the dense
skeleton). The core result is settled — see `docs/traces-analysis.md` §33 and the
README: decode throughput is a clean function of how much of the expert working
set fits in your RAM cache.

Read this whole file before running anything heavy — several steps will hang, OOM,
or land the model on the wrong GPU if done naively. The gotchas below are load-
bearing.

## 1. One-shot setup

```
scripts/bootstrap.sh              # build llama.cpp + arena + venv (idempotent)
MODEL=air scripts/bootstrap.sh    # ...and download GLM-4.5-Air (47 GB, best first run)
```

That wraps `scripts/setup_llama.sh` (llama.cpp Vulkan at the pinned tag) → `./build.sh`
(compiles `src/achilles-arena`) → the Python venv. All no-root. To do it by hand,
run those three in order.

Prereqs (install first, needs root):
```
# Fedora / Bazzite
sudo dnf install gcc-c++ cmake liburing-devel vulkan-loader-devel vulkan-headers
# Debian / Ubuntu
sudo apt install g++ cmake liburing-dev libvulkan-dev vulkan-tools
```
`build.sh` finds `liburing` in the system path, via `pkg-config`, or under a
homebrew prefix automatically — no action needed if it's installed.

## 2. Models

| command | model | size | notes |
|---|---|---|---|
| `scripts/download_air.sh` | GLM-4.5-Air | 47 GB | fits RAM → ~15 tok/s; start here |
| `scripts/download_glm46.sh` | GLM-4.6 IQ2_XXS | 115 GB | ~31% resident → 2.7 tok/s |
| `unsloth/GLM-5.2-GGUF` UD-Q2_K_XL | GLM-5.2 | 247 GB | the 744B target → 1.6 tok/s |

**Put big GGUFs on the fastest NVMe.** If that's a different mount than `models/`,
download there and symlink: `DEST=/mnt/fast/glm46-gguf scripts/download_glm46.sh`
then `ln -s /mnt/fast/glm46-gguf models/glm46-gguf`. Decode reads experts from this
drive on every cache miss — its latency is your floor.

## 3. Running — the memory-safe way (do NOT skip)

The arena will allocate up to its budget plus llama.cpp overhead. Run it **inside a
systemd memory envelope** so a runaway never OOMs the box, and cap the runtime so a
hang self-kills:

```
VK_GPU_NAME="<your GPU>" source bench/vkdev.sh     # resolve $VKDEV (see gotcha #1)

systemctl --user reset-failed arena 2>/dev/null
systemd-run --user --wait --unit arena \
  -p WorkingDirectory="$PWD" \
  -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G -p RuntimeMaxSec=600 \
  bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena \
    -m models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf -p 'hello' \
    -n 96 -t 10 -ngl 99 -ot exps=CPU \
    --budget-gib 30 --workers 6 --policy reuse --stats > /tmp/arena.log 2>&1"
```

- **Redirect goes INSIDE `bash -c`.** Put `> log 2>&1` on the `systemd-run` line
  and you capture only systemd's status text — the arena's `--stats` output vanishes
  into the journal (`journalctl --user -u arena`).
- `-ngl 99 -ot exps=CPU` = dense skeleton on GPU, every expert tensor to the arena
  (it intercepts them despite the "CPU" label; works for any `glm4moe` model).
- `--budget-gib N` is the one knob that matters — the RAM cache cap. Bigger = higher
  hit rate = faster, up to your RAM.
- Tune `MemoryHigh/Max` to your RAM. On 64 GB, 44/48 leaves headroom.

Adjust the model down (`--budget-gib`) for less RAM. CPU-only (`-ngl 0`) works but
is not the point.

## 4. Gotchas that will bite you

1. **GPU by name, always.** Vulkan enumeration order flips across reboots (iGPU/dGPU
   swap); a hardcoded index silently lands the model on the iGPU → `vm_validate
   ENOMEM` / `DeviceLost`. Use `VK_GPU_NAME="RX 9070" source bench/vkdev.sh` →
   `$VKDEV`, and pass `GGML_VK_VISIBLE_DEVICES=$VKDEV`. List names with
   `llama.cpp/build-vk/bin/llama-bench --list-devices`.
2. **Bazzite / AMD TTM kernel args.** The GPU TTM page pool has an invisible cap
   that can spiral into global OOM. Add `ttm.pages_limit=8388608
   ttm.page_pool_size=1048576`. Do **NOT** lower `pages_limit` — a smaller value
   caps GTT and causes `DeviceLost`.
3. **`--pstream 1` can hang (io_uring lost completion).** Under heavy back-to-back
   runs a dropped completion leaves a counter stuck and the stall-loop spins forever
   (one core hot, drive idle, no output). `RuntimeMaxSec` (above) self-kills it;
   otherwise `systemctl --user stop arena`. Prefer `--pstream 0` if you hit it.
   This is a known latent bug (watchdog = TODO).
4. **Benchmark hygiene.** `tuned-adm profile throughput-performance-bazzite` before a
   measurement, restore `balanced-bazzite` after. Check `MemAvailable ≥ budget+headroom`
   before launching (`awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo`).
5. **Shadow (`--shadow`) helps prefill, not decode.** Decode is latency-bound on a
   fast drive, so the expert-major repack buys nothing there (§31, §33). Only build
   it (`scripts/repack_shadow.py <first-shard.gguf> <out-prefix>`) if you care about
   long-prompt prefill.

## 5. Reading `--stats`

```
ACHILLES decode: 96 tok in 35.1s = 2.73 tok/s
ACHILLES arena:  ... (hit rate 0.80)
ACHILLES decode-detail: stall=10.5s ... hit=0.868 (decode-only)   <- the number that matters
ACHILLES profile(ms/tok): wall=366 stall=109 score=28 cb_total=191 ...
```

Decode-only `hit` ≈ resident fraction outcome; higher = faster. If `hit` is low and
decode is slow, raise `--budget-gib` or use a smaller model/quant — that's the whole
story (§33). Token identity: run with and without `--shadow`; the `OUTPUT:` lines
must be byte-identical (the arena guarantees output equals unpaged inference).

## 6. Where things live

- `src/arena.cpp` — the entire runtime (single-file llama.cpp wrapper)
- `README.md` — user-facing quickstart + results
- `docs/traces-analysis.md` — full experimental record (§1–33); §33 = the conclusion
- `bench/` — benchmark scripts (memory-envelope patterns to copy) + `bench/results/`
- `scripts/` — `bootstrap.sh`, `setup_llama.sh`, `download_*.sh`, `repack_shadow.py`
