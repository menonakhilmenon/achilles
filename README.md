# Achilles

Run a frontier-scale MoE model (**GLM-5.2**, 744B params, 247 GB on disk) on a
64 GB gaming desktop by treating memory as a small cache over a paged expert
store: VRAM holds the dense skeleton, RAM holds a managed expert arena, NVMe
serves the misses — with a predictor steering both prefetch and eviction.

It's a single-file llama.cpp wrapper (`src/arena.cpp`), not a fork: llama.cpp
mmaps the GGUF as usual, and the arena replaces every expert tensor's pages
in-place with anonymous memory it pages from disk on demand.

## Status: it works

| GLM-5.2 UD-Q2_K_XL | naive (kernel mmap) | **achilles-arena, full profile** |
|---|---|---|
| decode | 0.30 tok/s | **1.60 tok/s** — 5.3× (Gen5 SSD + vectorized scorer + wide prefetch, §30–31; 1.15 on Gen4) |
| prefill (~2700-token prompt) | ~0.5 tok/s | **25.1 tok/s** — ~50× (layer-streaming + shadow + io_uring + full-batch ubatches) |

Every number cold, controlled, token-identity-gated (output is bit-identical to
unpaged inference). GLM-4.5-Air (110B) runs at 3.4 tok/s memory-constrained /
15.3 unconstrained. Full experimental record in `docs/traces-analysis.md`.

## Quickstart

Tested on Linux (Bazzite/Fedora, kernel 6.x) with a discrete GPU. You need:

- **A discrete GPU with Vulkan** and enough VRAM for the dense (non-expert)
  layers — roughly 8–12 GB depending on model. (CPU-only works via `-ngl 0` but
  is far slower and not what this is for.)
- **RAM** ≥ your chosen expert-cache budget + a few GB of headroom (48 GB is
  comfortable; it runs on less with a smaller `--budget-gib`).
- **A fast NVMe** with room for the model (48 GB for Air, 250 GB for GLM-5.2).
- Toolchain: `g++` (C++17), `cmake`, the Vulkan loader + your GPU's Vulkan
  driver, `liburing` dev headers, and Python 3 for the model download.

  ```
  # Fedora / Bazzite
  sudo dnf install gcc-c++ cmake liburing-devel vulkan-loader-devel vulkan-headers
  # Debian / Ubuntu
  sudo apt install g++ cmake liburing-dev libvulkan-dev vulkan-tools
  ```

### 1. Build llama.cpp (Vulkan) + the arena

```
scripts/setup_llama.sh     # clones + builds llama.cpp (Vulkan) at the pinned tag
./build.sh                 # compiles src/achilles-arena against it
```

`setup_llama.sh` fetches stock llama.cpp — the arena only uses its public API, so
no fork is needed for decode/prefill/prefetch. (The optional `--spec-mtp`
self-drafting path needs two extra commits that live in a private fork.)

### 2. Get a model

Start with **GLM-4.5-Air** (47 GB) — small enough to be practical:

```
python3 -m venv .venv && .venv/bin/pip install huggingface_hub hf_transfer
scripts/download_air.sh          # -> models/glm45-air-gguf/
```

The 247 GB headline target is `unsloth/GLM-5.2-GGUF` (UD-Q2_K_XL), same layout.

### 3. Run

```
src/achilles-arena \
  -m models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf \
  -p "Explain mixture-of-experts language models in one paragraph." \
  -n 128 -t "$(nproc)" -ngl 99 -ot exps=CPU \
  --budget-gib 16 --workers 6 --policy reuse --stats
```

- `-ngl 99 -ot exps=CPU` — put the dense skeleton on the GPU and hand every
  expert tensor to the arena (it intercepts them regardless of the "CPU" label).
- `--budget-gib N` — cap the RAM expert cache at N GiB; everything else is paged
  from NVMe on demand.
- `--stats` — print hit rate, tok/s, and a per-token timing breakdown.

**Picking the GPU.** With an iGPU + dGPU, Vulkan may enumerate the wrong one.
Resolve it by name:

```
llama.cpp/build-vk/bin/llama-bench --list-devices        # find your GPU's name
VK_GPU_NAME="RTX 4090" source bench/vkdev.sh              # sets $VKDEV by name
GGML_VK_VISIBLE_DEVICES=$VKDEV src/achilles-arena ...     # then run
```

**Bazzite / AMD kernel note.** On Bazzite (and some AMD setups) the GPU's TTM
page pool has an invisible cap that never shrinks and can spiral into a global
OOM. Add these kernel args (do **not** lower `pages_limit` — a smaller value
caps GTT and causes `DeviceLost`):

```
ttm.pages_limit=8388608 ttm.page_pool_size=1048576
```

### Tuning flags (arena-specific)

All standard llama.cpp flags work (`-m -p -n -t -ngl -ot -ub …`) — the arena
passes them through. On top of those:

| flag | default | meaning |
|---|---|---|
| `--budget-gib N` | — | RAM expert-cache cap in GiB (the main knob) |
| `--policy reuse\|lru` | lru | eviction policy; `reuse` (decayed popularity + plan-aware protection) is recommended |
| `--fetch N` | 6 | gate-ahead prefetch width per predicted layer (6 suits a fast Gen5 drive; use 2 on a bandwidth-saturated Gen4 — §31) |
| `--workers N` | 6 | io_uring worker threads |
| `--pstream 0\|1` | 1 | prefill layer-streaming (big speedup on long prompts) |
| `--shadow PREFIX` | off | read experts from an expert-major repacked file (`scripts/repack_shadow.py`) — faster sequential reads |
| `--stats` | off | print hit rate, tok/s, timing |
| `--no-pager` | off | disable the arena (baseline kernel-mmap behavior, for comparison) |

Optional: `scripts/repack_shadow.py <model.gguf> models/<name>-shadow` builds the
expert-major `--shadow` file (token-identical, faster reads).

## How it works

1. llama.cpp mmaps the GGUF; the arena replaces every expert tensor's pages
   in-place with anonymous memory (`mmap MAP_FIXED`) — tensor pointers survive.
2. The router's `topk` is observable via the scheduler callback *before* the
   expert matmul runs: misses are loaded (io_uring O_DIRECT, parallel workers)
   with guaranteed validity; gate-ahead probes prefetch layers ahead.
3. Eviction is decayed-popularity LRU with **plan-aware protection**
   (predicted-soon experts are spared — approximate Belady) under a hard byte
   budget, with a janitor + backpressure so real RSS never runs ahead.

Key empirical findings (full detail in `docs/traces-analysis.md`): 40% of a
token's experts recur within 1 token, 80% within 8; routing margins are
razor-thin (95% of top-8 boundaries < 0.01) and turnover is ~70%/token, which
caps how far prediction can go; on a bandwidth-saturated drive prefetch bytes are
poison, but on a latency-bound one (Gen5) with idle bandwidth, wider prefetch
pays; prediction spent on *eviction* is always free.

## Hardware notes (measured on Ryzen 9800X3D / 64 GB / RX 9070 XT)

- SSD sustained reads dominate decode: 3.9 GB/s (poor airflow) → 7.3 GB/s (case
  fans) on Gen4; a Gen5 drive (Samsung 9100 PRO, ~13 GB/s) took decode to 1.6 tok/s.
- RAM-bandwidth ceiling (infinite-RAM bound): ~5.5–6 tok/s on dual-channel DDR5.

## Layout

- `build.sh`, `scripts/setup_llama.sh` — build the arena and its llama.cpp
- `src/arena.cpp` — the runtime (the whole thing); `src/pager.cpp` — an earlier
  page-cache-steering prototype
- `docs/research.md` — feasibility math, prior-art survey
- `docs/plan.md` — phased plan with measured gates
- `docs/traces-analysis.md` — the full experimental record (§1–32)
- `scripts/` — model download, tracing, probe training, analysis
- `bench/` — benchmark scripts and their recorded results (`bench/results/`)
