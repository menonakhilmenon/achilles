# Achilles

Run a frontier-scale MoE model (**GLM-5.2**, 744B params, 247 GB on disk) on a
64 GB gaming desktop by treating memory as a small cache over a paged expert
store: VRAM holds the dense skeleton, RAM holds a managed expert arena, NVMe
serves the misses — with a predictor steering both prefetch and eviction.

## Status (2026-07-12): it works

| GLM-5.2 UD-Q2_K_XL | naive (kernel mmap) | **achilles-arena, full profile** |
|---|---|---|
| decode | 0.30 tok/s | **1.57 tok/s** — 5.2× (Gen5 SSD + vectorized scorer, §30; 1.15 on Gen4) |
| prefill (~2700-token prompt) | ~0.5 tok/s | **25.1 tok/s** — ~50× (layer-streaming + shadow + io_uring + full-batch ubatches) |

Every number cold, controlled, token-identity-gated. Polite profile (desktop
stays usable): ~0.4–0.5 tok/s decode. Earlier 0.94 readings were warm-cache
flattered — traces-analysis §21 has the honest accounting; §21–23 the
one-day 0.64→1.12 progression (fetch tuning, io_uring, expert-major shadow).

GLM-4.5-Air (110B) runs at 3.4 tok/s memory-constrained / 15.3 unconstrained.
Output is token-identical to unpaged inference — the arena is bit-exact.

## How it works (src/arena.cpp — a llama.cpp wrapper, no fork)

1. llama.cpp mmaps the GGUF; the arena replaces every expert tensor's pages
   in-place with anonymous memory (`mmap MAP_FIXED`) — tensor pointers survive.
2. The router's `topk` is observable via the scheduler callback *before* the
   expert matmul runs: misses are loaded (io_uring O_DIRECT, parallel workers)
   with guaranteed validity; gate-ahead/trained probes prefetch layers ahead.
3. Eviction is LRU with **plan-aware protection** (predicted-soon experts are
   spared — approximate Belady; this alone was worth +40% tok/s) under a hard
   byte budget with janitor + backpressure so real RSS never runs ahead.

Key empirical findings along the way (details in docs/traces-analysis.md §1–17):
40% of a token's experts recur within 1 token, 80% within 8; trained linear
probes predict experts 8 layers ahead at 52–90% recall (model-dependent);
routing margins are razor-thin (95% of top-8 boundaries < 0.01); cross-token
prediction is a dead end on GLM-5.2; on SSD-bound hardware, prediction spent
on *eviction* is free while extra prefetch bytes are poison.

## Hardware notes (measured on Ryzen 9800X3D / 64GB / RX 9070 XT / Fury Renegade)

- SSD sustained reads: 3.9 GB/s (poor airflow) → **7.3 GB/s** (case fans) — the
  cheapest 2× in the project. Gen5 drive projected → ~2.5–3.5 tok/s.
- Bazzite kernel gotcha: the TTM GPU page pool defaults to a **30 GB invisible
  cap** and never shrinks → global OOM spiral. Fix: `ttm.pages_limit=1048576
  ttm.page_pool_size=1048576` kernel args.
- RAM-bandwidth ceiling (infinite-RAM bound): ~5.5–6 tok/s on dual-channel DDR5.

## Layout

- `docs/research.md` — feasibility math, prior-art survey
- `docs/plan.md` — phased plan with measured gates
- `docs/traces-analysis.md` — the full experimental record (§1–17)
- `src/arena.cpp` — the runtime; `src/pager.cpp` — v1 (page-cache steering)
- `scripts/` — tracing, probe training, analysis; `bench/` — benchmarks
