# Trace Analysis: Expert Routing Locality & Predictability

*Phase 1 results, 2026-07-12. Raw numbers in `bench/results/olmoe-analysis.json`.*

## Setup

- **Model**: OLMoE-1B-7B-Instruct — 16 layers × 64 experts, top-8, softmax router.
  (Small enough to trace with full hidden states in PyTorch on CPU; Qwen3-30B-A3B
  traces — 48 × 128, top-8, llama.cpp — are being collected for scale confirmation.)
- **Corpus**: 60 prompts × 6 domains (code, math, chat, essay, reasoning, structured),
  ≤1024 sampled decode tokens each (temp 0.7) → **37,128 decode tokens**, plus prefill.
- Per token × layer we record router logits (all 64) and the router's input hidden
  state, so every analysis below is exact, not sampled.

## Findings

### 1. Temporal locality is strong — the core premise holds

![reuse](assets/olmoe-reuse.png)

**40%** of a token's 8 experts were used by the immediately previous token; **80%**
appear within the previous 8 tokens; **96%** within 32 (median layer, IQR band tight).
A cache that merely keeps "recently used experts" resident captures most traffic —
this is the single most load-bearing empirical fact for the whole design.

### 2. Popularity skew alone is NOT enough

![skew](assets/olmoe-skew.png)

The most popular half of experts covers only ~70% of activations (median layer;
OLMoE is load-balanced by training). A **static** hot set is a weak cache on its own —
dynamic recency/frequency does the real work (see next) — but it still matters as the
*placement* prior for what should live in RAM vs SSD at cold start.

### 3. Cache policy: decayed-LFU > LRU > static; capacity is the lever

![cache](assets/olmoe-cache.png)

| capacity (frac of experts) | 12.5% | 25% | 37.5% | 50% | 75% |
|---|---|---|---|---|---|
| LRU | .28 | .54 | .68 | .79 | .93 |
| **LFU (0.98 decay/token)** | **.35** | **.56** | **.71** | **.82** | **.94** |
| static top-C | .26 | .43 | .57 | .69 | .89 |

Decayed-LFU (an LCP-style policy) wins at every size, consistent with the literature.
Per-layer caches, reset per prompt (conservative — a long-running server does better).

### 4. Gate-ahead prediction: high recall with ZERO training

![gate-ahead](assets/olmoe-gate-ahead.png)

Applying layer *l+Δ*'s (real) router to layer *l*'s hidden state and prefetching the
top-m:

| lookahead Δ | m=8 | m=16 | m=32 |
|---|---|---|---|
| 1 | .84 | **.96** | .99 |
| 4 | .66 | .85 | **.96** |
| 8 | .49 | .69 | **.88** |

Hidden states drift slowly across layers, so the router itself is a free predictor.
2× overprovisioning (m=16) buys ~12pp of recall at Δ=1; 4× keeps 88% recall even a
full 8 layers ahead. **This is the floor a trained predictor must beat** — and it
already clears the bar the paging design needs for RAM→compute prefetch. The long-
horizon (SSD→RAM) predictor is the part where training should pay.

### 5. Router margins are razor-thin — substitution is cheap, prediction is hard at the boundary

![margins](assets/olmoe-margins.png)

The probability gap between the 8th and 9th expert is < 0.01 in **95.5%** of
(token, layer) events (median 0.0015). Two consequences:

- The last cache slots are inherently hard to predict (near-coin-flips) — chasing
  100% recall is the wrong goal; overprovision + demand-fallback instead.
- Swapping a marginal expert for its runner-up should cost little quality
  (HOBBIT/ ReMoE-style fallbacks look viable — to be verified in Phase 2 with
  perplexity measurements).

### 6. Trained predictors beat gate-ahead — and the gap grows with lookahead

![predictor](assets/olmoe-predictor.png)

Per-layer predictors trained on the traces (KL-distilled from the true router
distribution; 80/20 prompt-level split; `scripts/train_predictor.py`):

| recall@16 of actual top-8 | Δ=1 | Δ=4 | Δ=8 |
|---|---|---|---|
| gate-ahead (no training) | .965 | .855 | .704 |
| **linear probe (warm-started, 2048→64)** | **.981** | **.940** | **.901** |
| MLP 2048→512→64 (cold start) | .908 | .888 | .876 |

The linear probe — literally the same shape as the router, warm-started from the
gate-ahead solution — dominates at every horizon, and its advantage *widens* with Δ
(+20 pp at Δ=8). Long-horizon prediction is learnable, which is precisely what the
SSD tier needs: at Δ=8 with 2× overprovision, 90% of the experts a layer will need
can be in flight 8 layers early. (The cold-start MLP shows the failure mode too:
without the warm start it stays below the probe — initialization from the router is
the trick. First training attempt with BCE-on-soft-probs underfit catastrophically;
KL distillation is the right loss.)

### 7. Qwen3-30B-A3B (48×128, top-8): locality holds, sparser pools cache BETTER

33,354 decode tokens, same corpus, expert IDs via the llama.cpp tracer
(`bench/results/qwen3-30b-analysis.json`, `assets/qwen3-30b-*.png`):

- **Temporal locality is architecture-invariant**: 43.5% prev-token reuse (OLMoE:
  40.4%); 77% within 8 tokens (OLMoE: 80%). The core premise transfers.
- **Fractional capacity works harder at higher expert counts**: at 12.5% of experts
  cached, LFU hits 57.6% (OLMoE: 35.4%); at 37.5%, 89.9% (OLMoE: 70.7%). With only
  8-of-128 firing per layer, the *relative* working set shrinks — and GLM-5.2's
  8-of-256 should extend the trend. The premise gets easier at target scale.
- Static popularity is even weaker at 128E (top-25% covers 47%); margins equally
  thin (98% of top-8 boundaries gap < 0.01). Same conclusions, stronger.

### 7b. GLM-4.5-Air (45×128, top-8, the target's family): same story + a live paging baseline

Partial run (4 prompts × 512 tokens — stopped early because the hard `MemoryMax=44G`
cap pushed ~12 GB into zram swap and made the desktop unusable; resume with
`MemoryHigh` + `ionice`):

- **Locality confirmed on GLM lineage**: 40% prev-token reuse, 69% within 8 tokens,
  LRU 47% hits at 12.5% fractional capacity — between OLMoE and Qwen, same shape
  (`bench/results/glm45-air-analysis.json`).
- **Naive-mmap paging baseline measured**: the 47.4 GB UD-Q2_K_XL under a 44 GB
  memory envelope decodes at **~2.3–2.7 tok/s** (llama.cpp, tracing overhead
  included) with PSI-memory ~40%. This is the "existing tricks" floor the managed
  runtime (Phase 2) must beat on this exact model — and the kernel's LRU already
  benefits from the locality above, which bounds how much headroom pure caching
  has left; the win must come from prefetch + O_DIRECT arena discipline (no swap,
  no reclaim storms).

### 8. First end-to-end projection for GLM-5.2 on this box

Replaying the Qwen traces through the paging simulator at GLM-5.2-Q2 geometry
(12 MiB experts, 17.8% fractional residency = 40 GB cache vs 225 GB of experts,
13 GB/token compute reads, 55 GB/s RAM, 7.5 GB/s SSD, Δ=6 m=16 prefetch):

| config | tok/s | hit rate | note |
|---|---|---|---|
| predictor recall .95 | **1.90** | .91 | SSD-bandwidth-bound (2.8 GB/tok incl. prefetch waste) |
| predictor recall .80 | 1.71 | .86 | |
| demand-only (no prefetch) | 1.87 | .60 | stall-bound instead — same wall, worse p99 |
| SSD throttled to 4 GB/s | ~1.2 | | why the NVMe heatsink matters |

Useful traffic (~1.95 GB/tok) almost exactly balances compute (242 ms/tok): the box
sits at the bandwidth-balance point, so **~2 tok/s is the honest projection for
GLM-5.2-Q2 on 64 GB RAM** — at the plan's success threshold. Prefetch overshoot
(m=16) is the current overhead to attack (adaptive m / dedupe against in-flight);
doubling RAM moves the projection to ~3.5–4 tok/s (compute-bound).

### 9. GPU hybrid measured (RX 9070 XT, Vulkan/RADV): the fast tier works today

llama.cpp Vulkan build (`build-vk/`, brew-supplied glslc/headers on the immutable
OS), `-ngl 99 -ot "exps=CPU"` = dense+attention+KV in VRAM, experts in RAM:

| model | config | pp512 | tg128 |
|---|---|---|---|
| Qwen3-30B-A3B Q4 | CPU only | 168 | 23.8 |
| | hybrid | **447 (2.7×)** | 24.8 (+4%) |
| GLM-4.5-Air Q2 (110B) | CPU only | 21.3 | 5.5 |
| | hybrid | **59.5 (2.8×)** | **12.6 (2.3×)** |

- Prefill: ~3× everywhere (compute-bound, GPU wins as expected).
- Decode: +4% on Qwen (experts dominate its bytes) but **2.3× on the GLM family**,
  whose dense skeleton is heavy enough that evicting it from RAM to VRAM frees the
  whole 55 GB/s for expert reads. GLM-5.2 has the same anatomy.
- Revised GLM-5.2-Q2 projection with the hybrid split: RAM serves ~7.2 GB/token of
  experts (~130 ms) with SSD misses overlapping inside it → **~3.5–4 tok/s**, up
  from ~1.9 CPU-only. VRAM budget: Q2 dense ≈ 6 GB + KV + a hot-expert cache in the
  remaining ~8 GB — the three-tier design (VRAM ← RAM ← SSD) is live on this box.
- Practical notes: RDNA4 exposes KHR_coopmat matrix cores under RADV; `-ngl 0` on a
  Vulkan build still opportunistically offloads prefill (not a CPU baseline); pin
  the dGPU with `GGML_VK_VISIBLE_DEVICES=1` (the iGPU enumerates first with 33 GB
  of GTT/host memory — itself interesting as a future pinned-staging path).

### 10. Three-tier stack live-fired: GPU + RAM + SSD on GLM-4.5-Air

Full stack test (`bench/air_tiers2.sh`): dense skeleton in VRAM, experts in a
RAM budget enforced by `MemoryHigh` (model evicted from page cache before each
run so the scope faults its own pages), overflow paged from NVMe by the kernel:

| RAM budget | resident | decode tok/s | NVMe read for 256 tokens |
|---|---|---|---|
| 48 GB (100%) | 44.2 GiB | **15.30 ± .14** | 45 GB (model load only) |
| 32 GB (72%) | 31.8 GiB | 4.24 ± 1.39 | 140 GB |
| 24 GB (54%) | 23.8 GiB | 1.67 ± .02 | 518 GB |
| 16 GB (36%) | 15.8 GiB | 0.85 ± .00 | 1,146 GB |

- **Naive mmap paging collapses**: 3.6× slower at 72% resident, 18× at 36%. This
  is the floor the managed runtime replaces — and it collapses for measurable,
  fixable reasons, not because the SSD is too slow.
- **The kernel moves 2–3× the true demand**: at 24 GB, real expert demand is
  ~300 GB (measured ~46% miss × ~2.6 GB/token) but 518 GB crossed the bus —
  4 KiB fault granularity, blind readahead, and churn. An O_DIRECT expert-granular
  arena reclaims this before prediction contributes anything.
- **Headroom for the design**: perfect caching+prefetch at these residencies pencils
  to ~4–5 tok/s (54%) and ~3.5–4 (36%) vs 1.67 / 0.85 naive → **2.5–4.5× win
  available on identical hardware**, consistent with the simulator's structure.
- Naive paging variance is severe (±33% at 32 GB) — deadline-scheduled prefetch
  should also flatten p99, not just the mean.

### 11. GLM-5.2 (744B) actually runs on this desktop — the floors are measured

First contact with the beast (UD-Q2_K_XL, 247 GB on a 64 GB / 16 GB-VRAM box):

| config | result |
|---|---|
| CPU-only naive mmap | 16 tokens did NOT complete in ~5 h → **< 0.05 tok/s** (fault-serialized, QD≈1) |
| **Three-tier naive** (dense in VRAM, experts in 44 GB `MemoryHigh` envelope ≈ 17% resident, SSD overflow) | **prefill 0.5 t/s, decode 0.3 tok/s**, coherent output (model began its reasoning preamble); 536 GiB of NVMe reads for a handful of tokens |
| Managed-runtime projection (simulator + measured locality/predictor) | **~3.5–4 tok/s** — a 10–13× gap for Phase 2 to close |

The model *works* end-to-end on the stack — quality intact, tiering functional —
and the entire remaining problem is I/O discipline: expert-granular O_DIRECT reads
instead of 4 KiB faults, decayed-LFU eviction instead of kernel LRU, and
predictor-driven prefetch instead of readahead. Every one of those deltas is
individually measured above.

(Tooling note: current llama-cli ignores `-no-cnv` in favor of its new chat UI and
spins printing prompts on closed stdin — use `llama-bench` or `--single-turn` for
unattended runs.)

### 12. Phase 2 v1 shipped: achilles-pager beats the kernel by 13–33%

`src/pager.cpp` — a llama.cpp wrapper (no fork): expert-slice map from the GGUF,
`ffn_moe_topk`/`post_attn_norm` capture via the scheduler callback, gate-ahead
prefetch (`fadvise WILLNEED`), decayed-LFU eviction (`madvise+fadvise DONTNEED`
via `/proc/self/maps`), residency budget. GLM-4.5-Air Q2, 64-token decode, cold:

| envelope | kernel mmap | pager demand-only | pager +gate-ahead | best |
|---|---|---|---|---|
| 24G (~52% experts) | 1.231 tok/s | 1.355 | 1.597 | **1.633 (+33%)** |
| 16G (~28% experts) | 0.742 tok/s | 0.773 | 0.838 | **0.838 (+13%)** |

What was learned (each iteration measured; see git history):
1. **The topk callback fires before the expert matmul** → same-layer WILLNEED
   turns serial 4 KiB faulting into parallel expert-sized readahead. Biggest
   single win; demand-only already beats the kernel.
2. **Gate-ahead prefetch is additive** (+15–18%) but zero-sum at capacity:
   every prefetch insert evicts something else, so it only pays when prediction
   beats LFU. Fresh prefetches must be protected from immediate eviction
   (score floor), else they're evicted before use — cache pollution was v1.1's
   failure (initially *slower* than the kernel).
3. **Over-fetching saturates the throttled SSD** and starves the demand path —
   the simulator's waste-bandwidth failure mode, reproduced live. fetch=9–10,
   Δ=2–3 is the sweet spot with untrained gate-ahead.
4. **Syscall placement matters**: DONTNEED page-table scans on the compute
   thread cost ~15s per 64 tokens; moved to worker threads.
5. **v1's structural ceiling**: fadvise has no queue-depth control, no
   completion events, and WILLNEED may be dropped under pressure; misses still
   fault at 4 KiB. At 70% hit rate the overlap-perfect budget is ~3.8 tok/s vs
   1.63 measured → **the remaining 2.3× needs Phase 2 v2**: an O_DIRECT expert
   arena with io_uring (explicit QD, completions, expert-granular reads) that
   llama.cpp reads via a custom buffer, plus the trained probe (+20 pp recall
   at long Δ) to make prefetch decisively better than LFU.

### 13. Phase 2 v2 spike: the owned expert arena — 2.1–2.7× over the kernel

`src/arena.cpp`. The route-(a) spike succeeded with a twist: instead of a custom
ggml buffer type (whose eager loading defeats paging), the arena **replaces the
expert tensors' page interiors in-place** (`mmap MAP_FIXED` anonymous over the
model mapping — tensor pointers unchanged, llama.cpp unmodified). Experts are
`pread` into their own addresses (parallel worker pool for demand misses and
prefetch alike; the topk callback fires before the matmul, guaranteeing
validity), evicted with `MADV_DONTNEED` (anonymous memory — no kernel fights).

| GLM-4.5-Air Q2, 64-tok decode | kernel | pager v1 | **arena v2 spike** |
|---|---|---|---|
| 24 G envelope (~52% experts) | 1.23 | 1.63 | **2.61 tok/s (2.1×)** |
| 16 G envelope (~28% experts) | 0.74 | 0.84 | **2.01 tok/s (2.7×)** |
| prefill | 2.1–2.3 t/s | — | **3.9–4.3 t/s** |

- The arena at 16 G beats the kernel at 24 G. Misses cost one expert-sized read,
  not a fault storm; parallel demand loads (a layer's misses scatter across 6
  workers) was worth +45% alone; eviction is free.
- Output is token-identical to v1/kernel runs (same seed) — correctness holds.
- Lessons: layer-window pinning (not whole-pass) or prefill blows the budget;
  the never-executed MTP layer spans two mappings and stays kernel-paged.
- Still on the table for v2 proper: io_uring + O_DIRECT (kill page-cache
  double-buffering, exact QD), the trained linear probe (+20 pp recall at long
  Δ), multi-shard GGUF (needed for GLM-5.2), and a VRAM hot-expert tier.
  Achieved 2.7× of the measured 2.5–4.5× headroom with none of those yet.

### 14. GLM-5.2 through the arena: 2.1× the naive floor; honest ceiling revised

Multi-shard arena (7 shards, 218.6 GiB of experts under management, 38 GiB
budget = 17% resident, dense skeleton in VRAM):

| GLM-5.2 UD-Q2_K_XL, 48-tok decode | tok/s |
|---|---|
| naive three-tier (kernel mmap) | 0.30 |
| arena v2 | 0.53 |
| arena v2 + post-read page-cache drop | **0.617 (2.1×)** |

Air progression at the 24 G envelope meanwhile reached **3.37 tok/s** (kernel
1.23 → v1 1.63 → spike 2.61 → multi-shard+SIMD+inline-scoring 3.37), within
~12% of its overlap-perfect ceiling.

**Revised outlook for GLM-5.2 on this box**: measured global hit rate at 17%
residency is 45% (the Qwen-extrapolated sim assumed ~85–90%, which needs ~2×
the RAM). At 45% hit, per-token demand is ~3.8 GB; even at burst-class 7 GB/s
with O_DIRECT that bounds decode at ~1.5–1.8 tok/s. So on 64 GB of RAM the
realistic v2-complete target is **~1–1.5 tok/s GLM-5.2 decode** (5× naive);
the earlier ~3.5–4 figure requires ~128 GB RAM (where the same math gives
~75%+ hit rates), or Air-class models (which already run at 3.4).

Effective SSD throughput during the run was ~2.3 GB/s vs 7.5 burst — the
remaining io_uring/O_DIRECT + prefetch-throttling headroom is real but bounded
by the bytes, not the latency. Every further tok/s on the beast must come from
hit rate: trained probe, better retention policy, or more RAM.

### 15. v2 complete: io_uring O_DIRECT + policy tuning — final Phase 2 numbers

io_uring with per-worker private rings and O_DIRECT (arena addresses are
congruent with file offsets mod 4 KiB, so alignment is free; 200k+ ops, zero
fallbacks — btrfs stores these GGUFs uncompressed), prefetch throttling
(queue ≥ 96 drops prefetch, demand never starves), tunable LFU decay.

**GLM-5.2 (744B) final on this box** (42 GiB budget ≈ 19% resident, GPU dense):

| config | tok/s |
|---|---|
| naive kernel three-tier | 0.30 |
| arena, demand-only (no prefetch) | 0.42 |
| arena, buffered preads | 0.62 |
| arena + io_uring + pure LFU, 48-tok | **0.70** |
| arena + io_uring, 128-tok steady state | 0.55 (52.5% hit) |

Honest band: **0.55–0.70 tok/s ≈ 2× naive**, content-dependent. Ablations:
gate-ahead prefetch +66%; io_uring/O_DIRECT + cache-drop ~+15% (bytes are the
wall, as §14 predicted); pure LFU beats decayed on short runs (decay matters
for long-session domain shifts — keep it a flag).

**GLM-4.5-Air (110B) final**: 3.2–3.4 tok/s at the 24 G envelope (2.7× kernel,
~12% off the overlap-perfect ceiling). At full 60 GB RAM this model simply
fits: 15.3 tok/s hybrid.

**Remaining backlog (v3)**: trained linear probe (hit rate is the only lever
left on the beast), VRAM hot-expert tier (~8 GB free on the 9070 XT ≈ +4 pp
residency), prefill layer-streaming, and — dominating all of it — a RAM
upgrade to 128 GB, which the recalibrated math says is worth more than every
remaining software optimization combined for GLM-5.2.

### 16. Stability engineering: janitor + backpressure; final day-2 numbers

The bimodal results (0.37–1.00 tok/s on identical configs) were memory-lifecycle
bugs, found via journald swap-peak forensics and fixed in two steps:
1. **Janitor thread**: eviction was accounted instantly but `madvise` drops
   queued behind busy workers → real RSS ran past the budget → 20–27 GB zram
   storms on unlucky runs.
2. **Backpressure**: prefill eviction bursts still outran the janitor; loaders
   now block when resident + pending-drop bytes exceed budget + 2 GiB.

Result: **GLM-5.2 decode 0.757 / 0.740 tok/s across repeat runs (±1%)** — no
bias, LRU, budget 34 GiB. The reproducible day-2 scoreboard:

| GLM-5.2 (744B) | tok/s |
|---|---|
| naive kernel three-tier | 0.30 |
| arena final (stable, quality-exact) | **0.75 (2.5×)** |

**Routing bias — parked by owner decision**: it demonstrably moves hit rate
(45→63%, quality within the 3% bar at 0.05) but showed no *reliable* speed win
in the SSD-saturated regime, and the owner prefers zero quality perturbation.
Revisit post-heatsink/Gen5 (with backpressure now in place) where its 30%
byte reduction converts directly to tok/s. Same story for the trained probes.
Remaining backlog: prefill layer-streaming; learned evictor (Belady gap 19 pp).

### 17. The rolling plan: prediction-driven EVICTION is the byte-free win

Owner's architecture realized: the predictor maintains a rolling plan of
upcoming expert needs; the preloader executes it and the **evictor consults
it** (`plan_hint`: experts predicted for the remainder of the token get a soft
eviction penalty — approximate Belady). Measured on GLM-5.2:

| config | hit rate | tok/s (3 runs) |
|---|---|---|
| pre-plan baseline | 41.2% | 0.74–0.76 |
| **plan-aware eviction (gate-ahead hints)** | **55.5%** | **0.83 / 0.90 / 0.945** |
| + far-stage Δ8 probe *fetching* | 51.7% | 0.77–0.82 (worse — adds bytes) |

Three structural lessons, now measured twice each:
1. On SSD-bound hardware, **spending prediction accuracy on eviction is free**
   (no bytes added); spending it on more fetching backfires.
2. Cross-token prediction is dead on GLM-5.2 (26% ceiling; probes can't beat
   it; prev-token trace features add nothing) — plans must roll within-token.
3. Warm-started Δ8 probes recall 52%@8 (+11 pp over gate-ahead) — the hint
   horizon can extend to ~100 ms; use them as *hints*, not fetches.

**Day-2 final: GLM-5.2 at 0.83–0.94 tok/s, quality-exact — 3× the naive floor,
software only.** Projected with owner's planned heatsink + Gen5 drive (+
optional headless ~47 GiB budget): **~2.5–3.5 tok/s**, at which point RAM
bandwidth becomes the wall.

### 18. The "prefill thrash" was a crash bug; layer-streaming is a +22% prefill win

Debugged on Qwen3-30B (13× smaller than GLM — POLITE-profile, desktop unaffected).
Reproduction attempt #1 found the truth immediately: **every long-prompt run
crashed, pstream on or off** — `GGML_ASSERT(offset + size <= ggml_nbytes)` inside
our own eval callback.

Root cause: the gate-ahead norm hook gated "single token" on `ne[1] == 1`, but the
graph hands norms over as reshaped 3D `[n_embd, 1, n_tokens]` — `ne[1]` is *always*
1, tokens live in `ne[2]`. For layers 0..n-2 this silently scored gate-ahead from
token 0's hidden during prefill (wrong, in-bounds). The last layer is sliced to
output rows only, and non-final prefill ubatches have **zero** output rows:
`ne=[2048,1,0]`, `nbytes=0` — the 8 KB hidden-state read aborted. Short prompts
(one ubatch, always has an output row) never trigger it, which is why every short
benchmark passed and every long-prompt run "mysteriously" died — including all
GLM prefill A/B attempts. Fix: gate on `ggml_nrows(t)==1` + a zero-element guard.

With the crash gone, the first clean pstream A/B (Qwen, ~2900-token prompt,
5 GiB budget ≈ 30% residency, 2 runs each, page cache evicted between runs):

| config | prefill tok/s | decode tok/s | demand loads | prefetch loads |
|---|---|---|---|---|
| pstream=0 | 30.5 / 30.5 | 4.2–4.4 | 27,8xx | 1,7xx |
| **pstream=1** | **37.1 / 37.2 (+22%)** | 4.2–4.4 | **1,031** | 37,80x |

Streaming the next layer's full expert set during the current layer's compute
eliminates 96% of demand stalls at identical byte traffic. Decode untouched.

The feared "0↔12G memory oscillation" appears **in both configs equally**
(max 1s swing 7–13 GB): it is the buffered-read page cache being reclaimed in
sweeps (janitor `fadvise` + cgroup pressure) — a bounded sawtooth around a
~8 GB mean, not a leak and not pstream's doing. The GLM staged-run "thrash"
was this sawtooth plus the crash loop, misattributed. `--pstream 1` is back on
in bench/staged_run.sh.

**GLM-5.2 confirmation (same day, FULL profile after two more environment
bugs — see below):** first successful long-prompt GLM runs ever. ~2700-token
prompt, 30 GiB budget, hybrid on the 9070 XT:

| config | prefill tok/s | demand loads | decode (8 tok, post-prefill cold cache) |
|---|---|---|---|
| pstream=0 | 6.83 | 101,781 | 0.525 |
| **pstream=1** | **8.67 (+27%)** | **2,756** | 0.530 |

Envelope held both legs (peak 45.0 G < 48 G max, mean ~41 G). Naive three-tier
prefill was ~0.5 tok/s — the managed runtime prefills **~17× faster**.

Two environment gotchas cost four dead runs before this measurement, both now
guarded in the repo:
1. `ttm.pages_limit=1048576` (from the TTM-leak fix) also caps **total live
   GTT at 4 GB** — amdgpu sizes its GTT domain from it at boot. Hybrid GLM
   needs ~5.5 GB GTT (measured); the karg is now 8388608 (32 G default), with
   only `page_pool_size` keeping the 4 GB pool cap that actually fixes the leak.
2. **Vulkan device enumeration order flips across boots** — the hardcoded
   `GGML_VK_VISIBLE_DEVICES=1` landed GLM's 25 GB dense skeleton on the iGPU
   (`amdgpu_vm_validate` ENOMEM → `vk::DeviceLostError` at the first ubatch
   upload). bench/vkdev.sh now resolves the dGPU index by name at runtime.

### 19. Reuse-distance eviction: hazard analysis → gap/pop policy (+4–6% decode)

Attacking the LRU→Belady gap (19 pp at 19% residency). Measured the reuse
hazard on the GLM streams (scripts/reuse_hazard.py): **h(g) is monotone
decreasing** (0.29 at gap 1 → ~0.03 plateau — so LRU's recency instinct is
right and there is no periodic structure), and **popularity multiplies the
hazard ~3× at every gap** — the one signal LRU wastes.

Model-based policies that ignore the burst autocorrelation LOSE outright
(geometric −10 pp, 2-state Markov −8 pp vs LRU — measured, sim). The winner
is minimal: **evict max staleness / EWMA-popularity** ("gap/pop"). Sim: +2.2 pp
hit over LRU at 14% residency, +1 pp over idealized plan-LRU. Everything
beyond (~17 pp to Belady) requires future-stream knowledge — features cannot
close it; speculation×plan lookahead is the only route.

Live (--policy reuse, one EWMA/expert, decay LUT, prefetch grace period):

| decode | LRU | reuse |
|---|---|---|
| Qwen3-30B, 256 tok (×2 pairs) | 8.47 / 8.53 tok/s (.873) | **8.99 / 9.08 (+6%)** (.875) |
| GLM-5.2, 192 tok | 0.616 tok/s (.760) | **0.639 (+3.7%)** (.766) |
| GLM-5.2, 48 tok | 0.643 (.710) | 0.641 (.714) — EWMA not yet converged |

Signature in both models: ~equal hit rate but **7–13% fewer prefetches and
6–12% fewer evictions** — the policy stops evicting experts that were about
to be re-fetched. Needs ~50+ tokens to warm up (α=0.02). Now the default in
bench/staged_run.sh.

### 20. Prompt-conditioned working-set prediction: real structure, bounded payoff

Owner's idea: a tiny NN predicting future expert usage from the prompt.
Validated on 20 diverse-domain prompt traces (Qwen, 192-tok decodes,
scripts/promptset_analysis.py; traces/qwen-prompts/ keeps hidden states as
future training inputs).

**The structure is real.** Decode usage cosine: 0.315 across prompts vs 0.854
within a prompt (split-half). Working sets are strongly prompt-specific — and
per §17, per-token *order* is unpredictable, so the aggregate working set is
the right target.

**But eviction converts little of it.** Hit-rate ladder (2nd-half, 25% cap):
LRU .824 → deployed reuse policy .844 → +global prior .841 (nothing — the
online EWMA already learns it) → **+oracle prompt prior .861**. A PERFECT
predictor is worth +1.6 pp ≈ +4–6% decode; a real NN, likely half that.
The online EWMA converges to the prompt's distribution within ~50 tokens, so
a prior mostly buys the early window plus lag correction.

**Free-prior variant also tested**: the prefill sweep's own routing histogram
(zero parameters, available in the callback). It correlates with decode usage
at 0.747 — yet adds ~0 pp in the sim (short prompts → noisy counts; the EWMA
catches up too fast for the prior to matter).

**Disposition: PARKED.** The idea is validated as real-but-bounded; the same
future-knowledge gap is attacked ~10× harder by speculation×plan lookahead
(verified multi-token routing, 17 pp Belady headroom). Revisit if long-prompt
workloads (where the prefill histogram is high-quality) become the norm.

### 21. The decode "regression" that wasn't: over-fetch + a warm-cache headline

Controlled re-measurement (2026-07-13 overnight) found GLM-5.2 decode at
0.62–0.64 tok/s where §17 recorded 0.83–0.94. Bisect eliminated, in order:
SSD health (6.5 GB/s raw O_DIRECT), code (yesterday's binary rebuilt: 0.616),
GPU placement (-ngl 40/55/99 all 0.55–0.64; the 3.9 GB GTT spill is NOT the
tax), CPU power profile (performance == balanced within noise).

New decode-detail instrumentation (stall / io bytes / per-stream bw) found it:
**stall-bound (53 s of 75 s) while moving 309 GB per 48 tokens — 4× true
demand — with the SSD already saturated (6×1.0 GB/s streams)**. Every wasted
prefetch byte delays a demand miss the compute thread is blocked on. The
--fetch sweep (cold, ±0.005 reproducibility):

| --fetch | decode tok/s | decode IO |
|---|---|---|
| 1 | 0.799 | 163 GB |
| **2 (new default)** | **0.803** | 168 GB |
| 5 | 0.767 | 199 GB |
| 10 (old default) | 0.643 | 309 GB |

With `--policy reuse` + pstream on top: **0.826 tok/s cold** — the new honest
headline config. The §17 0.83–0.94 numbers came from a back-to-back triple
with no page-cache eviction between runs and a tell-tale rising pattern
(0.83→0.90→0.945): warm-state flattered, not fraudulent, but not the cold
number either. README updated per owner's honest-numbers directive.

Also fixed: pstream trigger n_tokens≥16 → ≥64 (a 23-token prompt was
full-layer-streaming 2.9 GB/layer and halving short prefill: 0.54 → 1.25).

Meta-lesson for every future benchmark here: **on a saturated SSD, tok/s is
set by total bytes moved, not hit rate** — hit .796 at fetch 10 lost to hit
.581 at fetch 1 by 25%. Optimize bytes, then overlap, then hits.

### 22. io_uring exonerated and enabled: +33%, GLM-5.2 crosses 1 tok/s

`--no-uring` had been baked into every script during the memory-leak hunt;
the TTM root cause (§16/§18) exonerated io_uring, and §21 made bytes the
binding constraint — so the buffered path's page-cache copy + cgroup-reclaim
tax on every miss byte became the next target. A/B at the §21 best config
(fetch 2, reuse, pstream), two alternating pairs, cold:

| decode | buffered (--no-uring) | **io_uring O_DIRECT** |
|---|---|---|
| tok/s | 0.842 / 0.842 | **1.122 / 1.122 (+33%)** |
| stall | 39.2 s / 48 tok | 25.3 s |
| cgroup peak | 43 G | 43 G (identical — no leak, fear retired) |
| O_DIRECT fallbacks | — | 0 |

**GLM-5.2 decode ≥ 1 tok/s for the first time: 1.12 cold, 3.7× naive.**
io_uring is now the default in every bench script; `--no-uring` remains a
flag for filesystems that reject O_DIRECT.

### 23. Expert-major shadow + combined headline (2026-07-13 morning)

Shadow repack (scripts/repack_shadow.py + arena --shadow): experts rewritten
contiguously in original-aligned windows (§ the O_DIRECT overshoot contract).
**Token-identical at both scales.** Qwen decode +15.6% (many small slices
merge); GLM decode neutral (slices were already ~4 MB) but GLM long-prompt
prefill +12% on top of layer-streaming (sequential reads get closer to raw).
Repack cost: 2.7 min for 219 GiB (the drive is faster than my estimates).

Combined headline suite (fetch 2 + reuse + io_uring + shadow + pstream,
throughput-performance profile, cold, all runs token-gated):

| GLM-5.2 UD-Q2_K_XL | result |
|---|---|
| decode, 96 tok × 3 | **1.113 / 1.120 / 1.124 tok/s** (3.7× naive) |
| prefill 2697 tok, pstream on | **9.69 tok/s** (vs 7.58 off, +28%; ~19× naive) |
| decode after long prefill | 0.98 (was 0.53 pre-uring/reuse) |
| spec 4 (n-gram) | 1.088 — 78% acceptance but only 9 draft attempts in 96 tok |

Speculation note: the n-gram draft only fires when the recent context repeats
an n-gram; on non-repetitive generation it's idle (9 tries/96 tok) and its
overhead nets slightly negative. Kept opt-in. The spec×plan callback work
(per-row gate-ahead during verify batches) is in and correct, but the
draft-source is the limiting reagent — a real cross-token plan needs a
draft model or richer lookup, which re-opens the parked draft-model track.

Cumulative, measured same-day, same box: 0.643 (old defaults, cold) →
0.803 (fetch fix) → 0.826 (+reuse) → 1.122 (+io_uring) → **1.12 confirmed**
at 96 tokens with shadow; prefill 6.83 → 9.69.

### 24. MTP self-drafting: built, works, and loses to its own byte bill

Ported GLM-5.2's NextN/MTP head to llama.cpp (private branch glm-dsa-mtp —
first glm-dsa MTP inference implementation; loader + deepseek2 h_nextn export
+ graph_mtp) and integrated as arena --spec-mtp. **It works**: the model
drafts for itself, 40% raw acceptance at temp 0.7, 62% with --spec-pmin 0.6.

Getting there cost three whole-system crashes (freezes + a kernel panic),
all one bug: **glm-dsa lacked the per-arch nextn KV filter (STEP35 has it),
so the MTP draft context allocated + memset a ~34 GB full-model KV** —
unreclaimable anon + invisible GTT. Root-caused with a caged harness
(src/mtp_probe.cpp; MemoryMax=8G, MemorySwapMax=0, gdb attach mid-climb).
Cage method + swap-0-for-experiments are now standing policy. Bonus finds:
CPU repack would silently break the MAP_FIXED invariant (now forced off);
full-size batch reserve on the MTP ctx wastes GBs (now draft-sized).

The economics, measured (64-tok decode, budget 28, hybrid, cold):

| config | acceptance | tok/s | decode IO |
|---|---|---|---|
| **no speculation** | — | **1.099** | **217 GB** |
| mtp d3 pmin .6 | 62% | 0.924 | 289 GB |
| mtp d3 pmin 0 | 39% | 0.556 | 540 GB |

Verify batches pay for the UNION of drafted tokens' expert sets, and
cross-token expert overlap is ~26% (§17's constant, striking a third time).
On a byte-bound SSD regime (§21), speculation's extra bytes exceed the
tokens it saves at any achievable acceptance. **PARKED with revival
conditions**: hit rate ≥ ~85% (bytes stop binding), Gen5-class SSD, or
RAM-rich boxes where verify batches amortize RAM bandwidth instead of SSD.
Known issue: --spec-pmin ≥ 0.8 core-dumps (untriaged; irrelevant while parked).

### 25. Stall packing: decode hits its latency floor; prefill jumps 2.3×

(A) demand-reserved workers + (B) batched io_uring submission (4 experts/
submit, QD 6→24 experts) landed with token-identical output and **halved
worker IO time (125 s → 58 s per 64 tokens; per-stream 1.73 → 3.76 GB/s)** —
and decode wall time did not move (1.084 vs 1.099). Decode stall is
**latency-bound, not throughput-bound**: ~3 misses/layer × 76 layers = 76
synchronous NVMe round-trips per token, and rounds can't be compressed
without earlier discovery — which is the prediction-recall ceiling measured
dead three times (§17, §20, §24). **Measured software floor for decode at
hit .63 / Gen4: ~1.1–1.2 tok/s.** Remaining decode gains are hardware
(Gen5 halves both bytes AND round-trip cost) or budget (headless +8–10%).

The same packing pays where throughput binds: **prefill ubatch scaling**.
Each ubatch sweeps the full expert store; fewer, larger ubatches:

| -ub | prefill (2697-tok prompt) |
|---|---|
| 512 (old default) | 9.52 tok/s |
| 1024 | 16.59 (+71%) |
| **2048 (new default)** | **22.40 (+131%; ~45× naive)** |

GTT held throughout (pages_limit fix); envelope swap-0 clean.

### 26. The decode round, fully dissected: 74% of drive physics, case closed

Instrumented every demand round (handoff, demand-IO, per-round stall) and ran
four targeted experiments, all token-identity-gated:

| experiment | decode | verdict |
|---|---|---|
| fetch re-sweep 4/6/8 under uring+shadow | 1.065/1.014/0.939 | fetch=2 stands; bytes still rule |
| prefetch-yield (drive queue clears for demand rounds) | 1.107 | +1% kept |
| demand fan-out (1 expert/worker, was 4 serialized) | **1.116** | +1.5% kept |
| 2 MB demand sub-reads (deeper NVMe QD) | 1.075 | reverted; command overhead > QD gain |

Round anatomy at the end: handoff 0.01 ms (software is free), ~7.3 ms/round
for ~35 MB of missed experts = **74% of the drive's absolute 6.5 GB/s**.
The stall is bandwidth at round granularity; there is no software left in it.
Decode on this hardware: **1.12 tok/s, measured to its wall from two
directions.** Gen5 moves the wall; nothing else does.

### 27. Budget sweep: +2.7% at 32 GiB; 36-40 GiB is headless-only

budget 28 -> 32 GiB: 1.116 -> **1.146 tok/s** (hit .629 -> .652), on the
predicted ~1.4%/4GiB curve. 36/40 GiB envelopes (52-56G max) do not fit
beside a logged-in desktop (~11G); preflight auto-skips them. Extrapolated
headless ceiling ~1.17-1.19. budget 32 is the new FULL default
(staged_run.sh, envelope 46/48G).

### 28. Burst mitigation, concluded: the answer to "can prediction still pay?"

Owner asked whether predictive load/eviction could still be rescued by other
techniques ("eager processing"). Research + three measured experiments:

- **Literature (2025-26)**: SP-MoE, MoE-SpeQ, ADEPT, Fiddler/HybriMoE — the
  field's answers are draft-predicted expert prefetch, domain priors, and
  compute/IO pipelining, all targeting GPU<->DRAM at PCIe speeds. We built or
  independently invented each; they lose here because our tier is 4x slower
  per byte and §21's byte-law bites first.
- **Expert-granular eager execution** (compute resident experts while misses
  stream): killed by arithmetic the owner spotted — the layer's output needs
  ALL experts, so only ~1.2 ms/layer of resident-expert compute can hide
  inside the 7.3 ms burst → +8-12% for kernel-level fork surgery. Declined.
  (True 2-token pipelining through stall windows needs llama-internals
  rearchitecture — noted for a future phase, not this hardware push.)
- **Prefetch quanta** (2 MB chunks so demand rounds start against a
  drainable queue): keeps higher fetch from LOSING (fetch 4: 1.124 → 1.144)
  but cannot make it WIN — the plateau is 1.144-1.149 across fetch 2-4.
  Kept for robustness; default fetch 2 stands.

**Final state: decode 1.149, rounds at ~74% of drive physics, and every
"other technique" now has a measurement, not an opinion, attached to its
grave. Bursts are mitigated to the extent this drive's command model allows;
the rest is Gen5.**

### 29. Gen5 drive installed: decode 1.44, prefill 25.1 — both walls relocated

Samsung 9100 PRO 1TB (Gen5 x4, 13.2 GB/s read measured). Migration: shards
copied in 63 s, shadow regenerated on-drive in 4 min, models/ symlinked
(arena now realpath()s shard paths — /proc/self/maps shows resolved paths,
so symlinked model dirs previously matched nothing). Token-identity gate
passed on the new drive before any benchmark.

| GLM-5.2 | Gen4 (final) | **Gen5** | naive× |
|---|---|---|---|
| decode (96 tok, b32) | 1.146 | **1.443** (triple 1.421/1.422/1.422 at b30) | **4.8×** |
| prefill 2697 tok, ub2048 | 22.4 | **25.1** | **~50×** |
| post-prefill decode | 0.98 | 1.24 | |

The round anatomy validated its own model: avg round 7.05 → 4.40 ms —
the ~2.7 ms transfer term halved on cue, the ~1.7 ms fixed tail didn't.
**Decode's next wall is NVMe latency + RAM compute, not bandwidth.**
Prefill barely moved because ub2048 already left the Gen4 drive 45% idle:
it is now CPU-compute-bound (31% drive utilization) — its endgame here.
MTP spec re-checked on Gen5: 1.385 vs 1.42 plain — §24 verdict stands.

Pending when the +32 GB RAM arrives: budget re-sweep (b56-60 → hit ~.77
should put decode ~1.7-1.8), and the Air MTP revival program (task #26).

## Implications for the runtime design

1. Cache = decayed-LFU over experts, sized as large as RAM allows; static popularity
   only decides SSD layout and cold-start placement.
2. Prefetch = warm-started linear probes: Δ=1–2/m=16 for the near tier, Δ=8/m=16–32
   for the SSD tier (90%+ recall measured); probes cost one extra router-sized matmul
   per layer — negligible. Cross-token horizons remain to be studied (Phase 3).
3. Misses concentrate in low-margin routing events → substitution fallback is
   promising; measure its quality cost early.
4. OLMoE caveat: 64 experts/layer is 4× denser than GLM-5.2's 256; per-layer sparsity
   patterns may differ at scale. Qwen3-30B (128 experts) traces will interpolate, and
   GLM-4.5-Air (same family as target) confirms.

## Reproduce

```
scripts/trace_olmoe.py --domains all --max-new 1024   # ~90 min, 2 workers
scripts/analyze_olmoe.py                              # charts + JSON
scripts/trace_qwen.sh                                 # llama.cpp expert-ID traces
scripts/sim_paging.py --traces traces/olmoe ...       # what-if throughput model
```
