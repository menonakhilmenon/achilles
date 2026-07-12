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
