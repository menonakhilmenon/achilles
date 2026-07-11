# Research: Hosting a Frontier MoE Model on ~36 GB VRAM via Expert Paging

*Last updated: 2026-07-11*

## 1. Problem statement

Goal: run a frontier-scale Mixture-of-Experts model (target: **GLM-5.2**, 744B total /
~40B active) on a device with ~36 GB VRAM, by keeping only the *dense* weights and a
small cache of experts resident in VRAM, and paging the remaining experts in and out
from lower tiers (host RAM, then NVMe SSD). Two core mechanisms:

1. **Prefetch/evict around the MoE phase** — bring an expert into VRAM just before its
   layer executes, evict it after, so peak VRAM ≪ total weights.
2. **A learned expert predictor** — a small NN that predicts which experts a token will
   route to, several layers (or tokens) ahead of time, so transfers overlap with compute
   and never sit on the critical path.

## 2. Target model: GLM-5.2 architecture (from `zai-org/GLM-5.2` config.json)

| Field | Value |
|---|---|
| Total / active params | 744B / ~40B |
| Layers | 78 (first 3 dense, **75 MoE layers**) |
| Routed experts per layer | **256**, top-8 per token, sigmoid scoring (`noaux_tc`) |
| Shared experts per layer | 1 (always active) |
| hidden_size / moe_intermediate_size | 6144 / 2048 |
| Attention | MLA (kv_lora_rank 512, q_lora_rank 2048) + DSA sparse attention (index_topk 2048, "IndexShare" — DSA layers reuse the previous full layer's top-k selection) |
| Vocab | 154,880 |

### Derived numbers (the core feasibility math)

- **Params per routed expert**: 3 matrices (gate/up/down) × 6144 × 2048 = **37.75M params**.
  - bf16: 75.5 MB · Q8: ~40 MB · Q4: **~20 MB** · Q3: ~15 MB · Q2: ~12 MB
- **Total routed experts**: 75 × 256 = **19,200 experts ≈ 725B params = 97.4% of all weights.**
  This is the whole premise: the non-expert "dense skeleton" (attention, DSA indexer,
  3 dense FFN layers, shared experts, embeddings) is only **~19B params ≈ 10 GB at Q4**.
- **KV cache is cheap** thanks to MLA: ~(512 + 64) values/token/layer → roughly
  ~0.1 MB/token bf16 across 78 layers → 32K context ≈ 3.5 GB (less with q4 KV cache).
- **VRAM budget on a 36 GB device (Q4 weights)**:
  - Dense skeleton ~10 GB + KV/activations/workspace ~4–6 GB → **~20 GB left for the expert cache**
  - ≈ **1,000 experts resident at Q4 (5.2% of 19,200)**, or ~1,700 at Q2.
- **Per-token expert demand (decode, batch 1)**: 75 layers × 8 = **600 expert activations/token**.
  Worst case (0% cache hits) = 600 × 20 MB = **12 GB of weight traffic per token**.

### The bandwidth wall (why cache hit rate + prediction is everything)

| Link | Realistic bandwidth | Time to move 12 GB (0% hit) | Time at 75% hit (3 GB) | at 90% hit (1.2 GB) |
|---|---|---|---|---|
| NVMe PCIe 4.0 x4 | ~6–7 GB/s | ~1.8 s/tok | ~0.45 s | ~0.18 s |
| NVMe PCIe 5.0 x4 (or 2× Gen4 RAID0) | ~12–14 GB/s | ~0.9 s/tok | ~0.23 s | ~0.09 s |
| Host RAM → VRAM (PCIe 4.0 x16, pinned) | ~25 GB/s | ~0.5 s/tok | ~0.12 s | ~0.05 s |
| Host RAM → VRAM (PCIe 5.0 x16) | ~50 GB/s | ~0.24 s/tok | ~0.06 s | ~0.024 s |

Takeaways:

- **A pure SSD↔VRAM design cannot fully "hide" the traffic** at low hit rates: per-token
  weight traffic (GBs) exceeds link bandwidth × per-token compute time by an order of
  magnitude. Prediction doesn't reduce bytes moved — it only removes latency from the
  critical path. **The hit rate of the VRAM (and RAM) cache determines throughput; the
  predictor determines whether the residual misses stall the pipeline.**
- **Host RAM is a mandatory middle tier**, not an optional one. Every GB of RAM directly
  buys hit rate for the fast tier. With e.g. 96–192 GB RAM caching hot experts, SSD
  becomes a *capacity* tier (cold experts, long-tail), which is exactly where it works well.
- Decode at batch 1 is memory-bound anyway (little compute to hide behind), so overlap
  must come from **pipelining across layers and across tokens**, not from "compute vs.
  load" overlap alone. Prefetch lookahead must be ≥ several layers.
- Interesting consequence of top-8-of-256 sparsity: per layer per token only 8/256 = 3.1%
  of experts fire, but 600 activations/token spread across layers means naive LRU over a
  5% cache gets meaningful but insufficient hit rates — skew + temporal locality (see §4)
  must be exploited deliberately.

### Prefill is a different regime

Prefill touches essentially *all* experts (many tokens × top-8 each). The right strategy
there is the opposite of decode: **stream experts layer-by-layer sequentially** (perfectly
predictable, sequential SSD reads at full bandwidth, batch all prompt tokens per expert).
Prefill throughput ≈ total-expert-bytes / SSD-bandwidth per pass (e.g. Q4: 380 GB / 7 GB/s
≈ 55 s per full pass, amortized over the whole prompt — and chunked prefill amortizes it
further). The design must treat prefill and decode as two separate scheduling modes.

### CPU-first execution strategy (POC simplification)

Decision (2026-07-11): the POC does **all compute on CPU** — the GPU enters later as an
accelerator tier. Consequences:

- **The hierarchy collapses to two tiers: RAM ← SSD.** RAM plays VRAM's role (compute
  memory + expert cache), SSD is the backing store. This deletes the hardest systems
  work (CUDA streams, pinned staging, VRAM cache manager) while leaving both core ideas
  — prefetch/evict around the MoE phase, and the learned predictor — fully testable.
  Overlap is also natural on CPU: SSD reads are DMA and cost almost no CPU, so a
  prefetch thread genuinely runs "behind" the compute threads.
- **The expert cache is huge.** All of RAM (minus overhead) instead of ~20 GB of VRAM →
  much higher hit rates for the same model. llama.cpp's mmap already provides a *passive*
  version of this (OS page cache + demand paging); our value-add is *active*,
  predictor-driven prefetch and eviction. That makes stock-llama.cpp-with-RAM<model the
  perfect control experiment.
- **New ceiling: RAM bandwidth, not PCIe.** CPU decode is memory-bound; per token it must
  read all *active* weights from RAM ≈ active_params × bytes/param. Ceilings
  (100% cache hit, Q4 ≈ 0.55 B/param):

  | Model | Active bytes/token | Dual-ch DDR5 (~80 GB/s) | Strix Halo (~256 GB/s) | 12-ch EPYC (~400+ GB/s) |
  |---|---|---|---|---|
  | GLM-4.5-Air (A12B) | ~6.6 GB | ~12 tok/s | ~38 tok/s | ~60 tok/s |
  | Qwen3-235B (A22B) | ~12 GB | ~7 tok/s | ~21 tok/s | ~33 tok/s |
  | GLM-4.6 (A32B) | ~18 GB | ~4.5 tok/s | ~14 tok/s | ~22 tok/s |
  | GLM-5.2 (A40B) | ~22 GB | ~3.5 tok/s | ~12 tok/s | ~18 tok/s |

  So on a normal desktop, CPU-only GLM-5.2 tops out ~3–4 tok/s even if paging is
  *perfectly* hidden — acceptable for the POC, and the reason the GPU tier returns for
  the endgame (36 GB of ~1 TB/s memory holding the dense skeleton + hot experts).
- **Prefill hurts most on CPU** (compute-bound, no GPU): expect ~tens of tok/s prefill.
  Fine for the POC; the layer-streaming prefill mode (above) still applies to the SSD tier.

### Staging-model ladder

| Stage | Model | Size on disk | Why |
|---|---|---|---|
| Science / tracing | Qwen3-30B-A3B | ~18 GB Q4 | Fits everywhere, minutes-fast iteration |
| Mechanics | GLM-4.5-Air (106B/A12B) | ~60 GB Q4 | GLM lineage; big enough to exercise paging on a 32–64 GB RAM box |
| **POC (~150 GB class)** | **GLM-4.6 (357B/A32B) @ Q3** (~145–170 GB) or Qwen3-235B-A22B @ Q4 (~133 GB) | > RAM | Must exceed the box's RAM so the SSD tier is real, not theoretical. GLM-4.6 keeps router/family continuity with the target; Qwen3-235B has smaller experts (~10 MB @Q4) and cleaner quants |
| Endgame | GLM-5.2 (744B/A40B) | ~245 GB Q2 / ~380 GB Q4 | The beast |

## 3. Prior art survey

The two core ideas both exist in the literature (which de-risks them — they work — but
the novelty must come from the combination, the SSD tier, and the target scale).
Curated list: [awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference/).

### Prediction-based prefetching (idea #2 has strong precedent)
- **[ProMoE](https://arxiv.org/abs/2410.22134)** (SJTU, 2024): the closest match. Trains a
  small **learned predictor** (per-layer MLP on intermediate hidden states) with
  "stride prefetching", plus skew-aware caching. 2.20×/2.07× speedup prefill/decode over
  reactive caching on edge GPUs.
- **[Fate](https://arxiv.org/pdf/2502.12224)** (2025): uses gate inputs from *adjacent
  layers* as prediction signal — hidden states drift slowly across layers, so layer *l*'s
  hidden state predicts layer *l+1*'s routing with high accuracy (~90%+ reported) at zero
  extra training. Good cheap baseline predictor.
- **Pre-gated MoE** (ISCA'24): moves the router itself one layer earlier (architecture
  change; needs fine-tuning). We likely can't retrain GLM-5.2, so predictor-on-top is
  the right call — but their result confirms cross-layer routing predictability.
- **[MoE-Beyond](https://arxiv.org/html/2508.17137v1)** (2025): learned expert-activation
  predictor for edge devices (79.9% recall vs 51.3% for heuristics on unseen workloads).
- **[MoE-SpeQ](https://arxiv.org/abs/2511.14102)** (2025): uses a small **draft model** to
  speculate future *tokens*, then prefetches the experts those tokens need — extends the
  prediction horizon across tokens, amortizing I/O. Up to 2.34× over SOTA offloading.
  (Natural stretch goal: speculative decoding and expert prefetch share one draft model.)
- **[DAOP](https://arxiv.org/pdf/2501.10375)** (2025): predictive pre-calculation,
  data-aware offload decisions.

### Caching / eviction policy
- **[MoE-Infinity](https://arxiv.org/abs/2401.14361)**: sequence-level Expert Activation
  Matrix tracing; observes decode activates a *handful* of experts per request with strong
  temporal locality. 3–20% of experts activated per request; 30–46% reused more than once.
- **[FlashMoE](https://arxiv.org/pdf/2601.17063)** (2026): **ML-based cache replacement for
  SSD-backed MoE on edge devices** — closest to our SSD story; worth a careful read.
- LCP ("least cache priority") policies combining recency + long-tail frequency beat
  LRU/LFU by ~1.5–12 pp hit rate at 30 experts/layer cached.
- **[ReMoE](https://arxiv.org/html/2605.27081)** (2026): router fine-tuning to *increase*
  expert reuse in memory-constrained inference (stretch: cache-aware router bias).

### Alternative miss-handling (fallbacks worth stealing)
- **[Fiddler](https://arxiv.org/abs/2402.07033)**: on a miss, **compute the expert on CPU**
  (moving activations ~24 KB is cheaper than moving 20 MB of weights when the expert is
  cold). Essential escape hatch — CPU FLOPS for 8×(6144×2048×3) GEMV is trivial.
- **[HOBBIT](https://arxiv.org/html/2411.01433v1)** (2024): mixed-precision fallback — keep
  a low-bit (e.g. Q2) copy of every expert *always resident or cheap to load*, fetch the
  high-bit version only when prediction gave enough lead time. Token still computes with
  *something* — converts stalls into small quality noise. 9.93× over llama.cpp offload.
- **EdgeMoE / SwapMoE / AdapMoE**: per-expert bitwidth adaptation + preload on edge.

### Systems baselines (what we must beat / build on)
- **llama.cpp `--cpu-moe` / `--override-tensor`**: keeps all experts in host RAM, dense on
  GPU ([guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)).
  Reported ~1–9 tok/s for models of this class on consumer boxes; GLM-5.2 Q2 needs
  ~245 GB total memory ([Unsloth guide](https://unsloth.ai/docs/models/glm-5.2): quants
  run 223 GB @1-bit → 372–475 GB @4-bit). **mmap already gives crude SSD paging** — with
  page-cache thrash and zero prefetch intelligence. This is our "existing tricks" baseline.
- **[KTransformers](https://github.com/kvcache-ai/ktransformers)**: experts on CPU (AMX/AVX
  kernels), attention+shared on GPU (~21 GB VRAM for DeepSeek-class); strong prefill. The
  "big RAM, no SSD" state of the art.
- **GPUDirect Storage / DeepNVMe**: direct NVMe→VRAM DMA, skipping CPU bounce buffers
  ([NVIDIA](https://developer.nvidia.com/blog/gpudirect-storage/)); single Gen5 drive
  ~14 GB/s. Relevant for the SSD tier; io_uring + pinned bounce buffers is the portable
  fallback and often within ~10–20% of GDS for large reads. **Caveat: GDS is
  NVIDIA-only — the current box has an AMD RX 9070 XT (16 GB), so on this hardware the
  GPU path is ROCm/Vulkan with io_uring staging, no GDS.**

### Positioning: what would actually be novel here

1. **Scale**: published expert-offload systems target Mixtral-8x7B / Phi-MoE / Switch-class
   models (≤ ~50B). Nobody has demonstrated a *744B* model on a 36 GB card at usable speed.
2. **Three-tier hierarchy** (VRAM ← RAM ← NVMe) with a learned predictor driving *both*
   promotion boundaries. Most papers do two tiers (VRAM←RAM); SSD-tier work (FlashMoE) is
   edge-scale.
3. **Deadline-scheduled prefetching**: treat each MoE layer as a deadline; the predictor's
   lookahead Δ and per-tier latency decide from which tier each predicted expert must be
   promoted, and mixed-precision (HOBBIT-style) or CPU-compute (Fiddler-style) fallbacks
   absorb mispredictions. That unified scheduler is a real contribution.

## 4. What the traces will likely show (hypotheses to verify in Phase 1)

From the literature, we expect — and must measure on GLM-5.2 specifically:

- **Skew**: expert popularity is long-tailed per domain; a static "hot set" per layer
  captures a large fraction of activations (helps size the RAM tier).
- **Temporal locality**: consecutive tokens reuse 30–46%+ of experts; higher within a
  domain-coherent generation.
- **Cross-layer predictability**: hidden state at layer *l* predicts routing at *l+Δ* with
  accuracy decaying in Δ; Δ=1 is ~90%+ (Fate), and we need to measure the recall@8 curve
  vs Δ for GLM-5.2's sigmoid/noaux router (all published numbers are softmax routers).
- **Router stability**: GLM-5.2's `moe_router_dtype: float32` + sigmoid scoring suggests
  routing decisions are sharp; need to check margin distribution (how often is the 8th
  expert nearly tied with the 9th — those are the hard-to-predict ones, and also the ones
  where a substitution costs least quality).

## 5. Key risks

| Risk | Mitigation |
|---|---|
| Per-token traffic simply exceeds achievable bandwidth at low hit rates | RAM tier sizing is the primary lever; Q2/Q3 experts (Unsloth dynamic quants hold up well: ~82% quality at 2-bit); HOBBIT-style low-bit fallback |
| Predictor recall too low at the lookahead the SSD tier needs (Δ large) | Cascade: cheap Δ=1 gate-ahead for RAM→VRAM, learned long-horizon predictor only for SSD→RAM promotion (where a miss costs less — it just stays a RAM miss) |
| Building a full GLM-5.2 runtime (MLA + DSA kernels) from scratch is months of work | Build on an existing engine (llama.cpp fork or KTransformers or vLLM) for the model math; own only the expert-cache/prefetch subsystem. Prototype the science in PyTorch on a small MoE first |
| No GLM-5.2-Air exists (community is asking; nothing released) | Stage on GLM-4.5-Air (106B/A12B) and/or Qwen3-30B-A3B for iteration speed; the machinery is model-agnostic |
| 36 GB VRAM target device unspecified (32 GB 5090? 2×16? unified memory?) | Keep VRAM budget a config parameter; decide when hardware is known — changes constants, not architecture |

## 6. Sources

- GLM-5.2: [model card](https://huggingface.co/zai-org/GLM-5.2) · [config.json](https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json) · [Unsloth run-locally guide](https://unsloth.ai/docs/models/glm-5.2) · [NVIDIA NeMo GLM-5 MoE+DSA](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/glm-5-moe-dsa) · [GLM-5 overview](https://glm-5.org/) · [MindStudio GLM-5.2 overview](https://www.mindstudio.ai/blog/what-is-glm-5-2-open-weight-model-1m-context) · [codersera local-run guide](https://codersera.com/blog/how-to-run-glm-5-2-locally-2026/)
- Prediction/prefetch: [ProMoE](https://arxiv.org/abs/2410.22134) · [Fate / cross-layer gate](https://arxiv.org/html/2502.12224v1) · [MoE-SpeQ](https://arxiv.org/abs/2511.14102) · [MoE-Beyond](https://arxiv.org/html/2508.17137v1) · [DAOP](https://arxiv.org/pdf/2501.10375)
- Caching/offload systems: [MoE-Infinity](https://ar5iv.labs.arxiv.org/html/2401.14361) · [HOBBIT](https://arxiv.org/html/2411.01433v1) · [FlashMoE](https://arxiv.org/pdf/2601.17063) · [fMoE](https://arxiv.org/pdf/2502.05370) · [ReMoE](https://arxiv.org/html/2605.27081) · [MoE-ERAS](https://openreview.net/pdf?id=o43eHjPEMO) · [awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference/)
- Systems/baselines: [llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) · [MoE tuning gist](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0) · [GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/) · [GDS 2026 guide](https://www.spheron.network/blog/gpu-direct-storage-nvme-ai-training-inference-guide/) · [PIPO pipelined offloading](https://arxiv.org/pdf/2504.03664)
