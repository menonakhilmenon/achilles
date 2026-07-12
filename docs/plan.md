# Plan of Action

Companion to [research.md](research.md) — read that first for the feasibility math and
prior art. Strategy (revised 2026-07-11 after discussion):

1. **All compute on CPU first.** Two tiers only (RAM ← SSD). This deletes the hardest
   systems work while keeping both core ideas — prefetch/evict around the MoE phase and
   the learned expert predictor — fully testable. Some speed loss is acceptable; if the
   theory holds on CPU, we add the GPU tier.
2. **POC on a ~150 GB-class model** (GLM-4.6 @ Q3 or Qwen3-235B-A22B @ Q4) — chosen to be
   *larger than the box's RAM* so the SSD tier is genuinely exercised — then the 744B beast.

Guiding numbers (research.md §2): experts are ~97% of MoE model weights; per decoded
token, top-8 × N-layers expert activations. Throughput ceiling on CPU = RAM bandwidth ÷
active-bytes-per-token (~3–4 tok/s for GLM-5.2 Q4 on desktop DDR5); paging efficiency
determines how close we get to that ceiling. Success = ceiling-limited, not I/O-limited.

## Target box (measured 2026-07-11)

| Component | Spec | Implication |
|---|---|---|
| CPU | AMD Ryzen 7 9800X3D — 8C/16T Zen 5, **full AVX-512** (incl. `avx512_bf16`, `avx512_vnni`, VAES/GFNI), 96 MB L3 (X3D) | Excellent CPU-inference ISA; llama.cpp Zen 5 kernels use it. 8 cores is modest for prefill; decode is bandwidth-bound anyway |
| RAM | **64 GB** DDR5, dual-channel (AM5), speed TBD → STREAM in Phase 0. Expect ~55–70 GB/s sustained | The binding resource — see RAM budget below |
| SSD | Kingston Fury Renegade 2 TB, **PCIe 4.0 ×4 confirmed** (16 GT/s ×4) | ~7.0–7.3 GB/s seq read spec; the paging tier. 2 TB fits every staged model |
| Also | 4 TB SATA HDD (archive only); 16 GB zram swap | Never page experts from HDD; runtime must not spill into zram |
| GPU | Radeon RX **9070 XT — 16 GB, AMD RDNA4** (+ iGPU) | **Not the 36 GB target device.** Phase 5 GPU path on this box = ROCm/Vulkan, no CUDA, no GPUDirect Storage (NVIDIA-only). CPU-first is doubly right |

### RAM budget (OOM-safety policy)

Hard rule: **the runtime gets ≤ 44 GB; ≥ 16 GB stays free for OS/desktop** (the box idles
at ~10 GB used and runs a desktop). Enforced, not hoped for:

- Run the engine under **cgroup v2** `memory.max=46G`, `memory.high=44G` (kills/throttles
  us, never the session).
- Expert cache = **self-managed arena** (~38–40 GB) + ~4 GB model/KV/buffers. Reads via
  **O_DIRECT** into the arena so the kernel page cache can't balloon uncontrolled; if we
  use the mmap embodiment instead, active `madvise(DONTNEED)` eviction keeps residency
  at the cap.
- Watch PSI (`/proc/pressure/memory`) during runs; any zram swap activity = budget bug.

### Phase 0 measured numbers (2026-07-12, bench/results/)

| Quantity | Measured | Notes |
|---|---|---|
| RAM read bw | **57 GB/s** best, 55.5 sustained (16T) | copy 41, triad 46 GB/s; STREAM-style, 8 GiB working set |
| RAM bw while SSD busy | 55.7 GB/s (−3%) | I/O DMA steals almost nothing; earlier 43 GB/s dip was CPU contention (compile), not I/O |
| NVMe rand read, 20 MiB blocks | **7.5 GB/s burst / ~3.9 GB/s sustained** | thermal throttle after ~30–60 s of saturated reads (controller sensor 77 °C); recovers in ~45 s idle. QD4 suffices; 11 ms avg latency @QD4 |
| NVMe rand read, 1 MiB blocks | 4.5 GB/s @QD8, 1.9 ms | small blocks lose ~40% — favor ≥4–20 MiB reads (GLM Q4 expert = 20 MB: perfect) |
| NVMe seq read | same envelope as rand @20 MiB | large-block random ≈ sequential on this drive |
| NVMe write | 3.35 GB/s over 128 GiB | one-time model writes only |
| Post-write GC | reads degrade ~20 min after heavy writes | write models, then let the drive settle before benchmarking/paging |

**Design consequences**: (1) decode paging at ≥80% hit rate needs 1–3 GB/s — below the
throttle envelope, so 7.5 GB/s burst is the operative number for prefetch deadlines,
while sustained-throttled 4 GB/s bounds prefill streaming and cold-start;
(2) a heatsink on the NVMe is the cheapest upgrade in the project (~2× sustained);
(3) expert reads should be ≥4 MiB — batch small experts into extents.

### Ceilings on this box (measured: 55 GB/s RAM; SSD 7.5 burst / 4 sustained)

| Model | Bytes/token | RAM-bw ceiling | Resident in 40 GB cache | Hit rate needed to stay ceiling-bound* |
|---|---|---|---|---|
| GLM-4.5-Air Q4 (~60 GB) | ~6.6 GB | ~8.3 tok/s | ~2/3 of model | ~86% (74%) |
| Qwen3-235B Q4 (~133 GB) | ~12 GB | ~4.6 tok/s | ~30% | ~85% (72%) |
| GLM-4.6 Q3 (~150–170 GB) | ~13 GB | ~4.2 tok/s | ~27% | ~85% (72%) |
| GLM-5.2 Q2 (~245 GB) | ~13 GB | ~4.2 tok/s | ~16% | ~85% (72%) |

\* hit rate at which per-token SSD traffic ÷ SSD bandwidth ≤ per-token RAM read time,
i.e. the SSD fully hides behind compute — first number at 4 GB/s sustained-throttled,
parenthesized at 7.5 GB/s burst (short generations / heatsinked drive). The gap between
"resident %" and "needed hit rate" is what skew + temporal locality + the predictor
must supply — and the OLMoE traces (docs/traces-analysis.md) show exactly that gap
being closed: 80% of experts recur within an 8-token window, and untrained gate-ahead
prediction recalls 85–96% at practical lookaheads.

## Phase 0 — Hardware truth & harness (days)

Box specs are recorded above; what remains is measuring the two bandwidths.

- [ ] STREAM-style RAM bandwidth benchmark (all cores) → replaces the 60 GB/s assumption.
- [ ] NVMe→RAM: io_uring + O_DIRECT sustained reads at expert granularity (~10–20 MB),
      queue-depth sweep, random vs sequential; concurrent with a memory-bound compute
      loop (does I/O steal bandwidth from compute?).
- [ ] Rerun the ceiling table above with measured numbers.
- [ ] Repo scaffolding: bench/ directory, results logging.

**Gate**: projected POC tok/s (ceiling × simulated hit-rate factor) ≥ ~1 tok/s for the
150 GB model. If not, fix hardware first (RAM upgrade / second NVMe / lower quant).

## Phase 1 — Tracing & predictability study (1–2 weeks)

The science, on models that iterate fast: **Qwen3-30B-A3B** first, then **GLM-4.5-Air**
(106B/A12B, GLM lineage). CPU inference via HF transformers/llama.cpp is fine — traces
don't care about speed.

- [ ] Hook routers (PyTorch forward hooks) → dump per-token, per-layer top-k expert IDs,
      router logits, and (subsampled) hidden states.
- [ ] Trace corpus: code / chat / math / long-form × varied decode lengths.
- [ ] Analysis notebooks (hypotheses in research.md §4):
      - expert popularity skew per layer/domain → static hot-set sizing curve
      - token-to-token reuse → cache hit-rate simulation vs cache size (LRU/LFU/LCP)
      - cross-layer predictability: recall@8 predicting layer l+Δ routing from layer l
        hidden state, Δ=1…8 — (a) Fate-style gate-ahead (zero training), (b) logistic
        probe, (c) small MLP
      - router margin distribution (near-ties = cheap substitutions, hard predictions)
- [ ] **Cache/paging simulator**: replay traces against configurable (RAM size, policy,
      predictor recall/lookahead, SSD latency/bandwidth) model → predicted tok/s.
      This plot is the go/no-go for the whole premise — before building anything.

**Gate**: simulation says the POC model on this box, with achievable predictor recall,
runs within ~2× of its RAM-bandwidth ceiling.

## Phase 2 — CPU paging runtime v1 (2–4 weeks)

Make idea #1 real: experts live on SSD, RAM is a managed cache, loads hidden behind
compute. Two candidate embodiments — decide after a 2–3 day spike on each:

- **(a) llama.cpp fork** — it already mmaps weights and has CPU MoE kernels. Add an
  expert-cache subsystem: intercept expert access, active prefetch thread (io_uring
  pread into a managed arena, or fadvise/madvise page-cache steering as the crudest
  cut), eviction policy, instrumentation of stall time per layer. Fastest path to a
  real model; constraint: working inside ggml's graph execution.
- **(b) Standalone mini-runtime** — own MoE loop in C++/Rust (or PyTorch+custom ops)
  for one model family. Total control, more work, easier research iteration later
  (predictor integration, scheduling experiments).

Milestones (either embodiment):

- [ ] Baseline A: everything-in-RAM (small model) — the ceiling.
- [ ] Baseline B: stock mmap with RAM < model (cgroup memory limit to force it) — the
      floor ("existing tricks").
- [ ] Managed cache + reactive (on-miss) loading; correctness vs baseline A (bit-exact
      logits).
- [ ] Prefetcher v1 (heuristics): pin per-layer hot set + previous-token experts.
- [ ] Prefetcher v2: gate-ahead (Fate-style, Δ=1–2) feeding async prefetch queue with
      per-layer deadlines.
- [ ] Miss fallbacks (flag-gated): stall (exact) vs compute-with-stale/substitute expert
      (measure quality delta).
- [ ] Metrics: tok/s, per-layer stall ms, hit rate, I/O utilization — vs baselines A & B.

Test model: GLM-4.5-Air with a cgroup RAM cap (emulates the POC ratio cheaply).

**Gate**: ≥2× over baseline B and within ~2× of baseline A at POC-like RAM:model ratio;
simulator predictions validated against reality (then we can trust it for sizing the
beast).

**STATUS (2026-07-12): Phase 2 executed ahead of schedule and beyond scope** — the
runtime was built as a llama.cpp wrapper (v1 page-cache steering → v2 owned arena
with MAP_FIXED tensor replacement + io_uring O_DIRECT; see traces-analysis.md
§12–14 and src/). Gates: Air beats stock mmap 2.7× at the POC ratio ✓; the
simulator's structure validated but its hit-rate extrapolation to 256-expert
layers was optimistic (measured 45–47% at 17–19% residency vs assumed ~85%) —
recalibrated in §14. GLM-5.2 itself (skipping ahead of the 150 GB POC model)
runs at 0.70 tok/s = 2.3× naive; the ~150 GB Phase 4 target class is expected to
land near Air-like ratios (3+ tok/s). Remaining v2 backlog: trained probe
integration, VRAM hot-expert tier, prefill layer-streaming.

## Phase 3 — Learned predictor (2–3 weeks, overlaps Phase 2)

- [ ] Train per-layer (or shared, layer-embedded) MLP: hidden state at layer l →
      multi-label top-8 prediction at l+Δ; distill from router logits (soft targets).
- [ ] Recall-vs-Δ sweep → pick the cascade: gate-ahead (Δ=1) for near-deadline
      promotion, learned long-horizon (Δ=4–8, possibly cross-token) for SSD reads.
- [ ] Overprovisioning knob: prefetch top-m (m>8); recall vs wasted-bandwidth curve.
- [ ] Integrate; predictor inference cost must be negligible (it's a ~hidden→256 head).
- [ ] Measure end-to-end gain over gate-ahead-only and over no-predictor.

**Gate**: SSD-tier stalls become rare (<5% of layer executions) at the lookahead the
measured SSD latency requires.

## Phase 4 — POC: the ~150 GB model (2–3 weeks)

- [ ] Model selection (finalize with Phase 1 data): **GLM-4.6 @ Q3** (~145–170 GB, family
      continuity with target) vs **Qwen3-235B-A22B @ Q4** (~133 GB, smaller experts,
      cleaner quant). Requirement: comfortably > box RAM.
- [ ] Retrain predictor on POC-model traces (collect with the paging runtime itself,
      slowly).
- [ ] Prefill mode: layer-sequential expert streaming (sequential SSD reads at full
      bandwidth, all prompt tokens per expert batch), chunked.
- [ ] End-to-end eval: decode tok/s (p50/p99 inter-token), TTFT, quality (perplexity +
      spot benchmarks) vs same quant fully-in-RAM on a big machine (or vs API);
      ablations: no predictor / no prefetch / stock mmap.

**Gate / the thesis test**: POC model decodes within ~2× of its RAM-bandwidth ceiling
on the box, ≥3–5× faster than stock mmap paging, quality unchanged. **If this holds,
the theory is validated and GPU + beast are next; if not, the writeup of why is still
a real result.**

## Phase 5 — GPU tier + the beast (4–8 weeks)

Only after Phase 4 passes. Two additions, in order:

- [ ] **GPU as accelerator tier** (ktransformers-style split, then beyond): dense
      skeleton (attention/MLA+DSA, shared experts, embeddings — ~10 GB @Q4) + KV cache +
      hottest experts in VRAM; CPU computes cold-but-in-RAM experts (Fiddler-style
      — moving 24 KB of activations beats moving 20 MB of weights); SSD feeds RAM under
      predictor control. GPU also fixes prefill (compute-bound).
      **De-risked early (2026-07-12)**: llama.cpp Vulkan on the RX 9070 XT already runs
      this split — GLM-4.5-Air decode 2.3× (5.5→12.6 tok/s), prefill 2.8×; see
      traces-analysis.md §9. Revised GLM-5.2-Q2 projection with hybrid: ~3.5–4 tok/s.
- [ ] **GLM-5.2** @ Q2–Q4 experts (dense skeleton kept high-precision — it's only ~19B
      params): needs MLA + DSA (IndexShare) kernels — inventory engine support
      (llama.cpp/KTransformers/SGLang) at that time and build on the best one.
- [ ] Full eval + ablations as in Phase 4.

**Success criterion** (refine after Phase 0): GLM-5.2 decoding at **≥2–5 tok/s** on
36 GB VRAM + desktop RAM + NVMe, quality ≈ quant baseline — vs ~0.1–0.5 tok/s naive
paging.

## Phase 6 — Stretch / research contributions

- Speculative decoding co-design: draft model predicts future tokens → batch expert
  prefetch across tokens (MoE-SpeQ-style).
- HOBBIT-style mixed precision: low-bit copies of hot-set experts always resident;
  high-bit fetched only when lead time allows — converts stalls into small quality noise.
- Cache-aware router biasing near ties (ReMoE-style) — locality/quality tradeoff.
- Disk layout optimization: co-locate co-activated experts.
- Write-up: blog post at minimum; plausibly a systems paper (research.md §3 positioning:
  nobody has shown a 744B model at usable speed on this class of hardware).

## Immediate next steps

1. Phase 0 microbenchmarks (RAM bandwidth, SSD throughput at expert granularity).
2. Phase 1 tracing on Qwen3-30B-A3B (~18 GB Q4 — fits the RAM budget with room to spare).

## Resolved / open questions

- ~~CPU or GPU compute first?~~ → **CPU-first** (owner, 2026-07-11); GPU returns in
  Phase 5 after the theory is validated.
- ~~Speed bar?~~ → some speed reduction is acceptable (owner, 2026-07-11); quantified
  gates above still apply so "slow" stays "usable".
- ~~Exact hardware?~~ → measured 2026-07-11, see "Target box" table. 64 GB RAM →
  runtime capped at 44 GB (OOM-safety policy above). Note: on-box GPU is a 16 GB AMD
  RX 9070 XT, **not** the 36 GB device — Phase 5 on this box means ROCm/Vulkan, and the
  dense-skeleton split must fit ~16 GB (it does at Q4: ~10 GB + KV).
- POC model: GLM-4.6 @ Q3 (~150–170 GB) vs Qwen3-235B @ Q4 (~133 GB) — both ≫ 64 GB RAM,
  so both genuinely exercise the SSD tier; decide with Phase 1 trace data.
- Quality floor for the beast: are Q2 experts (~82% quality) acceptable, or is Q4 the bar?
  (On this box GLM-5.2 is Q2-only anyway until the SSD tier proves itself — Q4 is 380 GB
  of a 2 TB drive, fine on disk, but per-token traffic doubles.)
- What is the eventual 36 GB VRAM device? (Doesn't block Phases 0–4.)
