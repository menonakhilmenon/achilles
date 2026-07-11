# Achilles

Run a frontier-scale MoE model (target: **GLM-5.2**, 744B params) on a device with
~36 GB VRAM by treating VRAM as a small cache over a three-tier memory hierarchy
(VRAM ← host RAM ← NVMe SSD) for the routed experts.

The premise: 97.4% of GLM-5.2's weights live in 19,200 tiny (~38M param) routed experts,
of which each token only uses 600 activations (75 MoE layers × top-8). If we (1) prefetch
experts just-in-time and evict them after their layer runs, and (2) train a lightweight
predictor that knows *which* experts a token will need several layers ahead, the
loads can be overlapped and hidden instead of stalling the critical path.

- **[docs/research.md](docs/research.md)** — feasibility math, GLM-5.2 architecture,
  bandwidth budget, prior art survey, risks.
- **[docs/plan.md](docs/plan.md)** — phased plan of action with go/no-go gates.

Approach: **CPU-first** — validate the paging + prediction theory with all compute on
CPU (two tiers: RAM ← SSD), POC on a ~150 GB-class model (GLM-4.6 / Qwen3-235B), then
add the GPU as the fast tier and scale to GLM-5.2.

Status: research/planning (2026-07-11). Next: Phase 0 hardware microbenchmarks and
Phase 1 router tracing on small MoE models.
