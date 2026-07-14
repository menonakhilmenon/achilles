# Optional MTP self-drafting patches for llama.cpp

The arena's `--spec-mtp` flag (Multi-Token-Prediction self-drafting: the model
drafts its own next tokens from NextN/MTP layers, then verifies them in one batch)
needs two small commits that are **not** in upstream llama.cpp. They live here as
patches so you can opt in without a fork.

Everything else — decode, prefill, prefetch, eviction, the whole caching result —
works on **stock** llama.cpp at the pinned tag. You only need these if you want to
experiment with `--spec-mtp`.

## What's here

| file | commit | what it does |
|---|---|---|
| `0001-glm-dsa-load-NextN-tensors-and-add-DeepSeek-style-MT.patch` | 961af24 | load NextN tensors + DeepSeek-style MTP draft graph for `glm-dsa` |
| `0002-glm-dsa-nextn-KV-layer-filter-for-MTP-contexts-was-a.patch` | 8143319 | NextN KV-layer filter (was over-allocating full-model KV) |
| `glm-dsa-mtp.combined.diff` | both, squashed | single-file alternative for `git apply` |
| `BASE_COMMIT.txt` | — | the exact base commit (`e3546c7`, = upstream tag **b9976**) |

Touches 4 files (`src/llama-model.cpp`, `src/models/deepseek2.cpp`,
`src/models/glm-dsa.cpp`, `src/models/models.h`), ~198 lines. Architecture:
`glm-dsa` (GLM-5.2). They do **not** affect the `glm4moe` models (Air, GLM-4.6).

## Apply

From your llama.cpp checkout, at the pinned base tag (see `scripts/setup_llama.sh`,
which checks out `b9976`):

```
cd llama.cpp
git checkout b9976                       # the base these patches target
# preferred — preserves the two commits with authorship:
git am /path/to/achilles/patches/llama.cpp-mtp/00*.patch
# or, if you just want the diff applied without commits:
git apply /path/to/achilles/patches/llama.cpp-mtp/glm-dsa-mtp.combined.diff

# then rebuild as usual
cmake --build build-vk -j"$(nproc)"
cd .. && ./build.sh
```

Or let the setup script do it: `LLAMA_MTP=1 scripts/setup_llama.sh`.

If `git am` fails on a different base, fall back to the combined diff with
`git apply --3way`, or cherry-pick against a newer tag and resolve by hand — the
changes are localized to the GLM model files.

## Reality check

MTP on GLM is **experimental and a mixed result** — see `docs/traces-analysis.md`
§24 ("built, works, and loses to its own byte bill"): the extra NextN weights and
verify batch cost I/O that, in the SSD-paged regime, often eats the speculative
win. It's here for experimentation, not because it's a guaranteed speedup. The
glm4moe (Air/GLM-4.6) MTP draft path is still unresolved (produces unaccepted
drafts). Set expectations accordingly.
