#!/usr/bin/env bash
# Phase 2 v1 gate: achilles-pager (managed cache+prefetch) vs kernel mmap paging
# on GLM-4.5-Air at the residencies where naive collapsed (air-tiers2.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf
P="Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps."
OUT=bench/results/pager-vs-kernel.txt
: > "$OUT"

kernel() { # cap
    python3 bench/evict_file.py "$M" > /dev/null; sleep 2
    echo "--- kernel MemoryHigh=$1 ---" | tee -a "$OUT"
    GGML_VK_VISIBLE_DEVICES=1 systemd-run --user --scope -p MemoryHigh=$1 --quiet \
      src/achilles-pager -m "$M" -p "$P" -n 64 -t 10 -ngl 99 -ot "exps=CPU" \
      --no-pager 2>&1 | grep -E "ACHILLES|OUTPUT" | tee -a "$OUT"
}

pager() { # budget_gib cap
    python3 bench/evict_file.py "$M" > /dev/null; sleep 2
    echo "--- pager budget=${1}GiB cap=$2 ---" | tee -a "$OUT"
    GGML_VK_VISIBLE_DEVICES=1 systemd-run --user --scope -p MemoryHigh=$2 --quiet \
      src/achilles-pager -m "$M" -p "$P" -n 64 -t 10 -ngl 99 -ot "exps=CPU" \
      --budget-gib "$1" --delta 2 --fetch 9 --stats 2>&1 \
      | grep -E "ACHILLES|OUTPUT|region" | tee -a "$OUT"
}

kernel 24G     # ~54% of model resident -> naive was 1.67 tok/s
pager  15 24G  # headroom below cap so kernel reclaim never engages
kernel 16G     # ~36% resident -> naive was 0.85 tok/s
pager  8 16G
echo "PAGER_BENCH_DONE"
