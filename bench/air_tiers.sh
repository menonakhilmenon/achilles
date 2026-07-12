#!/usr/bin/env bash
# Three-tier test on GLM-4.5-Air UD-Q2_K_XL (44.18 GiB):
#   GPU  : dense skeleton (-ngl 99, experts overridden to CPU)
#   RAM  : expert cache = page cache, constrained by scope MemoryHigh
#   SSD  : overflow experts paged via mmap
# Sweeps MemoryHigh and records decode tok/s + real NVMe read traffic.
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf
OUT=bench/results/air-tiers.txt
: > "$OUT"

read_sectors() { awk '$3=="nvme0n1" {print $6}' /proc/diskstats; }

for CAP in 48G 32G 24G 16G; do
    echo "=== MemoryHigh=$CAP ===" | tee -a "$OUT"
    S0=$(read_sectors)
    T0=$(date +%s)
    GGML_VK_VISIBLE_DEVICES=1 systemd-run --user --scope \
        -p MemoryHigh=$CAP --quiet \
        llama.cpp/build-vk/bin/llama-bench -m "$M" -t 10 -ngl 99 \
        -ot "exps=CPU" -p 0 -n 128 -r 2 2>/dev/null | grep tg128 | tee -a "$OUT"
    S1=$(read_sectors)
    T1=$(date +%s)
    echo "nvme read: $(( (S1 - S0) / 2048 )) MiB in $((T1 - T0))s (incl model load)" | tee -a "$OUT"
done
echo "AIR_TIERS_DONE"
