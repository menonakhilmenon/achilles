#!/usr/bin/env bash
# Three-tier sweep, take 2: evict model from page cache before each run so the
# benchmark scope faults (and is charged for) its own pages -> MemoryHigh bites.
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf
OUT=bench/results/air-tiers2.txt
: > "$OUT"

read_sectors() { awk '$3=="nvme0n1" {print $6}' /proc/diskstats; }

for CAP in 48G 32G 24G 16G; do
    python3 bench/evict_file.py "$M" > /dev/null
    sleep 3
    RES0=$(fincore -b "$M" 2>/dev/null | awk 'NR==2{printf "%.1f", $1/2**30}' || echo "?")
    echo "=== MemoryHigh=$CAP (resident before: ${RES0} GiB) ===" | tee -a "$OUT"
    S0=$(read_sectors); T0=$(date +%s)
    GGML_VK_VISIBLE_DEVICES=1 systemd-run --user --scope \
        -p MemoryHigh=$CAP --quiet \
        llama.cpp/build-vk/bin/llama-bench -m "$M" -t 10 -ngl 99 \
        -ot "exps=CPU" -p 0 -n 128 -r 2 2>/dev/null | grep tg128 | tee -a "$OUT"
    S1=$(read_sectors); T1=$(date +%s)
    RES1=$(fincore -b "$M" 2>/dev/null | awk 'NR==2{printf "%.1f", $1/2**30}' || echo "?")
    echo "nvme read: $(( (S1 - S0) / 2048 )) MiB in $((T1 - T0))s; resident after: ${RES1} GiB" | tee -a "$OUT"
done
echo "AIR_TIERS2_DONE"
