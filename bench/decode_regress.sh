#!/usr/bin/env bash
# Decode regression bisect (FULL): dense-weight placement. CPU-only needs the
# ~25GB dense skeleton in RAM -> smaller expert budget per config.
#   ngl0  : all dense in RAM        -> budget 16
#   ngl40 : ~10GB VRAM, rest RAM    -> budget 22
#   ngl99 : max offload (spills)    -> budget 30 (tonight's 0.64 baseline)
#   repro : yesterday's exact staged config (-n 24, pstream 0, lru, ngl 99)
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/decode-regress.txt
: > "$OUT"

run_cfg() {
  name=$1; shift
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$AVAIL" -lt 48 ]; then echo "PREFLIGHT ABORT before $name: ${AVAIL}GB" | tee -a "$OUT"; exit 1; fi
  UNIT=achilles-glm-regress
  systemctl --user reset-failed $UNIT 2>/dev/null
  LOG=$T/glm-regress-$name.log
  systemd-run --user --unit $UNIT -p WorkingDirectory="$PWD" \
    -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G -p IOWeight=50 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps.' \
      $* --workers 6 --stats > $LOG 2>&1"
  VMAX=0; GMAX=0
  while systemctl --user is-active --quiet $UNIT.service; do
    for c in /sys/class/drm/card*/device; do
      vt=$(cat $c/mem_info_vram_total 2>/dev/null || echo 0)
      if [ "$vt" -gt 8000000000 ]; then
        v=$(cat $c/mem_info_vram_used 2>/dev/null || echo 0)
        g=$(cat $c/mem_info_gtt_used 2>/dev/null || echo 0)
        [ "$v" -gt "$VMAX" ] && VMAX=$v
        [ "$g" -gt "$GMAX" ] && GMAX=$g
      fi
    done
    sleep 2
  done
  echo "--- $name ---" >> "$OUT"
  if systemctl --user is-failed --quiet $UNIT.service; then
    echo "RUN FAILED: $(systemctl --user show $UNIT.service -p Result --value)" >> "$OUT"
  fi
  grep -aE "prefill|decode:|arena:|DeviceLost" "$LOG" >> "$OUT"
  echo "dgpu_vram_peak=$((VMAX/1048576))M dgpu_gtt_peak=$((GMAX/1048576))M" >> "$OUT"
  sleep 15
}

run_cfg ngl0  -n 48 -t 10 -ngl 0            --budget-gib 16 --pstream 1 --policy lru
run_cfg ngl40 -n 48 -t 10 -ngl 40 -ot exps=CPU --budget-gib 22 --pstream 1 --policy lru
run_cfg ngl99 -n 48 -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --pstream 1 --policy lru
run_cfg repro -n 24 -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --pstream 0 --policy lru
echo "REGRESS_DONE" >> "$OUT"
