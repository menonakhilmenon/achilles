#!/usr/bin/env bash
# Eviction policy A/B on GLM-5.2 decode, FULL profile: lru vs reuse.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/glm-policy-ab.txt
: > "$OUT"
for pol in lru reuse; do
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$AVAIL" -lt 48 ]; then echo "PREFLIGHT ABORT: ${AVAIL}GB" | tee -a "$OUT"; exit 1; fi
  UNIT=achilles-glm-pol
  systemctl --user reset-failed $UNIT 2>/dev/null
  LOG=$T/glm-pol-$pol.log
  systemd-run --user --unit $UNIT -p WorkingDirectory="$PWD" \
    -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps.' \
      -n 192 -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --workers 6 --pstream 1 --policy $pol --stats > $LOG 2>&1"
  while systemctl --user is-active --quiet $UNIT.service; do sleep 5; done
  echo "--- policy=$pol ---" >> "$OUT"
  grep -aE "prefill|decode|arena:|DeviceLost|GGML_ASSERT" "$LOG" >> "$OUT"
  sleep 15
done
echo "AB_DONE" >> "$OUT"
