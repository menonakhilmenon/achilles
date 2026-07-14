#!/usr/bin/env bash
# Eviction policy A/B on Qwen3-30B decode (POLITE profile): lru vs reuse
# (gap/pop reuse-distance evictor; sim on GLM streams: +2pp hit rate).
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/qwen-policy-ab.txt
: > "$OUT"

for pol in lru reuse; do
  python3 bench/evict_file.py "$M" >/dev/null
  sleep 10
  UNIT=achilles-qwen-pol
  systemctl --user reset-failed $UNIT 2>/dev/null
  LOG=$T/qwen-pol-$pol-$RANDOM.log
  systemd-run --user --unit $UNIT -p WorkingDirectory="$PWD" \
    -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=1G -p Nice=10 -p IOWeight=10 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'Write a detailed technical explanation of how flash attention reduces memory bandwidth requirements in transformer inference, covering tiling, softmax rescaling, and the IO complexity analysis.' \
      -n 256 -t 6 --budget-gib 5 --workers 4 --policy $pol --stats > $LOG 2>&1"
  while systemctl --user is-active --quiet $UNIT.service; do sleep 2; done
  echo "--- policy=$pol ---" >> "$OUT"
  grep -aE "prefill|decode|arena:" "$LOG" >> "$OUT"
done
echo "AB_DONE" >> "$OUT"
