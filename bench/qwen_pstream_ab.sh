#!/usr/bin/env bash
# Prefill layer-streaming thrash reproduction on Qwen3-30B (task #19).
# POLITE profile: MemoryHigh=20G, 6 threads, 4 workers, nice 10, IOWeight 10.
# Long prompt (~2300 tok) -> 2 prefill chunks; pstream 0 vs 1, 2 runs each,
# model page cache evicted between runs; 1s cgroup memory sampler per run.
set -u
cd /var/home/akhil/achilles
. bench/vkdev.sh
M=models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/qwen-pstream-ab.txt
: > "$OUT"
head -c 9000 docs/research.md > "$T/longprompt.txt"

for ps in 0 1 0 1; do
  python3 bench/evict_file.py "$M" >/dev/null
  sleep 15
  UNIT=achilles-qwen-ab
  systemctl --user reset-failed $UNIT 2>/dev/null
  LOG=$T/qwen-ab-ps$ps-$RANDOM.log
  systemd-run --user --unit $UNIT -p WorkingDirectory=/var/home/akhil/achilles \
    -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=1G -p Nice=10 -p IOWeight=10 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' -f '$T/longprompt.txt' \
      -n 8 -t 6 --budget-gib 5 --workers 4 --no-uring --pstream $ps --stats > $LOG 2>&1"
  CG=$(systemctl --user show $UNIT.service -p ControlGroup --value)
  echo "--- pstream=$ps ---" >> "$OUT"
  SAMP="$T/samp-ps$ps.txt"; : > "$SAMP"
  while systemctl --user is-active --quiet $UNIT.service; do
    cur=$(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null || echo 0)
    echo "$(date +%s.%N) $((cur/1048576))M" >> "$SAMP"
    sleep 1
  done
  grep -aE "prefill|decode|arena:|io:" "$LOG" >> "$OUT"
  # summarize the sampler: min/max/mean + biggest 1s swing
  awk '{v=$2+0; if(NR==1||v<mn)mn=v; if(v>mx)mx=v; s+=v; if(NR>1){d=v-p; if(d<0)d=-d; if(d>sw)sw=d} p=v}
       END{if(NR)printf "mem: min=%dM max=%dM mean=%dM max_1s_swing=%dM samples=%d\n", mn, mx, s/NR, sw, NR}' "$SAMP" >> "$OUT"
done
echo "AB_DONE" >> "$OUT"
