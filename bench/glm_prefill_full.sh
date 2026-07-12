#!/usr/bin/env bash
# GLM-5.2 long-prompt prefill A/B at FULL profile (owner-authorized):
# MemoryHigh=44G, budget 30GiB, 10 threads, 6 workers. pstream 1 then 0.
set -u
cd /var/home/akhil/achilles
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/glm-prefill-full.txt
UNIT=achilles-glm-full
: > "$OUT"
head -c 9000 docs/research.md > "$T/longprompt.txt"

for ps in 1 0; do
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$AVAIL" -lt 48 ]; then
    echo "PREFLIGHT ABORT before pstream=$ps: only ${AVAIL}GB available (need 48+)" | tee -a "$OUT"
    exit 1
  fi
  LOG=$T/glm-full-ps$ps.log
  systemctl --user reset-failed $UNIT 2>/dev/null
  systemd-run --user --unit $UNIT -p WorkingDirectory=/var/home/akhil/achilles \
    -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' -f '$T/longprompt.txt' \
      -n 8 -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --workers 6 --pstream $ps --stats > $LOG 2>&1"
  CG=$(systemctl --user show $UNIT.service -p ControlGroup --value)
  echo "--- pstream=$ps ---" >> "$OUT"
  SAMP=$T/glm-full-mem-ps$ps.txt; : > "$SAMP"
  GTT=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | tail -1)
  while systemctl --user is-active --quiet $UNIT.service; do
    cur=$(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null || echo 0)
    g=$(cat "$GTT" 2>/dev/null || echo 0)
    echo "$(date +%s) $((cur/1048576))M gtt=$((g/1048576))M" >> "$SAMP"
    sleep 2
  done
  grep -aE "arena:|prefill|decode|io:|GGML_ASSERT|OUTPUT" "$LOG" | head -12 >> "$OUT"
  awk '{v=$2+0; if(NR==1||v<mn)mn=v; if(v>mx)mx=v; s+=v; if(NR>1){d=v-p; if(d<0)d=-d; if(d>sw)sw=d} p=v
        sub(/^gtt=/,"",$3); g=$3+0; if(g>gx)gx=g}
       END{if(NR)printf "mem: min=%dM max=%dM mean=%dM max_2s_swing=%dM gtt_peak=%dM samples=%d\n", mn, mx, s/NR, sw, gx, NR}' "$SAMP" >> "$OUT"
  sleep 20
done
echo "GLM_DONE" >> "$OUT"
