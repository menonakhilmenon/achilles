#!/usr/bin/env bash
# GLM-5.2 long-prompt prefill verification, POLITE profile (owner present):
# MemoryHigh=32G, budget 20GiB, 6 threads, 4 workers, nice 10, IOWeight 10.
# Verifies the ggml_nrows callback fix + pstream at GLM scale. Single run.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/glm-prefill-polite.txt
LOG=$T/glm-prefill.log
UNIT=achilles-glm-polite
: > "$OUT"
head -c 9000 docs/research.md > "$T/longprompt.txt"

AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$AVAIL" -lt 40 ]; then
  echo "PREFLIGHT ABORT: only ${AVAIL}GB available (need 40+ for the 32G polite envelope)" | tee -a "$OUT"
  exit 1
fi
systemctl --user reset-failed $UNIT 2>/dev/null
systemd-run --user --unit $UNIT -p WorkingDirectory="$PWD" \
  -p MemoryHigh=32G -p MemoryMax=36G -p MemorySwapMax=2G -p Nice=10 -p IOWeight=10 \
  bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' -f '$T/longprompt.txt' \
    -n 8 -t 6 -ngl 99 -ot exps=CPU --budget-gib 20 --workers 4 --pstream 1 --stats > $LOG 2>&1"
CG=$(systemctl --user show $UNIT.service -p ControlGroup --value)
SAMP=$T/glm-prefill-mem.txt; : > "$SAMP"
while systemctl --user is-active --quiet $UNIT.service; do
  cur=$(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null || echo 0)
  echo "$(date +%s) $((cur/1048576))M" >> "$SAMP"
  sleep 2
done
grep -aE "arena:|prefill|decode|io:|GGML_ASSERT|OUTPUT" "$LOG" | head -12 >> "$OUT"
awk '{v=$2+0; if(NR==1||v<mn)mn=v; if(v>mx)mx=v; s+=v; if(NR>1){d=v-p; if(d<0)d=-d; if(d>sw)sw=d} p=v}
     END{if(NR)printf "mem: min=%dM max=%dM mean=%dM max_2s_swing=%dM samples=%d\n", mn, mx, s/NR, sw, NR}' "$SAMP" >> "$OUT"
echo "GLM_DONE" >> "$OUT"
