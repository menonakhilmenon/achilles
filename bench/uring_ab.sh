#!/usr/bin/env bash
# io_uring O_DIRECT vs buffered (--no-uring) on the post-§21 best config.
set -u
cd /var/home/akhil/achilles
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/uring-ab.txt
: > "$OUT"
for io in nouring uring nouring uring; do
  [ "$io" = nouring ] && FLAG="--no-uring" || FLAG=""
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$AVAIL" -lt 48 ] && { echo "ABORT: ${AVAIL}G" | tee -a "$OUT"; exit 1; }
  UNIT=achilles-glm-uring
  systemctl --user reset-failed $UNIT 2>/dev/null
  systemd-run --user --unit $UNIT -p WorkingDirectory=/var/home/akhil/achilles \
    -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps.' \
      -n 48 -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --workers 6 $FLAG --pstream 1 --policy reuse --stats > $T/uring-$io-$RANDOM.log 2>&1"
  CG=$(systemctl --user show $UNIT.service -p ControlGroup --value)
  PK=0
  while systemctl --user is-active --quiet $UNIT.service; do
    c=$(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null || echo 0); [ "$c" -gt "$PK" ] && PK=$c
    sleep 2
  done
  echo "--- $io ---" >> "$OUT"
  systemctl --user is-failed --quiet $UNIT.service && echo "FAILED: $(systemctl --user show $UNIT.service -p Result --value)" >> "$OUT"
  grep -aE "decode|detail|io:" $(ls -t $T/uring-$io-*.log | head -1) >> "$OUT"
  echo "cgroup_peak=$((PK/1073741824))G" >> "$OUT"
  sleep 15
done
echo AB_DONE >> "$OUT"
