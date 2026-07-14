#!/usr/bin/env bash
# Detached Air MTP tuning pipeline (owner away). Base + depth x pmin grid.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
tuned-adm profile throughput-performance-bazzite 2>/dev/null
M=models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf
T=/home/akhil/.claude/jobs/34438cab/tmp
OUT=bench/results/air-mtp-tune.txt
: > "$OUT"
run() {
  name=$1; shift
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$AVAIL" -lt 50 ] && { echo "ABORT $name ${AVAIL}G" >> "$OUT"; return 1; }
  systemctl --user reset-failed achilles-air 2>/dev/null
  systemd-run --user --wait --unit achilles-air -p WorkingDirectory="$PWD" \
    -p MemoryHigh=46G -p MemoryMax=48G -p MemorySwapMax=0 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -f $T/shortprompt.txt -n 128 -t 10 -ngl 99 -ot exps=CPU --budget-gib 38 --workers 6 --policy reuse $* --stats > $T/airt-$name.log 2>&1" 2>/dev/null
  echo "--- $name ---" >> "$OUT"
  systemctl --user is-failed --quiet achilles-air.service && echo "FAILED: $(systemctl --user show achilles-air.service -p Result --value)" >> "$OUT"
  grep -aE "spec:|decode:|OUTPUT" $T/airt-$name.log >> "$OUT"
  sleep 10
}
run base
run d3-p06 --spec-mtp 3 --spec-pmin 0.6
run d3-p04 --spec-mtp 3 --spec-pmin 0.4
run d2-p06 --spec-mtp 2 --spec-pmin 0.6
run d4-p06 --spec-mtp 4 --spec-pmin 0.6
run d3-p08 --spec-mtp 3 --spec-pmin 0.8
run base2
if diff <(grep -a OUTPUT $T/airt-base.log) <(grep -a OUTPUT $T/airt-d3-p06.log) >/dev/null 2>&1; then
  echo "TOKEN_IDENTICAL base vs d3-p06" >> "$OUT"
else
  echo "OUTPUT DIFFERS base vs d3-p06" >> "$OUT"
fi
tuned-adm profile balanced-bazzite 2>/dev/null
echo TUNE_DONE >> "$OUT"
