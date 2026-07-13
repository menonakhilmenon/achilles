#!/usr/bin/env bash
# Phase 4: GLM shadow validation + combined headline re-measure.
set -u
cd /var/home/akhil/achilles
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
T=/home/akhil/.claude/jobs/34438cab/tmp
S=bench/results/phase4-status.txt
OUT=bench/results/glm-headline.txt
: > "$S"; : > "$OUT"
log() { echo "$(date +%H:%M:%S) $*" >> "$S"; }
tuned-adm profile throughput-performance-bazzite 2>/dev/null
printf 'Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps.' > $T/shortprompt.txt
head -c 9000 docs/research.md > $T/longprompt.txt

run_glm() {  # name promptfile [flags...] — flags must be space-free tokens
  name=$1; pf=$2; shift 2
  AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$AVAIL" -lt 48 ] && { log "ABORT $name: ${AVAIL}G"; exit 1; }
  systemctl --user reset-failed achilles-glm-h 2>/dev/null
  systemd-run --user --wait --unit achilles-glm-h -p WorkingDirectory=/var/home/akhil/achilles \
    -p MemoryHigh=44G -p MemoryMax=48G -p MemorySwapMax=2G \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' -f '$pf' \
      -t 10 -ngl 99 -ot exps=CPU --budget-gib 30 --workers 6 --policy reuse $* --stats > $T/glm-h-$name.log 2>&1" 2>/dev/null
  echo "--- $name ---" >> "$OUT"
  systemctl --user is-failed --quiet achilles-glm-h.service && echo "FAILED: $(systemctl --user show achilles-glm-h.service -p Result --value)" >> "$OUT"
  grep -aE "prefill|decode:|detail|accept|error" $T/glm-h-$name.log >> "$OUT"
  sleep 15
}

log "identity gate"
run_glm ident-plain  $T/shortprompt.txt -n 24 --pstream 0
run_glm ident-shadow $T/shortprompt.txt -n 24 --pstream 0 --shadow models/glm52-shadow
A=$(grep -a OUTPUT $T/glm-h-ident-plain.log)
B=$(grep -a OUTPUT $T/glm-h-ident-shadow.log)
if [ -z "$A" ] || [ -z "$B" ] || [ "$A" != "$B" ]; then
  log "GLM SHADOW GATE FAILED (empty or mismatched output)"
  exit 1
fi
log "GLM TOKEN_IDENTICAL - headline runs"

for i in 1 2 3; do
  run_glm decode$i $T/shortprompt.txt -n 96 --shadow models/glm52-shadow
done
run_glm spec4 $T/shortprompt.txt -n 96 --spec 4 --shadow models/glm52-shadow
run_glm prefill-ps1 $T/longprompt.txt -n 8 --pstream 1 --shadow models/glm52-shadow
run_glm prefill-ps0 $T/longprompt.txt -n 8 --pstream 0 --shadow models/glm52-shadow
log "PHASE4_DONE"
