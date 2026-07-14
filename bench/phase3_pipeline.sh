#!/usr/bin/env bash
# Detached Phase-3 pipeline: qwen repack -> validate -> GLM repack.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
T=/home/akhil/.claude/jobs/34438cab/tmp
S=bench/results/phase3-status.txt
: > "$S"
log() { echo "$(date +%H:%M:%S) $*" >> "$S"; }

log "qwen repack start"
.venv/bin/python scripts/repack_shadow.py models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf models/qwen3-shadow > $T/repack-qwen.log 2>&1 || { log "QWEN REPACK FAILED"; exit 1; }
log "qwen repack done"

M=models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
for mode in plain shadow; do
  [ "$mode" = shadow ] && EX="--shadow models/qwen3-shadow" || EX=""
  python3 bench/evict_file.py "$M" >/dev/null; python3 bench/evict_file.py models/qwen3-shadow.bin >/dev/null 2>&1
  sleep 5
  systemctl --user reset-failed achilles-qwen 2>/dev/null
  systemd-run --user --wait --unit achilles-qwen -p WorkingDirectory="$PWD" \
    -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=1G -p Nice=10 -p IOWeight=10 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'The three main causes of the French Revolution were' \
      -n 64 -t 6 --budget-gib 5 --workers 4 --policy reuse $EX --stats > $T/qwen-shadow3-$mode.log 2>&1" 2>/dev/null
done
if diff <(grep -a OUTPUT $T/qwen-shadow3-plain.log) <(grep -a OUTPUT $T/qwen-shadow3-shadow.log) >/dev/null; then
  log "QWEN TOKEN_IDENTICAL: $(grep -a 'decode:' $T/qwen-shadow3-plain.log) || shadow: $(grep -a 'decode:' $T/qwen-shadow3-shadow.log)"
else
  log "QWEN OUTPUT MISMATCH - aborting before GLM repack"
  exit 1
fi

log "glm repack start (~2-3h)"
.venv/bin/python scripts/repack_shadow.py models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf models/glm52-shadow > $T/repack-glm.log 2>&1 || { log "GLM REPACK FAILED"; exit 1; }
log "glm repack done"
log "PHASE3_PIPELINE_DONE"
