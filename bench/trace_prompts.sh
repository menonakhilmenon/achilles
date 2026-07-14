#!/usr/bin/env bash
# Trace decode expert usage for a diverse prompt set on Qwen3-30B (POLITE).
# One arena run per prompt, --dump to traces/qwen-prompts/pNN.bin.
# Warm page cache is fine — traces record routing, not IO.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
mkdir -p traces/qwen-prompts
n=0
while IFS= read -r prompt; do
  n=$((n+1))
  id=$(printf "p%02d" $n)
  [ -s "traces/qwen-prompts/$id.bin" ] && { echo "$id exists, skip"; continue; }
  UNIT=achilles-qwen-trace
  systemctl --user reset-failed $UNIT 2>/dev/null
  systemd-run --user --wait --unit $UNIT -p WorkingDirectory="$PWD" \
    -p MemoryHigh=22G -p MemoryMax=26G -p MemorySwapMax=1G -p Nice=10 -p IOWeight=10 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p '$(printf "%s" "$prompt" | sed "s/'/'\\\\''/g")' \
      -n 192 -t 6 --budget-gib 14 --workers 4 --policy lru \
      --dump traces/qwen-prompts/$id.bin --stats > $T/trace-$id.log 2>&1" 2>/dev/null
  echo "$id: $(grep -ac . traces/qwen-prompts/$id.bin 2>/dev/null || echo '?') $(grep -aE 'decode:' $T/trace-$id.log | tail -1)"
done < bench/prompts20.txt
echo "TRACES_DONE"
