#!/usr/bin/env bash
# Speculation x plan A/B (run only when disk is quiet): spec 0/4/8 on Qwen,
# then GLM at FULL if the Qwen signal is positive.
set -u
cd /var/home/akhil/achilles
. bench/vkdev.sh
M=models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
T=${CLAUDE_TMP:-/home/akhil/.claude/jobs/34438cab/tmp}
OUT=bench/results/spec-ab.txt
: > "$OUT"
for spec in 0 4 8 0 4 8; do
  python3 bench/evict_file.py "$M" >/dev/null; python3 bench/evict_file.py models/qwen3-shadow.bin >/dev/null 2>&1
  sleep 8
  systemctl --user reset-failed achilles-qwen-spec 2>/dev/null
  systemd-run --user --wait --unit achilles-qwen-spec -p WorkingDirectory=/var/home/akhil/achilles \
    -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=1G -p Nice=10 -p IOWeight=10 \
    bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
      -p 'Write a detailed technical explanation of how flash attention reduces memory bandwidth requirements in transformer inference, covering tiling, softmax rescaling, and the IO complexity analysis.' \
      -n 256 -t 6 --budget-gib 5 --workers 4 --policy reuse --shadow models/qwen3-shadow --spec $spec --stats > $T/spec-$spec-$RANDOM.log 2>&1" 2>/dev/null
  echo "--- spec=$spec ---" >> "$OUT"
  grep -aE "decode:|detail|accept" $(ls -t $T/spec-$spec-*.log | head -1) >> "$OUT"
done
echo AB_DONE >> "$OUT"
