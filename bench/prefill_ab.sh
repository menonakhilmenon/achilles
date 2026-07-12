#!/usr/bin/env bash
# Prefill layer-streaming A/B on GLM-5.2 with a long prompt (~600 tokens).
cd /var/home/akhil/achilles
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=bench/results/prefill-ab.txt
: > "$OUT"
head -c 9000 docs/research.md > /tmp/claude-1000/longprompt.txt
for ps in 0 1 0 1; do
  for f in models/glm52-gguf/UD-Q2_K_XL/*.gguf; do python3 bench/evict_file.py "$f" >/dev/null; done
  sleep 75
  echo "--- pstream=$ps ---" >> "$OUT"
  GGML_VK_VISIBLE_DEVICES=1 src/achilles-arena -m "$M" -f /tmp/claude-1000/longprompt.txt \
    -n 8 -t 10 -ngl 99 -ot "exps=CPU" --budget-gib 30 --delta 3 --fetch 8 --workers 6 \
    --pstream $ps --stats 2>&1 | grep -aE "prefill|decode" >> "$OUT"
done
echo "AB_DONE" >> "$OUT"
