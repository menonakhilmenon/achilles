#!/usr/bin/env bash
# Post-janitor bias A/B, detached-unit friendly. Results to bench/results/.
cd /var/home/akhil/achilles
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
P="Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps."
OUT=bench/results/ab-bias-janitor.txt
: > "$OUT"
for cfg in 0 0; do
  for f in models/glm52-gguf/UD-Q2_K_XL/*.gguf; do python3 bench/evict_file.py "$f" >/dev/null; done
  sleep 75
  echo "--- bias=$cfg ---" >> "$OUT"
  GGML_VK_VISIBLE_DEVICES=1 src/achilles-arena -m "$M" -p "$P" -n 48 -t 10 -ngl 99 \
    -ot "exps=CPU" --budget-gib 34 --delta 3 --fetch 8 --workers 6 --bias $cfg --stats 2>&1 \
    | grep -aE "decode|arena: prefetch" >> "$OUT"
done
echo "AB_DONE" >> "$OUT"
