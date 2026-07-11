#!/usr/bin/env bash
# Trace Qwen3-30B-A3B expert routing across the prompt corpus via trace-moe.
# Runs under a systemd memory cap; Q4 model (~18GB) + ctx fits well under 30G.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=$(ls models/qwen3-30b-a3b-gguf/*Q4_K_M*.gguf | head -1)
OUT=traces/qwen3-30b
N_NEW=${N_NEW:-640}
mkdir -p "$OUT"

.venv/bin/python - <<'EOF' > /tmp/qwen_prompts.tsv
import json, pathlib
prompts = json.loads(pathlib.Path("scripts/prompts.json").read_text())
for domain, ps in prompts.items():
    for i, p in enumerate(ps):
        p = p.replace("\t", " ").replace("\n", " ")
        print(f"{domain}\t{i:02d}\t{p}")
EOF

while IFS=$'\t' read -r domain pid prompt; do
    out="$OUT/${domain}-${pid}.bin"
    if [ -s "$out" ]; then echo "skip $domain-$pid"; continue; fi
    # Qwen3 chat format, thinking disabled by template default in Instruct-2507
    full="<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n"
    TRACE_OUT="$out" systemd-run --user --scope -p MemoryMax=30G --quiet \
      scripts/trace-moe -m "$MODEL" -p "$full" -n "$N_NEW" -t 14 -c 4096 \
      2>>"$OUT/log.txt" || echo "FAIL $domain-$pid" >> "$OUT/log.txt"
    echo "done $domain-$pid ($(stat -c%s "$out" 2>/dev/null || echo 0) bytes)"
done < /tmp/qwen_prompts.tsv

.venv/bin/python scripts/parse_moe_trace.py "$OUT"/*.bin
echo "QWEN_TRACE_DONE"
