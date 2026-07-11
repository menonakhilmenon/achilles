#!/usr/bin/env bash
# Trace GLM-4.5-Air (UD-Q2_K_XL, 47.4GB) expert routing.
# Model exceeds the 44GB cap -> mmap page-cache paging under pressure; this run
# doubles as the first natural-paging observation (log tok/s per prompt).
# 3 prompts/domain x 512 tokens is enough for family-level locality stats.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf
OUT=traces/glm45-air
N_NEW=${N_NEW:-512}
LIMIT=${LIMIT:-3}
mkdir -p "$OUT"

.venv/bin/python - <<EOF > /tmp/air_prompts.tsv
import json, pathlib
prompts = json.loads(pathlib.Path("scripts/prompts.json").read_text())
for domain, ps in prompts.items():
    for i, p in enumerate(ps[:$LIMIT]):
        p = p.replace("\t", " ").replace("\n", " ")
        print(f"{domain}\t{i:02d}\t{p}")
EOF

while IFS=$'\t' read -r domain pid prompt; do
    out="$OUT/${domain}-${pid}.bin"
    if [ -s "$out" ]; then echo "skip $domain-$pid"; continue; fi
    full="[gMASK]<sop><|user|>\n${prompt}<|assistant|>\n"
    start=$(date +%s)
    TRACE_OUT="$out" systemd-run --user --scope -p MemoryMax=44G --quiet \
      scripts/trace-moe -m "$MODEL" -p "$full" -n "$N_NEW" -t 14 -c 4096 \
      2>>"$OUT/log.txt" || echo "FAIL $domain-$pid" >> "$OUT/log.txt"
    echo "done $domain-$pid in $(($(date +%s)-start))s ($(stat -c%s "$out" 2>/dev/null || echo 0) bytes)"
done < /tmp/air_prompts.tsv

.venv/bin/python scripts/parse_moe_trace.py "$OUT"/*.bin
echo "AIR_TRACE_DONE"
