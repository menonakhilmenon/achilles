#!/usr/bin/env bash
# Migrate models to the Gen5 drive, regenerate shadow, re-run headline.
set -u
cd "$(dirname "$0")/.."
D=/run/media/akhil/Hyperspace
S=bench/results/gen5-status.txt
: > "$S"
log() { echo "$(date +%H:%M:%S) $*" >> "$S"; }

log "copy glm shards start (247G, ~10 min)"
mkdir -p $D/glm52-gguf/UD-Q2_K_XL $D/models
cp models/glm52-gguf/UD-Q2_K_XL/*.gguf $D/glm52-gguf/UD-Q2_K_XL/ || { log COPY_FAILED; exit 1; }
log "copy glm done"
cp models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf $D/models/ 2>/dev/null
cp models/glm45-air-gguf/GLM-4.5-Air-UD-Q2_K_XL.gguf $D/models/ 2>/dev/null
log "small models copied"

log "shadow regen on gen5 start (~10 min)"
.venv/bin/python scripts/repack_shadow.py $D/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf $D/glm52-shadow > /home/akhil/.claude/jobs/34438cab/tmp/repack-gen5.log 2>&1 || { log SHADOW_FAILED; exit 1; }
log "shadow regen done"

# symlink swap (originals untouched on the old drive)
mv models/glm52-gguf models/glm52-gguf.gen4 2>/dev/null
ln -sfn $D/glm52-gguf models/glm52-gguf
ln -sf $D/glm52-shadow.bin models/glm52-shadow.bin
ln -sf $D/glm52-shadow.idx models/glm52-shadow.idx
log "symlinks swapped"

bash bench/phase4_headline.sh
log "GEN5_DONE"
