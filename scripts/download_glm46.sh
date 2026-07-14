#!/usr/bin/env bash
# GLM-4.6 UD-IQ2_XXS (~115GB, glm4moe, top-8): a mid-size target between Air
# (47GB) and GLM-5.2 (247GB). At a 30 GiB budget it is ~31% RAM-resident, so it
# genuinely exercises the SSD paging path (decode 2.73 tok/s on a Gen5 drive; see
# docs/traces-analysis.md §33). Same architecture as Air — the arena runs it with
# no code changes.
#
# Pick a different quant with QUANT=UD-IQ1_M (~107GB) etc; see the repo's HF page.
# DEST defaults to models/glm46-gguf (symlink this to a big/fast disk if needed).
set -euo pipefail
cd "$(dirname "$0")/.."
QUANT=${QUANT:-UD-IQ2_XXS}
DEST=${DEST:-models/glm46-gguf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
mkdir -p "$DEST"
.venv/bin/hf download unsloth/GLM-4.6-GGUF --include "*${QUANT}*" \
    --local-dir "$DEST" 2>&1 | tail -3
echo "GLM46_DOWNLOAD_DONE"
du -sh "$DEST"
echo "first shard: $(ls "$DEST"/*/*00001-of-*.gguf 2>/dev/null | head -1)"
