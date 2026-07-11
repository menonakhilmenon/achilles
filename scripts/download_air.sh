#!/usr/bin/env bash
# GLM-4.5-Air UD-Q2_K_XL (47.4GB): best routing fidelity that stays ~90%
# page-cache-resident under the 44GB RAM cap (mmap pages are reclaimable).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/hf download unsloth/GLM-4.5-Air-GGUF --include "*UD-Q2_K_XL.gguf" \
    --local-dir models/glm45-air-gguf 2>&1 | tail -2
echo "AIR_DOWNLOAD_DONE"
du -sh models/glm45-air-gguf
