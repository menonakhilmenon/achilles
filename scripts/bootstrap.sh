#!/usr/bin/env bash
# One-shot local setup for the achilles stack. Idempotent — safe to re-run.
#   1. build llama.cpp (Vulkan) at the pinned tag         [scripts/setup_llama.sh]
#   2. compile src/achilles-arena against it              [./build.sh]
#   3. create the Python venv for model downloads
#   4. (optional) fetch a first model                     [MODEL=air|glm46|none]
#
# Usage:
#   scripts/bootstrap.sh                # steps 1-3, then print next steps
#   MODEL=air   scripts/bootstrap.sh    # also download GLM-4.5-Air (47 GB)
#   MODEL=glm46 scripts/bootstrap.sh    # also download GLM-4.6   (115 GB)
#
# Nothing here needs root. Skips work that's already done.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-none}

echo "== [1/4] llama.cpp (Vulkan) =="
if [ -e llama.cpp/build-vk/bin/libllama.so ]; then
  echo "   already built -> $(cd llama.cpp && git rev-parse --short HEAD 2>/dev/null)"
else
  scripts/setup_llama.sh
fi

echo "== [2/4] achilles-arena =="
if [ -x src/achilles-arena ] && [ src/achilles-arena -nt src/arena.cpp ]; then
  echo "   up to date"
else
  ./build.sh
fi

echo "== [3/4] python venv =="
if [ -x .venv/bin/hf ]; then
  echo "   .venv present"
else
  python3 -m venv .venv
  .venv/bin/pip install -q huggingface_hub hf_transfer
  echo "   created .venv with huggingface_hub + hf_transfer"
fi

echo "== [4/4] model =="
case "$MODEL" in
  air)   scripts/download_air.sh ;;
  glm46) scripts/download_glm46.sh ;;
  none)  echo "   skipped (set MODEL=air or MODEL=glm46 to fetch one)" ;;
  *)     echo "   unknown MODEL=$MODEL (use air|glm46|none)"; exit 1 ;;
esac

cat <<'EOF'

== done ==
Next: pick your GPU by name and run (see AGENTS.md for the memory-safe wrapper):
  VK_GPU_NAME="<your GPU>" source bench/vkdev.sh
  GGML_VK_VISIBLE_DEVICES=$VKDEV src/achilles-arena -m <model.gguf> \
    -p "hello" -n 64 -t "$(nproc)" -ngl 99 -ot exps=CPU \
    --budget-gib 16 --workers 6 --policy reuse --stats
EOF
