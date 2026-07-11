#!/usr/bin/env bash
# Phase 1 environment: venv + models + llama.cpp (CPU build).
set -euxo pipefail
cd "$(dirname "$0")/.."

# 1. Python env (3.12: guaranteed torch wheel coverage; system 3.14 is too new to bet on)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python \
    transformers accelerate "huggingface_hub[cli,hf_transfer]" \
    numpy pandas matplotlib safetensors sentencepiece

mkdir -p models traces bench/results

# 2. Models (background-parallel downloads)
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/hf download allenai/OLMoE-1B-7B-0125-Instruct \
    --local-dir models/olmoe-1b-7b-instruct &
OLMOE_PID=$!
.venv/bin/hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
    --include "*Q4_K_M*" --local-dir models/qwen3-30b-a3b-gguf &
QWEN_PID=$!

# 3. llama.cpp CPU build (while downloads run)
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build llama.cpp/build --config Release -j16 --target llama-cli llama-server

wait $OLMOE_PID $QWEN_PID
echo "SETUP_PHASE1_DONE"
df -h /var/home | tail -1
