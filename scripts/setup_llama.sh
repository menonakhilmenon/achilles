#!/usr/bin/env bash
# Clone and build llama.cpp with Vulkan, pinned to the upstream tag the arena
# was built against. The arena is a plain llama.cpp wrapper (it only uses the
# public API + common), so any nearby upstream build works; this tag is what
# the numbers in the README were measured on.
#
# NOTE: MTP self-drafting (--spec-mtp) needs two extra commits that live only in
# a local fork and are NOT on this tag. Everything else — decode, prefill,
# prefetch, eviction — works on stock llama.cpp at this tag.
#
# Override the tag with LLAMA_TAG=bXXXX, or point at an existing tree with
# LLAMA_CPP=/path/to/llama.cpp (then just run the cmake lines below yourself).
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${LLAMA_TAG:-b9976}
PATCHDIR="$(pwd)/patches/llama.cpp-mtp"   # resolve before we cd into llama.cpp

if [ ! -d llama.cpp/.git ]; then
  echo ">> cloning llama.cpp"
  git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
echo ">> checking out $TAG"
git fetch --tags --force origin 2>/dev/null || true
git checkout "$TAG"

# Optional: apply the MTP self-drafting patches (only needed for --spec-mtp).
# See patches/llama.cpp-mtp/README.md. Off by default; everything else is stock.
if [ "${LLAMA_MTP:-0}" = "1" ]; then
  echo ">> applying MTP patches (LLAMA_MTP=1)"
  git am "$PATCHDIR"/00*.patch || {
    echo "!! git am failed; aborting the am and leaving llama.cpp at $TAG"
    git am --abort 2>/dev/null || true
    echo "!! apply manually per patches/llama.cpp-mtp/README.md"; exit 1; }
fi

echo ">> configuring (Vulkan, shared libs)"
cmake -B build-vk \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF

echo ">> building (this takes a few minutes)"
cmake --build build-vk -j"$(nproc)"

echo
echo "llama.cpp built at $(git rev-parse --short HEAD) -> $(pwd)/build-vk/bin"
echo "next: ./build.sh"
