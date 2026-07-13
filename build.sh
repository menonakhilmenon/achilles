#!/usr/bin/env bash
# Build the achilles-arena runtime (src/arena.cpp) against a Vulkan-enabled
# llama.cpp build. Run scripts/setup_llama.sh first if you don't have one.
#
#   LLAMA_CPP=/path/to/llama.cpp   (default: ./llama.cpp)
#   LLAMA_BUILD=/path/to/build     (default: $LLAMA_CPP/build-vk)
set -euo pipefail
cd "$(dirname "$0")"

LLAMA=${LLAMA_CPP:-llama.cpp}
BUILD=${LLAMA_BUILD:-$LLAMA/build-vk}
LIB="$BUILD/bin"

# --- locate llama.cpp source + built shared libs ---------------------------
[ -f "$LLAMA/include/llama.h" ] || {
  echo "ERROR: llama.cpp source not found at '$LLAMA'."
  echo "       Run scripts/setup_llama.sh, or set LLAMA_CPP=/path/to/llama.cpp"; exit 1; }
[ -e "$LIB/libllama.so" ] || {
  echo "ERROR: llama.cpp not built (no $LIB/libllama.so)."
  echo "       Run scripts/setup_llama.sh, or:"
  echo "       cmake -B '$BUILD' -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON '$LLAMA' && cmake --build '$BUILD' -j"; exit 1; }

# --- locate liburing headers (system, or a prefix like homebrew) -----------
URING_INC=""; URING_L=""; URING_RPATH=""
if [ -e /usr/include/liburing.h ]; then
  :  # system default; nothing to add
elif pkg-config --exists liburing 2>/dev/null; then
  URING_INC=$(pkg-config --cflags liburing)
  URING_L=$(pkg-config --libs-only-L liburing)
else
  for p in /usr/local "${HOMEBREW_PREFIX:-/home/linuxbrew/.linuxbrew}"; do
    if [ -e "$p/include/liburing.h" ]; then
      URING_INC="-I$p/include"; URING_L="-L$p/lib"; URING_RPATH="-Wl,-rpath,$p/lib"; break
    fi
  done
fi
[ -e /usr/include/liburing.h ] || [ -n "$URING_INC" ] || {
  echo "ERROR: liburing.h not found. Install it:"
  echo "       Fedora:  sudo dnf install liburing-devel"
  echo "       Debian:  sudo apt install liburing-dev"; exit 1; }

echo ">> building src/achilles-arena"
g++ -O3 -march=native -std=c++17 -o src/achilles-arena src/arena.cpp \
  -I"$LLAMA/include" -I"$LLAMA/common" -I"$LLAMA/ggml/include" $URING_INC \
  -L"$LIB" $URING_L \
  -lllama -lllama-common -lggml -lggml-base -luring \
  -Wl,-rpath,"$LIB" $URING_RPATH

echo "built: src/achilles-arena"
echo "run:   src/achilles-arena --help-ish (see README Quickstart)"
