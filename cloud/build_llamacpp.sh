#!/usr/bin/env bash
# Build llama.cpp with CUDA on the instance. Idempotent.
#
# RQ2's faithful setup needs a program model larger than the monitor, and on one 24 GiB
# L4 that only works if the program is quantized and its routed experts live in system
# RAM. bitsandbytes cannot do it -- it replaces nn.Linear, and a MoE checkpoint's fused
# expert tensors are not nn.Linear -- so the program is served by llama.cpp instead,
# which `LlamaCppBackend` talks to over an OpenAI-compatible endpoint.
set -euo pipefail
cd "$HOME"

if [ -x "$HOME/llama.cpp/build/bin/llama-server" ]; then
    echo "llama-server already built"
    "$HOME/llama.cpp/build/bin/llama-server" --version 2>&1 | head -2 || true
    exit 0
fi

# nvcc is present on the DLVM image but not on PATH, and cmake's CUDA language probe
# fails without it. Name the compiler explicitly rather than relying on the shell.
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
test -x "$CUDACXX" || { echo "no nvcc at $CUDACXX" >&2; exit 1; }
echo "-- nvcc: $($CUDACXX --version | tail -1)"

echo "-- build deps"
sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential libcurl4-openssl-dev git >/dev/null

echo "-- source"
[ -d llama.cpp/.git ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

echo "-- configure (CUDA)"
# The L4 is sm_89. Naming it directly avoids compiling every architecture, which is
# most of the build time here.
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=ON \
      -DCMAKE_CUDA_COMPILER="$CUDACXX" \
      -DCMAKE_BUILD_TYPE=Release >/dev/null

echo "-- compile (this is the slow part)"
cmake --build build --config Release -j"$(nproc)" --target llama-server >/dev/null

test -x build/bin/llama-server
echo "-- built"
build/bin/llama-server --version 2>&1 | head -2 || true
