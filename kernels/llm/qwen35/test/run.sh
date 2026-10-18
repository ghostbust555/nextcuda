#!/usr/bin/env bash
# kernels/llm/qwen35/test/run.sh HOST [GPU...]: the qwen35 kernels' selftest on the GPU host - this tree's kernels built
# apart from the live ones (~/nextcuda/q35test: its own objects, seeded from the deploy's, and libraries), then
# q35_test against that libnextcuda-llm.so on each GPU given (default 0 and 1).
set -euo pipefail
cd "$(dirname "$0")/../../.."
HOST=${1:?usage: run.sh HOST [GPU...]}; shift || true
GPUS=${*:-0 1}
quiet() { grep -v 'alacritty\|^\]11\|snapshot' || true; }
ssh -o BatchMode=yes "$HOST" "mkdir -p nextcuda/q35test/kernels" 2>&1 | quiet
rsync -a --delete -e "ssh -o BatchMode=yes" ./ "$HOST":nextcuda/q35test/kernels/ 2>&1 | quiet
ssh -o BatchMode=yes "$HOST" "set -e; export PATH=/usr/local/cuda/bin:\$PATH; T=\$HOME/nextcuda/q35test
  [ -d \$T/obj ] || { mkdir -p \$T/obj; cp -a \$HOME/nextcuda/src/obj/. \$T/obj/ 2>/dev/null || true; }
  cd \$T/kernels && OUT=\$T/dist OBJ=\$T/obj ./build.sh 2>&1 | grep -v '^cached' | tail -12
  nvcc -std=c++17 -O2 -Ins -Illm/nemotronh -Illm/qwen35 llm/qwen35/test/q35_test.cu -L\$T/dist -lnextcuda-llm -Xlinker -rpath -Xlinker \$T/dist -o \$T/q35_test
  for g in $GPUS; do \$T/q35_test \$g; done" 2>&1 | quiet
