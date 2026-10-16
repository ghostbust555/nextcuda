#!/usr/bin/env bash
# nextcuda's build: the release binary, built in the platform's debian:trixie toolchain image so it links against the
# glibc 2.41 the GPU host (Debian 13) has. `./build.sh` -> dist/nextcuda plus what it needs beside it.
#   TOOLCHAIN_IMAGE   the image (default localhost/nextcuda-toolchain: Rust on debian:trixie; made from
#                     tooling/Containerfile when it is missing)
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${TOOLCHAIN_IMAGE:-localhost/nextcuda-toolchain}"
TOOL="${CONTAINER_TOOL:-$(command -v podman || command -v docker)}"
mkdir -p dist
"$TOOL" image exists "$IMAGE" 2>/dev/null || "$TOOL" build -q -t "$IMAGE" -f tooling/Containerfile tooling >/dev/null
"$TOOL" run --rm -v "$PWD":/build:Z -w /build -e CARGO_TARGET_DIR=/build/target-trixie "$IMAGE" \
    cargo build --release -p nextcuda
install -m 0755 target-trixie/release/nextcuda dist/nextcuda
# the web front ends (node is the build's only use of it)
./wfe/build.sh >/dev/null
# what the binary finds beside itself: the python worker, the model cards, the docs it serves
rm -rf dist/python dist/model-cards
mkdir -p dist/python
cp image/qwenimage21/python/*.py dist/python/
cp -r model-cards dist/model-cards
echo ">> dist/nextcuda ready"
