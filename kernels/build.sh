#!/usr/bin/env bash
# kernels/build.sh: the kernel libraries, one a kind (libnextcuda-image.so ...): each the shared part (kernels/ns: GPUs,
# memory, copies, the C ABI's core) and that kind's engines (kernels/<kind>/<arch>), built with nvcc (the image
# library also holds the bridge to stable-diffusion.cpp, kernels/image/sd); and the silos,
# a model's kernels tuned for it alone (kernels/<kind>/<arch>/silo/*.cu -> dist/silo/libnextcuda-<arch>.so), which its
# engine opens at load. Runs where nvcc is (the GPU host; ../build.sh copies the sources there).
#   ARCHS="86 120" (default: the cards of the box: sm_86 3060, sm_120 5070; + PTX of the newest)   JOBS=4
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${CUDA_HOME:-/usr/local/cuda}
NVCC=${NVCC:-$CUDA/bin/nvcc}
ARCHS=${ARCHS:-"86 120"}
JOBS=${JOBS:-4}
OUT=${OUT:-../dist}
OBJ=${OBJ:-../target/kernels}
mkdir -p "$OUT/silo" "$OBJ"
SDCPP=${SDCPP:-$HOME/stable-diffusion.cpp}
SDLIB=${SDLIB:-$HOME/nextcuda/src/sdcpp-build}

GEN=()
last=
for a in $ARCHS; do GEN+=(-gencode "arch=compute_$a,code=sm_$a"); last=$a; done
GEN+=(-gencode "arch=compute_$last,code=compute_$last")      # PTX of the newest: a newer card JITs it
FLAGS=(-std=c++17 -O3 --use_fast_math -Xcompiler -fPIC -lineinfo "${GEN[@]}" -Ins -Ins -Idiffusion -Illm/nemotronh -Illm/nemotronh/third_party -Illm/nemotronh/third_party/ggml -I"$SDCPP/include")
# --use_fast_math is not on for the numerics that must match a reference: a source names its own flags below
precise() { case "$1" in diffusion/*|llm/*|audio/*) return 0 ;; esac; return 1; }

KINDS="llm image video audio"
# stable-diffusion.cpp (ggml-cuda), the CUDA implementation the image engines are built on: its source tree (headers)
# and a shared build of it (-DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON), copied beside the libraries
sources() {
  ls ns/*.cu diffusion/*.cu 2>/dev/null || true
  for k in $KINDS; do ls $k/*/*.cu 2>/dev/null || true; done
  [ -f "$SDCPP/include/stable-diffusion.h" ] && ls image/sd/*.cpp 2>/dev/null || true
}
silo_sources() { for k in $KINDS; do ls $k/*/silo/*.cu 2>/dev/null || true; done; }

compile() {
  f=$1; o="$OBJ/$(echo "$f" | tr / _).o"
  fl=()
  for a in "${FLAGS[@]}"; do
    precise "$f" && [ "$a" = "--use_fast_math" ] && continue
    fl+=("$a")
  done
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ ns/ns_internal.hpp -nt "$o" ]; then
    "$NVCC" "${fl[@]}" -c "$f" -o "$o" && echo "ok   $f" || { echo "FAIL $f"; return 1; }
  else echo "cached $f"; fi
}
export -f compile precise
export NVCC OBJ
export FLAGS_STR="${FLAGS[*]}"
fail=0
for f in $(sources) $(silo_sources); do compile "$f" || fail=1; done
[ "$fail" = 0 ] || { echo "==> compile failed"; exit 1; }

for k in $KINDS; do
  objs="$(ls "$OBJ"/ns_*.o 2>/dev/null) $(ls "$OBJ"/${k}_*.o 2>/dev/null | grep -v '_silo_' || true)"
  [ -n "$(ls "$OBJ"/${k}_*.o 2>/dev/null | grep -v '_silo_' || true)" ] || continue
  # the shared diffusion kernels (nsd) are in every kind's library but llm's
  [ "$k" = llm ] || objs="$objs $(ls "$OBJ"/diffusion_*.o 2>/dev/null || true)"
  extra=()
  sdso=$(ls "$SDLIB"/bin/libstable-diffusion.so "$SDLIB"/libstable-diffusion.so 2>/dev/null | head -1 || true)
  if [ "$k" = image ] && [ -n "$sdso" ]; then
    # beside the old one, then renamed over it: a running server keeps the library it mapped (writing over it in place
    # changes the code under it - it cost a running bench its process)
    cp -L "$sdso" "$OUT/.libstable-diffusion.so.new" && mv -f "$OUT/.libstable-diffusion.so.new" "$OUT/libstable-diffusion.so"
    extra=(-L"$OUT" -lstable-diffusion -Xlinker -rpath -Xlinker '$ORIGIN')
  fi
  echo "==> linking libnextcuda-$k.so"
  "$NVCC" -shared "${GEN[@]}" $objs "${extra[@]}" -lcublas -lcublasLt -o "$OUT/.libnextcuda-$k.so.new"
  mv -f "$OUT/.libnextcuda-$k.so.new" "$OUT/libnextcuda-$k.so"
done
for d in $(ls -d */*/silo 2>/dev/null); do
  arch=$(basename "$(dirname "$d")")
  objs=$(ls "$OBJ"/$(echo "$d" | tr / _)_*.o 2>/dev/null || true)
  [ -n "$objs" ] || continue
  echo "==> linking silo/libnextcuda-$arch.so"
  "$NVCC" -shared "${GEN[@]}" $objs -o "$OUT/silo/.libnextcuda-$arch.so.new"
  mv -f "$OUT/silo/.libnextcuda-$arch.so.new" "$OUT/silo/libnextcuda-$arch.so"
done
ls -la "$OUT"/libnextcuda*.so "$OUT"/silo/*.so 2>/dev/null || true
