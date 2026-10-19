#!/usr/bin/env bash
# tooling/deploy.sh HOST: build here, build the CUDA kernels where nvcc is, put it all on the GPU host and restart
# the control plane as its systemd user service (nextcuda.service, :8000). Safe to run again: nothing is switched
# until the new files are in place, and the kernels build only what changed.
#   NEXTCUDA_HOST    the GPU host, when not given as the argument     SKIP_BUILD=1   use dist/ as it is
#   SKIP_KERNELS=1   do not rebuild the CUDA libraries            RESTART=0      put the files there, leave the service
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:-${NEXTCUDA_HOST:-}}
[ -n "$HOST" ] || { echo "usage: tooling/deploy.sh HOST   (an ssh host with nvcc and CUDA; or NEXTCUDA_HOST)" >&2; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
RSYNC=(rsync -a -e "ssh -o BatchMode=yes")
quiet() { grep -v 'alacritty\|^\]11\|snapshot' || true; }

[ -n "${SKIP_BUILD:-}" ] || ./build.sh
echo "==> $HOST: files"
"${SSH[@]}" 'mkdir -p ~/nextcuda/dist ~/nextcuda/src ~/.config/systemd/user' 2>&1 | quiet
# the libraries in dist/ on the host are built there (nvcc): a copy of dist/ must not remove them
"${RSYNC[@]}" --exclude 'lib*.so' dist/ "$HOST":nextcuda/dist/ 2>&1 | quiet
# the program there is this one (a copy that failed - a full drive - once left the old one running, said nothing)
want=$(md5sum dist/nextcuda | cut -d' ' -f1)
have=$("${SSH[@]}" 'md5sum ~/nextcuda/dist/nextcuda 2>/dev/null | cut -d" " -f1; df -h ~ | tail -1' 2>&1 | quiet)
if [ "$(echo "$have" | head -1)" != "$want" ]; then
  echo "==> $HOST: the copy of dist/nextcuda did not arrive (the drive there: $(echo "$have" | tail -1)); nothing restarted" >&2
  exit 1
fi
"${RSYNC[@]}" --delete kernels/ "$HOST":nextcuda/src/kernels/ 2>&1 | quiet
if [ -z "${SKIP_KERNELS:-}" ]; then
  echo "==> $HOST: kernels"
  "${SSH[@]}" 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/nextcuda/src/kernels && OUT=$HOME/nextcuda/dist OBJ=$HOME/nextcuda/src/obj ./build.sh 2>&1 | tail -8' 2>&1 | quiet
fi
"${RSYNC[@]}" tooling/nextcuda.service "$HOST":.config/systemd/user/nextcuda.service 2>&1 | quiet
[ "${RESTART:-1}" = 0 ] && { echo "==> files are there; the service is as it was"; exit 0; }
echo "==> $HOST: service"
"${SSH[@]}" 'systemctl --user daemon-reload && systemctl --user enable nextcuda.service >/dev/null 2>&1; systemctl --user restart nextcuda.service' 2>&1 | quiet
# up when the front door answers (any HTTP status: the API wants a token)
for _ in $(seq 1 30); do
  code=$("${SSH[@]}" 'curl -s -o /dev/null -m 3 -w "%{http_code}" http://127.0.0.1:8000/ 2>/dev/null' 2>&1 | quiet | tail -1)
  case "$code" in 200|401|403|404) echo "==> :8000 answers ($code)"; break ;; esac
  sleep 1
done
"${SSH[@]}" 'systemctl --user --no-pager --lines=0 status nextcuda.service | head -4' 2>&1 | quiet
