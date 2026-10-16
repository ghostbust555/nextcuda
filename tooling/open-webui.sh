#!/usr/bin/env bash
# tooling/open-webui.sh: (re)creates the Open WebUI container on this host, its chats through the front door (:8000/v1
# with the API token from serve.json): every enabled chat model is listed, loaded on its first request. Its data (the
# accounts, the chats) is the `open-webui` volume, kept. Run it again after the token changes; open-webui.service
# starts the container with the box.
#   OWUI_PORT (3000)   OWUI_IMAGE (ghcr.io/open-webui/open-webui:main)   NEXTCUDA_HOME (~/nextcuda)
set -euo pipefail
HOME_DIR=${NEXTCUDA_HOME:-$HOME/nextcuda}
TOKEN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['token'])" "$HOME_DIR/serve.json")
podman rm -f open-webui >/dev/null 2>&1 || true
# X-Keep-Up-S is sent per connection from Open WebUI's settings (Admin > Connections > Headers: {"X-Keep-Up-S": "300"})
podman run -d --name open-webui -p "${OWUI_PORT:-3000}:8080" -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://host.containers.internal:8000/v1 -e OPENAI_API_KEY="$TOKEN" \
    -e ENABLE_OLLAMA_API=false -e AIOHTTP_CLIENT_TIMEOUT=1800 \
    "${OWUI_IMAGE:-ghcr.io/open-webui/open-webui:main}" >/dev/null
echo "open-webui on :${OWUI_PORT:-3000} -> the front door's /v1"
