#!/usr/bin/env bash
# Build and publish to the webgpu-chat container (~/docker/webgpu-chat), served at https://llm.aminox.dev.
# The models are not part of the site: browsers download them from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${WEBGPU_CHAT_DEPLOY_DIR:-$HOME/docker/webgpu-chat}"  # override for your own server

npm run build
rsync -a --delete dist/ "$TARGET/data/html/"
if ! cmp -s deploy/nginx.conf "$TARGET/nginx.conf"; then
  echo "Aviso: deploy/nginx.conf difiere de $TARGET/nginx.conf; cópialo y recarga nginx (docker exec webgpu-chat nginx -s reload)."
fi
echo "Web publicada en $TARGET/data/html"
