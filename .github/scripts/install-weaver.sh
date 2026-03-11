#!/usr/bin/env bash
set -euo pipefail

for attempt in 1 2 3 4 5; do
  if curl -sSfL https://github.com/open-telemetry/weaver/releases/latest/download/weaver-installer.sh | sh; then
    break
  fi
  echo "Weaver install attempt $attempt failed, retrying in 10s..."
  sleep 10
done

echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
