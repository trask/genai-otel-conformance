#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../versions.env"

weaver_url="https://github.com/open-telemetry/weaver/releases/download/${WEAVER_VERSION}/weaver-installer.sh"

installed=false
for attempt in 1 2 3 4 5; do
  if curl -sSfL "$weaver_url" | sh; then
    installed=true
    break
  fi
  echo "Weaver install attempt $attempt failed for ${WEAVER_VERSION}, retrying in 10s..."
  sleep 10
done

if [[ "$installed" != true ]]; then
  echo "Failed to install Weaver ${WEAVER_VERSION} after 5 attempts." >&2
  exit 1
fi

echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
