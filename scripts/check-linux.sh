#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
platform="${PLATFORM:-linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')}"
docker build --platform "$platform" -t veclite-test -f docker/Dockerfile .
docker run --rm --platform "$platform" -e SEEDS="${SEEDS:-500}" veclite-test "$@"
