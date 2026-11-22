#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
out="${1:-build}"
shift || true
mkdir -p "$out"
{ cat src/vdb.mojo; grep -v '^from vdb import' src/store.mojo; } > "$out/lib.mojo"
mojo build "$@" "$out/lib.mojo" --emit shared-lib -o "$out/libvdb.dylib"
