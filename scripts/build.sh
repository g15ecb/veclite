#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
out="${1:-build}"
shift || true
case "$(uname -s)" in
  Darwin) ext=dylib; gpu_default=1 ;;
  Linux) ext=so; gpu_default=0 ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
gpu="${VECLITE_GPU:-$gpu_default}"
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "VECLITE_GPU must be 0 or 1" >&2; exit 1; }
mkdir -p "$out"
srcs=(src/kernel.mojo src/store.mojo)
if [[ "$gpu" == 1 ]]; then srcs+=(src/gpu.mojo); else srcs+=(src/gpu_stub.mojo); fi
{ cat "${srcs[0]}"; for f in "${srcs[@]:1}"; do grep -v '^from kernel import' "$f"; done; } > "$out/lib.mojo"
mojo build "$@" "$out/lib.mojo" --emit shared-lib -o "$out/libveclite.$ext"
