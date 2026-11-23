#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
SAMPLES="${SAMPLES:-20000}"
fail=0
expect() {
  local inst="$1" want="$2" inv="$3" out got
  out=$(quint run gpu_route.qnt --main "$inst" --invariant "$inv" --max-samples "$SAMPLES" --max-steps 25 --seed 1 2>&1)
  if grep -q "No violation found" <<<"$out"; then got=pass; elif grep -q "Found an issue" <<<"$out"; then got=fail; else got=error; fi
  printf '%-22s %-28s want=%-4s got=%s\n' "$inst" "$inv" "$want" "$got"
  [[ "$got" == "$want" ]] || { fail=1; tail -30 <<<"$out"; }
}
expect fixed pass inv
expect fixed fail witnessStaleGpu
expect fixed fail witnessDeleteAfterRefresh
expect fixed pass witnessReaderOlderThanGpu
expect mutantStaleLive fail inv
expect mutantSkipGap fail inv
expect mutantNoRefreshLimit pass inv
exit $fail
