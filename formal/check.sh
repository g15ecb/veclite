#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
SAMPLES="${SAMPLES:-20000}"
fail=0
expect() {
  local inst="$1" want="$2" inv="$3" out got
  out=$(quint run topk.qnt --main "$inst" --invariant "$inv" --max-samples "$SAMPLES" --max-steps 12 --seed 1 2>&1)
  if grep -q "No violation found" <<<"$out"; then got=pass; elif grep -q "Found an issue\|violated" <<<"$out"; then got=fail; else got=error; fi
  printf '%-22s %-24s want=%-4s got=%s\n' "$inst" "$inv" "$want" "$got"
  [[ "$got" == "$want" ]] || { fail=1; tail -40 <<<"$out"; }
}
for i in batch single moreChunksThanRows kExceedsRows; do expect "$i" pass inv; expect "$i" fail witnessDone; done
for w in witnessTruncated witnessTiesInResult witnessFiltersDiffer; do expect batch fail "$w"; done
for m in mutantNoTieBreak mutantPostFilter mutantLocalKMinusOne mutantQueryMix; do expect "$m" fail inv; done
exit $fail
