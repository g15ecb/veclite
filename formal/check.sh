#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
SAMPLES="${SAMPLES:-20000}"
fail=0
expect() {
  local inst="$1" want="$2" inv="$3"
  local out
  out=$(quint run topk.qnt --main "$inst" --invariant "$inv" --max-samples "$SAMPLES" --max-steps 12 --seed 1 2>&1)
  local got
  if grep -q "No violation found" <<<"$out"; then got=pass; elif grep -q "Found an issue\|violated" <<<"$out"; then got=fail; else got=error; fi
  printf '%-22s %-26s want=%-4s got=%-5s\n' "$inst" "$inv" "$want" "$got"
  if [[ "$got" != "$want" ]]; then fail=1; echo "$out" | tail -40; fi
}
for i in filtered unfiltered moreChunksThanRows kExceedsRows; do expect "$i" pass inv; done
for i in filtered unfiltered; do
  expect "$i" fail witnessDone
  expect "$i" fail witnessTruncated
  expect "$i" fail witnessTiesInResult
done
expect kExceedsRows fail witnessDone
expect moreChunksThanRows fail witnessDone
expect mutantNoTieBreak fail inv
expect mutantPostFilter fail inv
expect mutantLocalKMinusOne fail inv
exit $fail
