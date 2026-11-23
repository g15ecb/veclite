#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
SAMPLES="${SAMPLES:-20000}"
fail=0
expect() {
  local inst="$1" want="$2" inv="$3" out got
  out=$(quint run keys.qnt --main "$inst" --invariant "$inv" --max-samples "$SAMPLES" --max-steps 25 --seed 1 2>&1)
  if grep -q "No violation found" <<<"$out"; then got=pass; elif grep -q "Found an issue" <<<"$out"; then got=fail; else got=error; fi
  printf '%-30s %-22s want=%-4s got=%s\n' "$inst" "$inv" "$want" "$got"
  [[ "$got" == "$want" ]] || { fail=1; tail -30 <<<"$out"; }
}
expect fixed pass inv
for w in witnessKeyReused witnessRejected witnessRollbackCrash; do expect fixed fail "$w"; done
for m in mutantRollbackKeepsIndex mutantDeleteKeepsKey mutantRecoveryIgnoresDeletes mutantDupCheckCommittedOnly; do expect "$m" fail inv; done
exit $fail
