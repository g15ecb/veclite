#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
SAMPLES="${SAMPLES:-20000}"
STEPS="${STEPS:-30}"
fail=0
expect() {
  local inst="$1" want="$2" inv="$3" samples="${4:-$SAMPLES}" out got
  out=$(quint run wal.qnt --main "$inst" --invariant "$inv" --max-samples "$samples" --max-steps "$STEPS" --seed 1 2>&1)
  if grep -q "No violation found" <<<"$out"; then got=pass; elif grep -q "Found an issue\|violated" <<<"$out"; then got=fail; else got=error; fi
  printf '%-26s %-30s want=%-4s got=%s\n' "$inst" "$inv" "$want" "$got"
  [[ "$got" == "$want" ]] || { fail=1; tail -30 <<<"$out"; }
}
expect base pass inv
for w in witnessTornRecovered witnessRestartThenCrash witnessReaderBehindCheckpoint witnessReplayOverCheckpoint; do expect base fail "$w"; done
expect mutantAckBeforeSync fail ackedDurable
expect mutantCkptIgnoresReaders fail readerSnapshot
expect mutantRestartEarly fail inv
expect mutantSkipDataSync fail inv
expect mutantNoRecoveryTruncate fail recoveryAtomic
expect mutantNoChain pass inv
expect mutantReaderNoSlotBound pass inv
for m in fixed unsyncedRestartHdr noRecoveryTruncate; do
  out=$(quint test wal_test.qnt --main "$m" 2>&1)
  n=$(grep -c "^    ok " <<<"$out")
  printf '%-26s %-30s want=2ok  got=%sok\n' "$m" "directed scenarios" "$n"
  [[ "$n" == 2 ]] || { fail=1; tail -30 <<<"$out"; }
done
exit $fail
