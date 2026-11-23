#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

echo "== quint: top-k"
SAMPLES="${SAMPLES:-20000}" formal/check.sh
echo "== quint: file format + WAL"
SAMPLES="${SAMPLES:-20000}" formal/check_wal.sh

echo "== quint: key index"
SAMPLES="${SAMPLES:-20000}" formal/check_keys.sh

echo "== quint: GPU routing"
SAMPLES="${SAMPLES:-20000}" formal/check_gpu_route.sh

echo "== lean"
for f in TopK Wal; do
  out=$(lean "formal/lean/$f.lean" 2>&1)
  if grep -q "error\|sorry" <<<"$out"; then echo "$out"; exit 1; fi
  grep "depends on axioms" <<<"$out"
done

echo "== lean mutants must fail"
mkdir -p build/leanmut
sed 's/(ins x acc).take k$/(ins x acc).take (k - 1)/' formal/lean/TopK.lean > build/leanmut/TopKMutant.lean
sed 's/{ r with live := false }/{ r with live := !r.live }/' formal/lean/Wal.lean > build/leanmut/WalToggle.lean
sed 's/(∀ M m, pick M = some m → m ∈ M ∧ ∀ x ∈ M, le m x) →/(∀ M m, pick M = some m → m ∈ M ∧ ∀ x ∈ M, le x m) →/' formal/lean/TopK.lean > build/leanmut/SelectMax.lean
grep -q "le x m) →" build/leanmut/SelectMax.lean
for m in TopKMutant WalToggle SelectMax; do
  if lean "build/leanmut/$m.lean" >/dev/null 2>&1; then echo "lean mutant $m unexpectedly proved"; exit 1; fi
  echo "lean mutant $m rejected"
done
echo "FORMAL CHECKS PASSED"
