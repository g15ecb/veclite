#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
L="$PWD/.venv/lib/python3.12/site-packages/modular/lib"
SEEDS="${SEEDS:-3000}"

echo "== quint: top-k"
SAMPLES="${SAMPLES:-20000}" formal/check.sh
echo "== quint: file format + WAL"
SAMPLES="${SAMPLES:-20000}" formal/check_wal.sh

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
for m in TopKMutant WalToggle; do
  if lean "build/leanmut/$m.lean" >/dev/null 2>&1; then echo "lean mutant $m unexpectedly proved"; exit 1; fi
  echo "lean mutant $m rejected"
done

cc() { clang -O1 -g -Wall -Wextra -Werror -Iinclude "$1" -L"$2" -lvdb -Wl,-rpath,"$PWD/$2" -Wl,-rpath,"$L" -o "$3" "${@:4}"; }

echo "== build + kernel tests"
./build.sh build
cc tests/c_host.c build build/c_host -fsanitize=address,undefined
./build/c_host
cc tests/dtype_test.c build build/dtype_test -fsanitize=address,undefined
./build/dtype_test 1000
cc tests/store_test.c build build/store_test -fsanitize=address,undefined
for dt in 0 1; do ./build/store_test "$SEEDS" 200 1 "$dt"; done

echo "== kernel mutants must fail"
for m in NO_TIE_BREAK POST_FILTER LOCAL_K_MINUS_ONE QUERY_MIX; do
  ./build.sh "build/mut_$m" -D "VDB_MUTANT_$m=true"
  cc tests/c_host.c "build/mut_$m" "build/mut_$m/c_host"
  if "./build/mut_$m/c_host" >/dev/null 2>&1; then echo "mutant $m survived"; exit 1; fi
  echo "mutant $m caught: $("./build/mut_$m/c_host" 2>/dev/null | tail -1)"
done

echo "== storage mutants must fail"
for m in ACK_BEFORE_SYNC CKPT_IGNORES_READERS RESTART_EARLY SKIP_DATA_SYNC UNSYNCED_RESTART_HDR NO_RECOVERY_TRUNCATE; do
  ./build.sh "build/smut_$m" -D "VDB_MUTANT_$m=true"
  cc tests/store_test.c "build/smut_$m" "build/smut_$m/store_test"
  if "./build/smut_$m/store_test" "$SEEDS" 200 >/dev/null 2>&1; then echo "mutant $m survived"; exit 1; fi
  echo "mutant $m caught: $("./build/smut_$m/store_test" "$SEEDS" 200 2>/dev/null | grep failing)"
done

echo "== storage defenses redundant under recovery truncation (model: mutantNoChain passes)"
./build.sh build/smut_NO_CHAIN -D VDB_MUTANT_NO_CHAIN=true
cc tests/store_test.c build/smut_NO_CHAIN build/smut_NO_CHAIN/store_test
echo "NO_CHAIN: $(./build/smut_NO_CHAIN/store_test "$SEEDS" 200 2>/dev/null | tail -1 | cut -c1-40)"

echo "== conformance smoke"
python -I bench/bench.py 20000 1536 1,8 | grep min_recall
echo "ALL CHECKS PASSED"
