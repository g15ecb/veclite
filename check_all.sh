#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
L="$PWD/.venv/lib/python3.12/site-packages/modular/lib"

echo "== quint"
SAMPLES="${SAMPLES:-20000}" formal/check.sh

echo "== lean"
out=$(lean formal/lean/TopK.lean 2>&1)
if grep -q "error\|sorry" <<<"$out"; then echo "$out"; exit 1; fi
grep "depends on axioms" <<<"$out"

echo "== lean mutant must fail"
mkdir -p build/leanmut
sed 's/(ins x acc).take k$/(ins x acc).take (k - 1)/' formal/lean/TopK.lean > build/leanmut/TopKMutant.lean
grep -q "take (k - 1)" build/leanmut/TopKMutant.lean
if lean build/leanmut/TopKMutant.lean >/dev/null 2>&1; then echo "lean mutant unexpectedly proved"; exit 1; fi
echo "lean mutant rejected"

echo "== build"
mkdir -p build
mojo build src/vdb.mojo --emit shared-lib -o build/libvdb.dylib
clang -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -Iinclude tests/c_host.c -Lbuild -lvdb \
  -Wl,-rpath,"$PWD/build" -Wl,-rpath,"$L" -o build/c_host
./build/c_host

echo "== code mutants must fail"
for m in NO_TIE_BREAK POST_FILTER LOCAL_K_MINUS_ONE; do
  mkdir -p "build/mut_$m"
  mojo build -D "VDB_MUTANT_$m=true" src/vdb.mojo --emit shared-lib -o "build/mut_$m/libvdb.dylib"
  clang -O2 -Iinclude tests/c_host.c -L"build/mut_$m" -lvdb -Wl,-rpath,"$PWD/build/mut_$m" -Wl,-rpath,"$L" -o "build/mut_$m/c_host"
  if "./build/mut_$m/c_host" >/dev/null 2>&1; then echo "mutant $m survived"; exit 1; fi
  echo "mutant $m caught: $("./build/mut_$m/c_host" 2>/dev/null | tail -1)"
done

echo "== conformance + bench smoke"
python -I bench/bench.py 20000 64 | grep min_recall
echo "ALL CHECKS PASSED"
