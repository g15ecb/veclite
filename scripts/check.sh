#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
SEEDS="${SEEDS:-3000}"
CC="${CC:-clang}"
L="$(python -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "modular", "lib"))')"
[[ -d "$L" ]] || { echo "Modular runtime libs not found at $L" >&2; exit 1; }
case "$(uname -s)" in Darwin) gpu_default=1 ;; *) gpu_default=0 ;; esac
export VECLITE_GPU="${VECLITE_GPU:-$gpu_default}"
SAN="-fsanitize=address,undefined"

cc() { "$CC" -O1 -g -Wall -Wextra -Werror -Iinclude "$1" -L"$2" -lveclite -lm -Wl,-rpath,"$PWD/$2" -Wl,-rpath,"$L" -o "$3" "${@:4}"; }
expect_fail() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "mutant $name survived"; exit 1; fi
  echo "mutant $name caught"
}

echo "== build ($(uname -s) $(uname -m), gpu=$VECLITE_GPU)"
scripts/build.sh build

echo "== C tests"
cc tests/c/c_host.c build build/c_host $SAN
./build/c_host
cc tests/c/dtype_test.c build build/dtype_test $SAN
./build/dtype_test 1000
cc tests/c/store_test.c build build/store_test $SAN
for dt in 0 1; do ./build/store_test "$SEEDS" 200 1 "$dt"; done
if [[ "$VECLITE_GPU" == 1 ]]; then
  cc tests/c/gpu_test.c build build/gpu_test
  ./build/gpu_test 300
fi

echo "== Python tests + example"
python -m pytest -q tests/python
PYTHONPATH=python python examples/quickstart.py

echo "== kernel mutants must fail"
for m in NO_TIE_BREAK POST_FILTER LOCAL_K_MINUS_ONE QUERY_MIX; do
  scripts/build.sh "build/mut_$m" -D "VECLITE_MUTANT_$m=true"
  cc tests/c/c_host.c "build/mut_$m" "build/mut_$m/c_host"
  expect_fail "$m" "./build/mut_$m/c_host"
  if [[ "$VECLITE_GPU" == 1 && "$m" != QUERY_MIX ]]; then
    cc tests/c/gpu_test.c "build/mut_$m" "build/mut_$m/gpu_test"
    expect_fail "gpu:$m" "./build/mut_$m/gpu_test" 150
  fi
done

if [[ "$VECLITE_GPU" == 1 ]]; then
  echo "== GPU routing mutants must fail"
  for m in GPU_STALE_LIVE GPU_SKIP_GAP; do
    scripts/build.sh "build/rmut_$m" -D "VECLITE_MUTANT_$m=true"
    lib="$(ls "$PWD"/build/rmut_$m/libveclite.*)"
    expect_fail "$m" env VECLITE_LIB="$lib" python -m pytest -q tests/python/test_device.py
  done
fi

echo "== storage mutants must fail"
for m in ACK_BEFORE_SYNC CKPT_IGNORES_READERS RESTART_EARLY SKIP_DATA_SYNC UNSYNCED_RESTART_HDR NO_RECOVERY_TRUNCATE ROLLBACK_KEEP_PENDING ROLLBACK_KEEP_ROWS; do
  scripts/build.sh "build/smut_$m" -D "VECLITE_MUTANT_$m=true"
  cc tests/c/store_test.c "build/smut_$m" "build/smut_$m/store_test"
  expect_fail "$m" "./build/smut_$m/store_test" "${MUTANT_SEEDS:-3000}" 200
done
echo "ALL CODE CHECKS PASSED"
