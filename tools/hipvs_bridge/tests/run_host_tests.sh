#!/usr/bin/env bash
# Host doubles exercise allocation/lifetime control flow, never GPU correctness.
set -Eeuo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT=${1:?usage: run_host_tests.sh OUTPUT_DIRECTORY}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
COMMON=(-Wall -Wextra -Werror -pthread -I"$SCRIPT_DIR/stubs" -I"$BRIDGE_DIR")
OPT=(-O2)
if [[ ${SANITIZE:-0} == 1 ]]; then
  OPT=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined)
fi
for language in c cpp; do
  if [[ $language == c ]]; then
    "${CC:-gcc}" -std=c11 "${COMMON[@]}" "${OPT[@]}" -D__HIP_PLATFORM_AMD__ \
      -c "$BRIDGE_DIR/seekdb_cuvs_bridge.c" -o "$OUTPUT/bridge-$language.o"
  else
    "${CXX:-g++}" -x c++ -std=c++17 "${COMMON[@]}" "${OPT[@]}" -D__HIP_PLATFORM_AMD__ \
      -c "$BRIDGE_DIR/seekdb_cuvs_bridge.c" -o "$OUTPUT/bridge-$language.o"
  fi
  "${CXX:-g++}" -std=c++17 "${COMMON[@]}" "${OPT[@]}" "$SCRIPT_DIR/test_bridge_host.cpp" \
    "$OUTPUT/bridge-$language.o" -o "$OUTPUT/test-$language"
  "$OUTPUT/test-$language" "$OUTPUT/mock-$language.snapshot"
done
if [[ ${SANITIZE:-0} != 1 ]]; then
  # Use the exact frozen bridge, with identical compiler flags and test doubles.
  git -C "$REPO_ROOT" show 634888d9b037153fa143d2c913ba4b25503e1d62:tools/hipvs_bridge/seekdb_cuvs_bridge.c > "$OUTPUT/baseline.c"
  "${CC:-gcc}" -std=c11 "${COMMON[@]}" "${OPT[@]}" -D__HIP_PLATFORM_AMD__ \
    -c "$OUTPUT/baseline.c" -o "$OUTPUT/baseline.o"
  "${CXX:-g++}" -std=c++17 "${COMMON[@]}" "${OPT[@]}" "$SCRIPT_DIR/test_bridge_host.cpp" \
    "$OUTPUT/baseline.o" -o "$OUTPUT/test-baseline"
  "$OUTPUT/test-baseline" --benchmark > "$OUTPUT/baseline-mock.json"
  "$OUTPUT/test-c" --benchmark > "$OUTPUT/candidate-mock.json"
  cat "$OUTPUT/baseline-mock.json" "$OUTPUT/candidate-mock.json"
fi
