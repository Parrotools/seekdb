#!/usr/bin/env bash
# One-shot GPU session for the CAGRA parameter sweep.
#
# Compiles the real bridge with the official build.sh, builds the sweep
# harness, generates clustered synthetic data when no real fvec set is
# provided, then measures a ladder of build/search configurations. The point
# is to find the fastest configuration whose Recall@10/100 delta against the
# frozen V5 settings stays within margin (the contest gate is 0.005 on the
# official datasets; we select with 0.001 to leave headroom for Hidden250K).
#
# usage: run_param_sweep.sh <output_dir>
# env:
#   SWEEP_BASE=<base.fvec>  SWEEP_QUERIES=<queries.fvec>   real data (optional)
#   SWEEP_NQ=<n>            queries per config (default 4096, eval's nq)
#   HIPVS_ROOT=/opt/hipvs   ROCM_ROOT=/opt/rocm
#
# CSV columns: algo,gd,igd,niter,itopk_min,sw,build_s,qps10,qps100,recall10,recall100
# Build algo: 0=NN_DESCENT 1=IVF_PQ 2=AUTO_SELECT 3=ITERATIVE_CAGRA_SEARCH
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
OUT=${1:?usage: run_param_sweep.sh OUTPUT_DIRECTORY}
NQ=${SWEEP_NQ:-4096}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

[[ -x "$ROCM_ROOT/bin/hipcc" ]] || { echo "hipcc is not executable: $ROCM_ROOT/bin/hipcc" >&2; exit 2; }
[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || { echo "hipVS headers missing under $HIPVS_ROOT/include" >&2; exit 2; }

echo "== 1. official bridge build =="
HIPVS_ROOT="$HIPVS_ROOT" ROCM_ROOT="$ROCM_ROOT" \
  "$SCRIPT_DIR/../build.sh" "$OUT/libseekdb_cuvs_bridge.so"

echo "== 2. sweep harness build =="
g++ -std=c++17 -O2 -Wall -Wextra -pthread \
  -I"$ROCM_ROOT/include" -I"$HIPVS_ROOT/include" \
  "$SCRIPT_DIR/param_sweep.cpp" "$OUT/libseekdb_cuvs_bridge.so" \
  -L"$HIPVS_ROOT/lib" -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" -lcuvs_c \
  -o "$OUT/param_sweep"

BASE=${SWEEP_BASE:-$OUT/synth.base.fvec}
QUERIES=${SWEEP_QUERIES:-$OUT/synth.queries.fvec}
if [[ ! -f $BASE || ! -f $QUERIES ]]; then
  echo "== 3. synthetic clustered data (no SWEEP_BASE given) =="
  gcc -O2 -Wall -Wextra "$SCRIPT_DIR/gen_synthetic.c" -lm -o "$OUT/gen_synthetic"
  # 250k x 256 rows, 4096 queries, 4096 clusters: a Hidden250K-shaped proxy.
  "$OUT/gen_synthetic" "$OUT/synth" 250000 256 "$NQ" 4096 20261030
  BASE=$OUT/synth.base.fvec
  QUERIES=$OUT/synth.queries.fvec
fi
echo "base=$BASE queries=$QUERIES nq=$NQ"

run_config() {  # algo gd igd niter itopk sw
  OBCUVS_BUILD_ALGO=$1 OBCUVS_GRAPH_DEGREE=$2 OBCUVS_INTERMEDIATE_DEGREE=$3 \
  OBCUVS_NNDESCENT_NITER=$4 OBCUVS_ITOPK_MIN=$5 OBCUVS_SEARCH_WIDTH=$6 \
    "$OUT/param_sweep" "$BASE" "$QUERIES" "$OUT/truth.ivec" "$NQ" | tee -a "$OUT/sweep.csv"
}

: > "$OUT/sweep.csv"

echo "== 4. build-parameter ladder (search fixed at V5: itopk 512 / sw 4) =="
run_config 0 128 256 100 512 4   # frozen V5 baseline settings
run_config 0 128 256 50  512 4
run_config 0 128 256 25  512 4
run_config 0 128 192 50  512 4
run_config 0 96  192 50  512 4
run_config 0 96  128 40  512 4
run_config 0 64  128 50  512 4
run_config 0 64  128 25  512 4
run_config 0 64  128 16  512 4
run_config 1 64  128 100 512 4   # IVF_PQ all-neighbors build
run_config 1 128 256 100 512 4
run_config 2 64  128 50  512 4   # AUTO_SELECT

echo "== 5. search-parameter ladder (build fixed at V5 settings) =="
run_config 0 128 256 100 256 4
run_config 0 128 256 100 128 4
run_config 0 128 256 100 512 2
run_config 0 128 256 100 256 2

echo "== 6. decision table (deltas vs row 1 = frozen V5 settings) =="
python3 - "$OUT/sweep.csv" << 'PYEOF' || echo "(python3 unavailable; raw CSV above)"
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 10]
if len(rows) < 2:
    print("no complete rows to compare"); sys.exit(0)
b = rows[0]
try:
    bb = float(b[5])
    b10, b100 = float(b[8]), float(b[9])
except ValueError:
    print("baseline row malformed"); sys.exit(0)
print(f"baseline: algo={b[0]} gd={b[1]} igd={b[2]} niter={b[3]} "
      f"build={bb:.3f}s recall10={b10:.4f} recall100={b100:.4f}")
for r in rows[1:]:
    try:
        build = float(r[5]); r10, r100 = float(r[8]), float(r[9])
    except ValueError:
        print(f"skipped malformed row: {r}"); continue
    speed = bb / build if build > 0 else 0.0
    d10, d100 = r10 - b10, r100 - b100
    flag = "  <== recall gate risk" if d10 < -0.001 or d100 < -0.001 else ""
    print(f"algo={r[0]} gd={r[1]} igd={r[2]} niter={r[3]} itopk={r[4]} sw={r[5]}: "
          f"build={build:.3f}s ({speed:.2f}x vs baseline) "
          f"recall10={r10:.4f} ({d10:+.4f}) recall100={r100:.4f} ({d100:+.4f}){flag}")
PYEOF
echo "done: $OUT/sweep.csv"
