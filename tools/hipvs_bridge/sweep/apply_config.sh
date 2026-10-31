#!/usr/bin/env bash
# Write a swept CAGRA configuration into the bridge defaults, then verify.
#
# usage: apply_config.sh <sweep.csv> [row]
#   row: 1-based data row in the CSV (row 1 is the frozen-V5 baseline run).
#        Without [row], the fastest build whose recall10/recall100 deltas vs
#        row 1 stay within -0.001 (contest gate 0.005; keep headroom).
#
# Patches the sk_env_size() fallback literals in seekdb_cuvs_bridge.c, runs the
# host tests, and leaves the change uncommitted for review.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CSV=${1:?usage: apply_config.sh sweep.csv [row]}
ROW=${2:-}

[[ -f $CSV ]] || { echo "no such file: $CSV" >&2; exit 2; }

# Pick the row (python3; fall back to explicit row numbering only).
if [[ -z $ROW ]]; then
  ROW=$(python3 - "$CSV" << 'PYEOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 11 and r[6] != "build_s"]
if len(rows) < 2:
    sys.exit("need at least a baseline row and one candidate")
b = rows[0]
b10, b100, bb = float(b[9]), float(b[10]), float(b[6])
best, best_t = None, None
for i, r in enumerate(rows[1:], start=2):
    try:
        t, r10, r100 = float(r[6]), float(r[9]), float(r[10])
    except ValueError:
        continue
    if t <= 0 or r10 < b10 - 0.001 or r100 < b100 - 0.001:
        continue
    if best is None or t < best_t:
        best, best_t = i, t
if best is None:
    sys.exit("no candidate stays within the 0.001 recall margin; keep V5 defaults")
print(best)
PYEOF
)
  echo "selected row $ROW"
fi

MAP=$(python3 - "$CSV" "$ROW" << 'PYEOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 11 and r[6] != "build_s"]
r = rows[int(sys.argv[2]) - 1]
names = {0: "AUTO_SELECT", 1: "IVF_PQ", 2: "NN_DESCENT", 3: "ITERATIVE_CAGRA_SEARCH"}
print(names.get(int(r[0]), "NN_DESCENT"), r[1], r[2], r[3], r[4], r[5])
PYEOF
)
read -r ALGO GD IGD NITER ITOPK SW <<< "$MAP"
echo "applying: algo=$ALGO gd=$GD igd=$IGD niter=$NITER itopk_min=$ITOPK search_width=$SW"

python3 - "$BRIDGE_DIR/seekdb_cuvs_bridge.c" "$BRIDGE_DIR/tests/test_bridge_host.cpp" \
  "$ALGO" "$GD" "$IGD" "$NITER" "$ITOPK" "$SW" << 'PYEOF'
import re, sys
bridge_path, test_path = sys.argv[1], sys.argv[2]
algo, gd, igd, niter, itopk, sw = sys.argv[3:9]

src = open(bridge_path).read()
subs = [
    (r'(sk_env_size\("OBCUVS_BUILD_ALGO", \(size_t\))NN_DESCENT(, 0, 3\))', r"\g<1>" + algo + r"\g<2>"),
    (r'(sk_env_size\("OBCUVS_GRAPH_DEGREE", )128(, 16, 512\))', r"\g<1>" + gd + r"\g<2>"),
    (r'(sk_env_size\("OBCUVS_INTERMEDIATE_DEGREE", )256(, 16, 1024\))', r"\g<1>" + igd + r"\g<2>"),
    (r'(sk_env_size\("OBCUVS_NNDESCENT_NITER", )100(, 1, 500\))', r"\g<1>" + niter + r"\g<2>"),
    (r'(sk_env_size\("OBCUVS_ITOPK_MIN", )512(, 64, 4096\))', r"\g<1>" + itopk + r"\g<2>"),
    (r'(sk_env_size\("OBCUVS_SEARCH_WIDTH", )4(, 1, 16\))', r"\g<1>" + sw + r"\g<2>"),
]
for pattern, repl in subs:
    new, n = re.subn(pattern, repl, src)
    if n != 1:
        sys.exit(f"bridge patch failed ({n} matches): {pattern}")
    src = new
open(bridge_path, "w").write(src)
print("bridge defaults patched")

# The host tests assert the shipped defaults; retarget them together.
# itopk assertions: stub default_itopk=128, topk=10 and topk=200 with 4*topk.
itopk10 = max(40, int(itopk), 128)
itopk200 = max(800, int(itopk), 128)
t = open(test_path).read()
tsubs = [
    (r"static size_t expect_graph_degree=128, expect_intermediate_degree=256, expect_niter=100, expect_search_width=4;",
     f"static size_t expect_graph_degree={gd}, expect_intermediate_degree={igd}, expect_niter={niter}, expect_search_width={sw};"),
    (r"expect_graph_degree=128; expect_intermediate_degree=256; expect_niter=100; expect_search_width=4;",
     f"expect_graph_degree={gd}; expect_intermediate_degree={igd}; expect_niter={niter}; expect_search_width={sw};"),
    (r"assert\(last_graph_degree==128 && last_intermediate_degree==256 && last_niter==100\);",
     f"assert(last_graph_degree=={gd} && last_intermediate_degree=={igd} && last_niter=={niter});"),
    (r"assert\(last_itopk==800\);", f"assert(last_itopk=={itopk200});"),
    (r"assert\(last_itopk==512\);", f"assert(last_itopk=={itopk10});"),
]
for pattern, repl in tsubs:
    new, n = re.subn(pattern, repl, t)
    if n < 1:
        sys.exit(f"test patch failed ({n} matches): {pattern}")
    t = new
open(test_path, "w").write(t)
print("host test expectations patched")
PYEOF

echo "== host tests =="
"$SCRIPT_DIR/../tests/run_host_tests.sh" /tmp/apply-config-tests 2>&1 | tail -1
echo "== change summary =="
git -C "$BRIDGE_DIR/.." diff --stat
echo "review the diff, then commit with the sweep evidence."
