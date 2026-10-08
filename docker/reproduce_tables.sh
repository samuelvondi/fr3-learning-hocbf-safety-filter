#!/usr/bin/env bash
# Reproduce Tables I-III and Fig. 4 (README section 1) inside the container.
#
# Environment overrides:
#   JOBS        parallel shards (default: nproc)
#   LIMIT       only run the first N benchmark scenarios (smoke test)
#   OUT         output directory (default: /out)
set -euo pipefail

JOBS="${JOBS:-$(nproc)}"
OUT="${OUT:-/out}"
LIST=/data/pd_failed_460.txt
MODEL=/repo/artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt
PREFIX="$OUT/pd_failed_final"

mkdir -p "$OUT/shards"
cd /repo/src

# --- 1. split the benchmark list into shards --------------------------------
if [[ -n "${LIMIT:-}" ]]; then
    head -n "$LIMIT" "$LIST" > "$OUT/shards/list.txt"
else
    cp "$LIST" "$OUT/shards/list.txt"
fi
N=$(wc -l < "$OUT/shards/list.txt")
(( JOBS > N )) && JOBS=$N
rm -f "$OUT"/shards/part_*
split -n "l/$JOBS" -d -a 3 --additional-suffix=.txt "$OUT/shards/list.txt" "$OUT/shards/part_"
echo "running $N scenarios x 4 modes in $JOBS shards"

# --- 2. run the four controllers --------------------------------------------
# One thread per process: the shards already saturate the cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for part in "$OUT"/shards/part_*; do
    [[ "$part" == *.log ]] && continue
    python fr3_compare_rollout_policies.py "$part" \
        --modes fixed,davide_online,nn,nn_g12_fallback \
        --p1-fixed 10.0 --p2-fixed 15.0 \
        --model "$MODEL" \
        --out-prefix "$part" > "$part.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if (( fail )); then
    echo "a shard failed; see $OUT/shards/*.log" >&2
    exit 1
fi

# --- 3. merge shards ---------------------------------------------------------
python - "$OUT/shards" "$PREFIX" <<'PY'
import sys
from pathlib import Path
import pandas as pd
shards, prefix = Path(sys.argv[1]), sys.argv[2]
for kind in ("summary", "steps"):
    parts = sorted(shards.glob(f"part_*_{kind}.csv"))
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(f"{prefix}_{kind}.csv", index=False)
    print(f"merged {len(parts)} shards -> {prefix}_{kind}.csv")
PY

# --- 4. Table I --------------------------------------------------------------
python - "$PREFIX" <<'PY' | tee "$OUT/table1.txt"
import sys
import pandas as pd
TOL = 1e-4
d = pd.read_csv(f"{sys.argv[1]}_summary.csv")
d["success"] = d["success"].astype(str).str.lower().isin(["true", "1", "1.0"])
d["unsafe"] = pd.to_numeric(d["min_h"], errors="coerce") < -TOL
print(f"{'':14s} {'S/succ':>6s} {'U/succ':>6s} {'S/fail':>6s} {'U/fail':>6s}   viol     FB     g/b")
for m, label in [("fixed", "Fixed g/b"), ("davide_online", "Fixed + FB"),
                 ("nn", "Learned (NN)"), ("nn_g12_fallback", "NN + Fallback")]:
    s = d[d["mode"] == m]
    print(f"{label:14s} {100*( s.success & ~s.unsafe).mean():6.1f} "
          f"{100*( s.success &  s.unsafe).mean():6.1f} "
          f"{100*(~s.success & ~s.unsafe).mean():6.1f} "
          f"{100*(~s.success &  s.unsafe).mean():6.1f}   "
          f"{100*s.unsafe.mean():5.1f}  {100*s.fallback_rate.mean():5.1f}  "
          f"{s.p1_mean.mean():.1f}/{s.p2_mean.mean():.1f}")
PY

# --- 5. Tables II-III and Fig. 4 --------------------------------------------
python fr3_make_final_metric_tables_generic_fixed.py \
  --run "${PREFIX}_summary.csv" "${PREFIX}_steps.csv" \
        "fixed=Fixed gamma/beta" "davide_online=Fixed + Fallback" \
        "nn=Learned (NN)" "nn_g12_fallback=NN + Fallback" \
  --order "Fixed gamma/beta" "Fixed + Fallback" "Learned (NN)" "NN + Fallback" \
  --safety-tol 1e-4 --out-prefix "$OUT/fr3_final_metric_tables"

python fr3_plot_compare_results.py \
  --steps "${PREFIX}_steps.csv" --summary "${PREFIX}_summary.csv" \
  --out-dir "$OUT/fr3_compare_plots" --h-tol 1e-4

echo "done -> $OUT"
printf 'total time: %dh %02dm %02ds\n' $((SECONDS / 3600)) $((SECONDS % 3600 / 60)) $((SECONDS % 60))
