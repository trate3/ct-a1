#!/usr/bin/env bash
# RQ1: certification overhead as a fraction of a REAL evaluation, across model sizes.
#
# The point of the sweep is the denominator. Protocol cost (hash, sign, sealed-box
# transport, verify) is essentially fixed per round; inference cost grows with model
# size and token budget. So the overhead FRACTION is only meaningful when measured
# against real generation, and it should shrink as the model grows. A mock workload
# divides a real protocol cost by ~zero and reports a meaningless ratio.
#
# Each point is run twice, once per arm order, because RUNBOOK Phase 1 notes the
# process-lifetime RSS high-water mark does not reset between arms and the first arm
# is advantaged. Counterbalancing lets you average that out.
#
# Usage:
#   MODELS="google/gemma-4-E2B-it" N=8 TOKENS=128 REPEATS=5 \
#     bash evals_runner/rq1_sweep.sh
set -uo pipefail

MODELS=${MODELS:-"google/gemma-4-E2B-it"}
N=${N:-8}
TOKENS=${TOKENS:-128}
OUTDIR=${OUTDIR:-evals_results/rq1}
# Repeats matter: on a real eval the protocol cost can be smaller than run-to-run
# timing noise, so a single pair can report a NEGATIVE overhead. Without repeats you
# cannot put an upper bound on the overhead, only observe its sign by luck.
REPEATS=${REPEATS:-1}
PY=${PY:-python}

mkdir -p "$OUTDIR"
echo "sweep: models=[$MODELS] n=$N max_new_tokens=$TOKENS -> $OUTDIR"

failed=0
for model in $MODELS; do
  slug=$(printf '%s' "$model" | tr '/:' '__')
  for rep in $(seq 1 "$REPEATS"); do
  for order in direct-first certified-first; do
    flag=""
    [ "$order" = "certified-first" ] && flag="--certified-first"
    out="$OUTDIR/${slug}__n${N}__t${TOKENS}__${order}__r${rep}.json"
    echo ""
    echo "=== $model | $order | rep $rep/$REPEATS | n=$N tokens=$TOKENS ==="
    start=$(date +%s)
    # shellcheck disable=SC2086
    if $PY -m evals_runner taps confidence \
         --program-model "$model" \
         --n "$N" --max-new-tokens "$TOKENS" $flag \
         --certification-overhead-out "$out"; then
      echo "ok in $(( $(date +%s) - start ))s -> $out"
    else
      echo "FAILED: $model / $order (continuing)"
      failed=$((failed + 1))
    fi
  done
  done
done

echo ""
echo "sweep done; $failed run(s) failed"
echo "now price it:"
echo "  $PY -m evals_runner cost-report $OUTDIR --instance a3-highgpu-1g --out $OUTDIR/cost.json"
exit 0
