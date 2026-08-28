#!/usr/bin/env bash
# Wait for a detached run to finish, then report and stage it.
#
# Separate from run_overhead.sh on purpose: once the work is launched with setsid on
# the instance, the local process is only a watcher. Losing the watcher -- a dropped
# IAP tunnel, a closed laptop -- must not cost the run, so re-attaching has to be a
# first-class operation rather than a reason to measure everything again.
#
#   collect.sh                       newest run under evals_results/*/
#   collect.sh evals_results/financial/cloud_l4_20260824T110840Z
source "$(dirname "$0")/config.sh"
: "${CASE:=financial}"

OUT="${1:-}"
if [ -z "$OUT" ]; then
    OUT=$( { onvm "ls -dt ~/$REMOTE_ROOT/evals_results/$CASE/cloud_l4_* 2>/dev/null | head -1" \
             || true; } | tr -d '\r' | tail -1)
    OUT="${OUT#*/certified-taps/}"
    [ -n "$OUT" ] || { echo "no run found for CASE=$CASE" >&2; exit 1; }
fi
say "collecting $OUT"

while :; do
    state=$( { onvm "cd ~/$REMOTE_ROOT && \
        if [ -f $OUT/DONE ]; then echo DONE; \
        elif [ -f $OUT/FAILED ]; then echo FAILED; \
        else echo RUNNING \$(ls -1 $OUT/*_r*.json 2>/dev/null | wc -l); fi" 2>/dev/null \
        || true; } | tr -d '\r' | tail -1)
    case "$state" in
        DONE*)    echo "  complete"; break ;;
        FAILED*)  echo "  the run failed on the instance:" >&2
                  { onvm "tail -30 ~/$REMOTE_ROOT/$OUT/runner.log" || true; } >&2
                  exit 1 ;;
        RUNNING*) echo "  $(echo "$state" | awk '{print $2}') result files so far" ;;
        *)        echo "  (tunnel down; will retry)" ;;
    esac
    sleep 120
done

say "report"
{ onvm "cd ~/$REMOTE_ROOT && .venv/bin/python cloud/report.py $OUT" || true; }

say "staging to $BUCKET"
{ onvm "cd ~/$REMOTE_ROOT && gcloud storage cp -r $OUT $BUCKET/$CASE/ --project '$PROJECT'" \
  || echo "  staging failed; results remain on the instance"; }

say "done"
echo "fetch locally with:  CASE=$CASE cloud/fetch.sh ${OUT##*cloud_l4_}"
