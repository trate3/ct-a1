#!/usr/bin/env bash
# The measurement. Same agent, same tap, same test set, same seed, run twice per
# process: once directly and once through the trusted-workload/certifier protocol.
# The run aborts if the two arms disagree on (m, m_fp).
#
#   CASE=financial   the Apollo linear probe on Gemma-2  (needs the HF token)
#   CASE=confidence  the CoT-confidence tap on Gemma-4   (ungated, no token)
#
# REPEATS fresh processes per ordering, and both orderings:
#   direct_first_rN      arm order: direct, then certified
#   certified_first_rN   arm order: certified, then direct  (--certified-first)
#
# Two reasons for that shape. A single ordering confounds the protocol's cost with
# whatever happens only once per process -- CUDA context creation, allocator warm-up,
# cuDNN autotuning, page cache -- so each ordering is run and the pair averaged.
# And repeats are what make a median meaningful: one process gives one sample, and a
# single sample of a wall-clock difference this small is noise.
source "$(dirname "$0")/config.sh"

: "${CASE:=financial}"
: "${REPEATS:=5}"
: "${N:=8}"                      # confidence: TreeCut items
: "${PROGRAM_MODEL:=google/gemma-4-E2B-it}"
: "${MONITOR_MODEL:=}"           # empty -> the tap runs on the program's own model

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="evals_results/$CASE/cloud_l4_${STAMP}"

case "$CASE" in
  financial)
    trials="$N_TRAJECTORIES insider-trading scenarios x 2 rounds"
    args="--model '$MODEL' --n-trajectories $N_TRAJECTORIES --threshold $THRESHOLD"
    ;;
  confidence)
    trials="$N TreeCut items"
    args="--program-model '$PROGRAM_MODEL' --n $N"
    [ -n "$MONITOR_MODEL" ] && args="$args --monitor-model '$MONITOR_MODEL'"
    ;;
  *) echo "CASE must be financial or confidence" >&2; exit 1 ;;
esac

say "measurement: $CASE"
cat <<MSG
  instance     : $INSTANCE ($MACHINE_TYPE, ${ACCELERATOR}, $ZONE)
  device/dtype : $DEVICE / $DTYPE
  test set     : $trials, seed $SEED
  tokens       : max_new_tokens=$MAX_NEW_TOKENS
  design       : $REPEATS repeats x 2 orderings = $((REPEATS * 2)) processes,
                 each running the evaluation twice (direct and certified)
  output       : ~/$REMOTE_ROOT/$OUT/
MSG

# Run detached on the instance and poll, rather than holding the work open on an SSH
# channel for an hour. A dropped IAP tunnel killed the local half of a previous run
# after all ten processes had finished, losing the report and staging steps even
# though the measurements were complete and on disk. setsid + nohup means the work
# survives the tunnel; the marker files are what we wait on.
cat > /tmp/ct_runner.sh <<RUNNER
#!/usr/bin/env bash
cd ~/$REMOTE_ROOT || exit 1
mkdir -p $OUT
[ -f .hf_token ] && export HF_TOKEN=\$(cat .hf_token)
export HF_HUB_DISABLE_PROGRESS_BARS=1
for r in \$(seq 1 $REPEATS); do
    for arm in direct_first certified_first; do
        flag=''
        [ \$arm = certified_first ] && flag='--certified-first'
        echo "=== $CASE repeat \$r arm \$arm ==="
        .venv/bin/python -m evals_runner taps $CASE \
            $args \
            --device '$DEVICE' --dtype '$DTYPE' \
            --max-new-tokens $MAX_NEW_TOKENS \
            --seed $SEED \
            --certification-overhead-out $OUT/\${arm}_r\${r}.json \$flag \
            > $OUT/\${arm}_r\${r}.log 2>&1 || {
                echo "FAILED: \$arm r\$r"
                tail -25 $OUT/\${arm}_r\${r}.log
                touch $OUT/FAILED
                exit 1
            }
        echo "  ok"
    done
done
touch $OUT/DONE
RUNNER

gc compute scp /tmp/ct_runner.sh "$INSTANCE":/tmp/ct_runner.sh --zone "$ZONE" --tunnel-through-iap >/dev/null
onvm "cd ~/$REMOTE_ROOT && mkdir -p $OUT && setsid nohup bash /tmp/ct_runner.sh > $OUT/runner.log 2>&1 < /dev/null & echo launched"

say "waiting (polling; a dropped tunnel no longer loses the run)"
done_count=0
while :; do
    state=$( { onvm "cd ~/$REMOTE_ROOT && \
        if [ -f $OUT/DONE ]; then echo DONE; \
        elif [ -f $OUT/FAILED ]; then echo FAILED; \
        else echo RUNNING \$(ls -1 $OUT/*_r*.json 2>/dev/null | wc -l); fi" 2>/dev/null \
        || true; } | tr -d '\r' | tail -1)
    case "$state" in
        DONE*)   echo "  all $((REPEATS * 2)) processes complete"; break ;;
        FAILED*) echo "  run failed on the instance; see $OUT/runner.log" >&2
                 onvm "tail -30 ~/$REMOTE_ROOT/$OUT/runner.log" >&2; exit 1 ;;
        RUNNING*) echo "  $(echo "$state" | awk '{print $2}')/$((REPEATS * 2)) processes done" ;;
        *)       echo "  (poll returned nothing; tunnel may be down, retrying)" ;;
    esac
    sleep 120
done

# Price the run from the live catalog, so the report carries a rate that can be
# re-derived rather than a constant someone has to trust.
if [ -z "${CT_USD_PER_HOUR:-}" ]; then
    CT_USD_PER_HOUR=$(python3 "$(dirname "$0")/pricing.py" --region "$REGION" --json 2>/dev/null \
        | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['running_usd_per_hour']:.4f}\")" 2>/dev/null || echo "")
    export CT_USD_PER_HOUR
fi
[ -n "${CT_USD_PER_HOUR:-}" ] && echo "instance rate: \$$CT_USD_PER_HOUR/hour (list)"

say "collecting"
bash "$(dirname "$0")/collect.sh" "$OUT"
exit 0

say "report"
onvm "cd ~/$REMOTE_ROOT && .venv/bin/python cloud/report.py $OUT"

say "staging to $BUCKET"
onvm "cd ~/$REMOTE_ROOT && gcloud storage cp -r $OUT $BUCKET/$CASE/ --project '$PROJECT'"

echo
echo "run id: $STAMP  (case $CASE)"
echo "next:   CASE=$CASE cloud/fetch.sh $STAMP"
