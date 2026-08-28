#!/usr/bin/env bash
# Bring a run's results back to this working copy. With no argument, fetches the
# newest run on the instance.
source "$(dirname "$0")/config.sh"
: "${CASE:=financial}"
stamp="${1:-}"

if [ -z "$stamp" ]; then
    stamp=$(onvm "ls -1 ~/$REMOTE_ROOT/evals_results/$CASE | grep '^cloud_l4_' | sort | tail -1" \
            | tr -d '\r' | sed 's/^cloud_l4_//')
    [ -n "$stamp" ] || { echo "no runs found on $INSTANCE" >&2; exit 1; }
fi
run="cloud_l4_${stamp}"
local_dir="evals_results/$CASE"
mkdir -p "$local_dir"

say "fetching $run"
gc compute scp --recurse --zone "$ZONE" --tunnel-through-iap \
    "$INSTANCE":"~/$REMOTE_ROOT/evals_results/$CASE/$run" "$local_dir/"

say "contents"
ls -la "$local_dir/$run"
echo
echo "GCS copy: $BUCKET/$CASE/$run"
