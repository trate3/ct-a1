#!/usr/bin/env bash
# Upload this working copy to the instance and provision both environments.
# Re-runnable: it replaces the code and leaves the venvs, HF cache, and any trained
# probe in place, so iterating on the eval does not re-download or re-train anything.
source "$(dirname "$0")/config.sh"
here="$(cd "$(dirname "$0")/.." && pwd)"

say "packing the working copy"
# third_party/ is cloned on the instance instead of uploaded: it is ~480 MB and comes
# down far faster from GitHub than up from a laptop. .venv is host-specific.
COPYFILE_DISABLE=1 tar czf /tmp/certified-taps-src.tgz -C "$here" \
    --exclude='.venv' --exclude='.git' \
    --exclude='third_party/treecut-math' \
    --exclude='third_party/deception-detection' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    --exclude='._*' \
    --exclude='evals_results/*/' \
    .
ls -lh /tmp/certified-taps-src.tgz | awk '{print "  " $5 " " $9}'

say "uploading"
gc compute scp /tmp/certified-taps-src.tgz "$INSTANCE":/tmp/ --zone "$ZONE" --tunnel-through-iap

say "provisioning (first run installs uv, Python $PYTHON_VERSION, torch; expect ~10 min)"
onvm "mkdir -p ~/$REMOTE_ROOT && tar xzf /tmp/certified-taps-src.tgz -C ~/$REMOTE_ROOT"
onvm "cd ~/$REMOTE_ROOT && \
      PYTHON_VERSION='$PYTHON_VERSION' PROJECT='$PROJECT' HF_SECRET='$HF_SECRET' \
      bash cloud/bootstrap.sh"

say "done"
echo "next:  cloud/train_probe.sh   then   cloud/run_overhead.sh"
