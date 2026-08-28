#!/usr/bin/env bash
# Stop or delete the instance. Billing for the L4 continues while it is RUNNING.
#   down.sh          stop  -- keeps the disk, the venvs, the HF cache, and the probe
#   down.sh --delete delete -- releases the disk and the GPU quota
source "$(dirname "$0")/config.sh"

if [ "${1:-}" = "--delete" ]; then
    say "deleting $INSTANCE (results in $BUCKET are unaffected)"
    gc compute instances delete "$INSTANCE" --zone "$ZONE" --quiet
else
    say "stopping $INSTANCE"
    gc compute instances stop "$INSTANCE" --zone "$ZONE"
    echo "the boot disk still bills (~\$0.10/GB-month for ${BOOT_DISK_SIZE}); "
    echo "use --delete to release it, or cloud/up.sh to start it again."
fi
