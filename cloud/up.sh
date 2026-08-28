#!/usr/bin/env bash
# Create the L4 instance and the results bucket. Idempotent: re-running reuses both.
source "$(dirname "$0")/config.sh"

say "preflight"
gc services enable compute.googleapis.com secretmanager.googleapis.com storage.googleapis.com \
    iap.googleapis.com >/dev/null
quota=$(gc compute regions describe "$REGION" \
    --format="value(quotas.filter(\"metric:NVIDIA_L4_GPUS\").extract(limit))" 2>/dev/null || true)
echo "L4 quota in $REGION: ${quota:-unknown}"

if ! gc secrets describe "$HF_SECRET" >/dev/null 2>&1; then
    cat <<MSG

  The probed checkpoint $MODEL is gated (HTTP 401 unauthenticated), so both the
  probe trainer and the evaluation need a Hugging Face read token that has been
  granted access to it.

  Store it yourself -- it should not pass through anyone else's hands, and this
  keeps it out of the repo and out of instance metadata:

    printf %s "hf_YOUR_TOKEN" | gcloud secrets create $HF_SECRET \\
        --project=$PROJECT --data-file=- --replication-policy=automatic

  Provisioning continues without it -- everything except the model download can
  be installed first. Run cloud/hf_token.sh once the secret exists.

MSG
else
    echo "HF token secret: $HF_SECRET present"
fi

say "results bucket"
gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 \
    || gcloud storage buckets create "$BUCKET" --project "$PROJECT" \
         --location "$REGION" --uniform-bucket-level-access

say "instance"
if gc compute instances describe "$INSTANCE" --zone "$ZONE" >/dev/null 2>&1; then
    echo "$INSTANCE already exists"
    state=$(gc compute instances describe "$INSTANCE" --zone "$ZONE" --format="value(status)")
    [ "$state" = "RUNNING" ] || gc compute instances start "$INSTANCE" --zone "$ZONE"
else
    created=""
    index=0
    for candidate in $ZONE_CANDIDATES; do
        index=$((index + 1))
        candidate_region="${candidate%-*}"
        candidate_subnet="$SUBNET"
        if [ "$candidate_region" != "$REGION" ]; then
            candidate_subnet="${SUBNET}-${candidate_region}"
            if ! gc compute networks subnets describe "$candidate_subnet" \
                    --region "$candidate_region" >/dev/null 2>&1; then
                echo "creating subnet $candidate_subnet in $candidate_region"
                gc compute networks subnets create "$candidate_subnet" \
                    --network "$NETWORK" --region "$candidate_region" \
                    --range "10.$((index + 10)).0.0/24" >/dev/null
            fi
        fi
        echo "trying $candidate ($candidate_subnet)"
        # A g2 stockout surfaces here, not in the quota check, so a failed create is
        # expected rather than fatal; only exhausting every candidate is fatal.
        if gc compute instances create "$INSTANCE" \
            --zone "$candidate" \
            --machine-type "$MACHINE_TYPE" \
            --accelerator "$ACCELERATOR" \
            --maintenance-policy TERMINATE \
            --provisioning-model "$PROVISIONING_MODEL" \
            --image-family "$IMAGE_FAMILY" \
            --image-project "$IMAGE_PROJECT" \
            --boot-disk-size "$BOOT_DISK_SIZE" \
            --boot-disk-type "$BOOT_DISK_TYPE" \
            --network "$NETWORK" \
            --subnet "$candidate_subnet" \
            --scopes cloud-platform \
            --metadata install-nvidia-driver=True \
            --labels purpose=certified-taps-overhead 2>&1 | tail -5
        then
            created="$candidate"
            break
        fi
        echo "  $candidate has no $MACHINE_TYPE + L4 capacity right now"
    done
    if [ -z "$created" ]; then
        cat >&2 <<MSG
no candidate zone has L4 capacity for $MACHINE_TYPE right now.
Options, roughly in order of preference:
  * retry in a few minutes -- g2 stockouts are usually short-lived;
  * MACHINE_TYPE=g2-standard-4 cloud/up.sh   (a different, often freer, pool;
    16 GB host RAM is enough for this run, the L4 is the same 24 GB card);
  * PROVISIONING_MODEL=SPOT cloud/up.sh      (a separate capacity pool, but a
    preemption mid-arm would discard half of a paired measurement);
  * add zones to ZONE_CANDIDATES -- L4 quota is 1 in nearly every region.
MSG
        exit 1
    fi
    ZONE="$created"
    printf '%s' "$ZONE" > "$(dirname "$0")/.zone"
    echo "pinned zone: $ZONE"
fi

say "waiting for ssh"
for _ in $(seq 1 40); do
    onvm true >/dev/null 2>&1 && break
    sleep 15
done
onvm true || { echo "ssh never came up; check the serial console" >&2; exit 1; }

say "waiting for the GPU"
onvm 'for i in $(seq 1 40); do nvidia-smi >/dev/null 2>&1 && break; sleep 15; done; nvidia-smi'

say "ready"
echo "next:  cloud/sync.sh"
