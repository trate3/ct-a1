#!/usr/bin/env bash
# Shared settings for the GCP port of the certification-overhead measurement.
# Every script sources this file; override any value by exporting it first.

set -euo pipefail

: "${PROJECT:=certified-taps}"
# L4 capacity moves around: a zone that is out of g2 stock returns
# ZONE_RESOURCE_POOL_EXHAUSTED at create time, not at quota time. up.sh walks these
# in order and pins whichever one accepted the instance to cloud/.zone, which every
# later script then reads -- so nothing has to be told the zone twice. The subnet is
# regional, so any zone in REGION works without touching the network.
: "${ZONE_CANDIDATES:=us-east1-c us-east1-d us-east1-b us-east4-a us-east4-c us-central1-c us-central1-a us-central1-b us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c northamerica-northeast1-b northamerica-northeast1-c}"
_zone_state="$(dirname "${BASH_SOURCE[0]:-$0}")/.zone"
if [ -z "${ZONE:-}" ] && [ -f "$_zone_state" ]; then
    ZONE="$(cat "$_zone_state")"
fi
: "${ZONE:=${ZONE_CANDIDATES%% *}}"
: "${REGION:=${ZONE%-*}}"
: "${INSTANCE:=ct-probe-l4}"

# g2-standard-8 is the smallest L4 shape with enough host RAM (32 GB) to unpickle
# the probe and hold the HF snapshot cache while the model is resident on the GPU.
# The project's quota is GPUS_ALL_REGIONS=1, so one L4 is the whole budget; A100 and
# H100 quota are both 0 and would need a quota increase before they can be selected.
: "${MACHINE_TYPE:=g2-standard-8}"
: "${ACCELERATOR:=type=nvidia-l4,count=1}"

# Deep Learning VM: NVIDIA driver 580 + CUDA 12.9 already in the image, so no
# driver build on first boot. Ubuntu 22.04 ships Python 3.10; uv supplies 3.11,
# which is the one version that satisfies both this project (>=3.11) and Apollo's
# torch~=2.2 pin (no 3.12+ wheels).
: "${IMAGE_FAMILY:=common-cu129-ubuntu-2204-nvidia-580}"
: "${IMAGE_PROJECT:=deeplearning-platform-release}"
: "${BOOT_DISK_SIZE:=200GB}"
: "${BOOT_DISK_TYPE:=pd-balanced}"
: "${PYTHON_VERSION:=3.11}"

# STANDARD, not SPOT: this is a wall-clock and CPU-time measurement, and a
# preemption mid-arm would silently discard one half of a paired comparison.
: "${PROVISIONING_MODEL:=STANDARD}"

# The project has no default network: ct-taps-net/ct-taps-subnet already exist, with
# an IAP-range-only SSH rule. The subnet has privateIpGoogleAccess=False, so an
# instance with no external IP could reach neither PyPI/GitHub/HF nor the Google APIs;
# an ephemeral external IP is the cheaper of the two fixes (the other is Cloud NAT).
# ct-taps-net is CUSTOM mode with a single us-central1 subnet, so a zone outside that
# region needs its own subnet; up.sh creates one on demand, named per region with a
# deterministic /24 so repeat runs never collide.
# NETWORK_TAGS is deliberately empty: the pre-existing allow-ct-taps-workload rule
# opens tcp:7401 to 0.0.0.0/0 for target tag ct-taps-workload, and this instance has
# no reason to carry that tag. Nothing here listens on a port.
: "${NETWORK:=ct-taps-net}"
: "${SUBNET:=ct-taps-subnet}"

: "${BUCKET:=gs://certified-taps-eval-results}"
: "${REMOTE_ROOT:=certified-taps}"   # relative to the instance home dir
: "${HF_SECRET:=hf-token}"

# The probed program and the probe must name the same weights: APOLLO_MODEL is the
# identifier Apollo's training config accepts, MODEL is the matching HF checkpoint.
: "${MODEL:=google/gemma-2-2b-it}"
: "${APOLLO_MODEL:=gemma-2b}"

# repe.yaml's own value, kept so the cloud run reproduces the local process rather
# than quietly changing the tap. gemma-2-2b has 26 layers, so 22 is far deeper in
# proportion than Apollo's 22-of-80; sweeping it is a probe-quality question,
# separate from this overhead measurement.
: "${DETECT_LAYERS:=[22]}"
: "${REG_COEFF:=10}"
# Dialogues per forward pass when extracting training activations. Apollo's default of
# 12 assumes a cluster card; 9B in bfloat16 is 18.5 GB of weights on a 24 GB L4 before
# any activations exist, so the batch has to come down to fit alongside them.
: "${ACT_BATCH_SIZE:=2}"

# Which Apollo environment to train in. `.venv-apollo` is pinned transformers<5 for the
# Gemma-2 track; `.venv-apollo4` carries transformers>=5 for Gemma 4, which the Gemma-2
# pins cannot express at the same time (transformers 5 needs torch>=2.4, Apollo pins
# torch~=2.2). Separate environments rather than one that is wrong for one of them.
: "${APOLLO_VENV:=.venv-apollo}"

: "${N_TRAJECTORIES:=26}"   # the full scenario set; 26 YAMLs fit the 256 KB fixed E
: "${MAX_NEW_TOKENS:=128}"
: "${SEED:=7}"
: "${THRESHOLD:=0.0}"
: "${DEVICE:=cuda}"
: "${DTYPE:=auto}"          # auto -> bfloat16 on cuda

gc()  { gcloud --project "$PROJECT" "$@"; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# Run a command on the instance. Quoted as a single string so the remote shell,
# not the local one, does the word splitting.
onvm() { gc compute ssh "$INSTANCE" --zone "$ZONE" --tunnel-through-iap --command "$*"; }
