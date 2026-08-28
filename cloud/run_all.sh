#!/usr/bin/env bash
# Whole pipeline, cold start to fetched results. Each step is separately re-runnable.
source "$(dirname "$0")/config.sh"
cd "$(dirname "$0")/.."

bash cloud/up.sh
bash cloud/sync.sh
bash cloud/hf_token.sh
bash cloud/train_probe.sh
bash cloud/run_overhead.sh
bash cloud/fetch.sh

say "remember to release the GPU"
echo "  cloud/down.sh            stop (keeps everything on disk)"
echo "  cloud/down.sh --delete   delete"
