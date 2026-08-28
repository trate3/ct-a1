#!/usr/bin/env bash
# RUNBOOK Phase 2 Step A on the L4: train the Apollo linear probe, then confirm the
# artifact loads through our own non-executable path before anything depends on it.
#
# Apollo's trainer is config-file driven. `repe.yaml` supplies method: lr,
# train_data: repe_honesty__plain (612 dialogues), reg_coeff and detect_layers;
# anything after --config_file overrides a yaml field. --eval_data '[]' and
# --control_data '[]' are required for a model Apollo never pre-generated rollouts
# for: the config's on-policy datasets are named per-model and Experiment() raises
# `Dataset not found` otherwise. Emptying them skips Apollo's own eval suite, which
# the probe does not need -- and is why the run later dies in the report step with
# KeyError: 'alpaca__plain', after detector.pt is already on disk.
source "$(dirname "$0")/config.sh"

say "training the probe on $APOLLO_MODEL (detect_layers=$DETECT_LAYERS, reg_coeff=$REG_COEFF)"
cat <<MSG
  Apollo model name : $APOLLO_MODEL   -> loads google/gemma-2-${APOLLO_MODEL#gemma-}-it
  our checkpoint    : $MODEL
  activation batch  : $ACT_BATCH_SIZE dialogues per forward pass
  apollo env        : $APOLLO_VENV
  These must name the same weights: the probe's directions live in that model's
  activation space, and the tap is applied to exactly those activations.
MSG

onvm "cd ~/$REMOTE_ROOT/third_party/deception-detection && \
      set -o pipefail && \
      export CT_ACT_BATCH_SIZE=$ACT_BATCH_SIZE && \
      $APOLLO_VENV/bin/python -m deception_detection.scripts.experiment run \
        --config_file repe.yaml \
        --model_name '$APOLLO_MODEL' \
        --detect_layers '$DETECT_LAYERS' \
        --reg_coeff $REG_COEFF \
        --eval_data '[]' --control_data '[]' \
        2>&1 | tee ~/$REMOTE_ROOT/train_probe.log; \
      echo '--- trainer exited '\$?' (KeyError: alpaca__plain here is expected) ---'"

say "locating detector.pt"
# Apollo names its output directory after the model it trained on, so require the
# NEWEST detector to be one for THIS model. Without this the script happily validated a
# previous run's artifact when training failed -- `latest_detector()` just takes the
# newest detector.pt anywhere under results/, and a stale probe for a different
# checkpoint loads perfectly well. It only fails later, at scoring time, or not at all
# if the dimensions happen to match.
onvm "cd ~/$REMOTE_ROOT && newest=\$(ls -t third_party/deception-detection/results/*/detector.pt 2>/dev/null | head -1); \
      echo \"newest detector: \$newest\"; \
      case \"\$newest\" in \
        *__${APOLLO_MODEL}__*) echo \"matches requested model ${APOLLO_MODEL}\" ;; \
        *) echo \"MISMATCH: newest detector is not for ${APOLLO_MODEL}; training did not produce one\" >&2; exit 1 ;; \
      esac"
# Training counts as successful only once the artifact exists AND loads. The trainer
# is expected to crash after saving, so its exit status is not the test.
onvm "cd ~/$REMOTE_ROOT && ls -la \$(ls -t third_party/deception-detection/results/*/detector.pt | head -1)"

say "verifying the artifact through our own loader"
onvm "cd ~/$REMOTE_ROOT && .venv/bin/python - <<'PY'
from pathlib import Path
from evals.financial.probe_tap import ApolloProbe
from evals.financial.insider_trading_probe_components import latest_detector

path = latest_detector()
probe = ApolloProbe.load(path)
print(f'detector      : {path}')
print(f'layers        : {probe.layers}')
print(f'directions    : {probe.directions.shape}  (must match the model hidden size)')
print(f'normalize     : {probe.normalize}')
print(f'content sha256: {probe.content_hash()}')

# The certified path never unpickles: the workload is handed the inert NPZ form.
# Check that round-trip now, not during a timed run.
import numpy as np
again = ApolloProbe.from_safe_bytes(probe.safe_bytes())
assert again.content_hash() == probe.content_hash(), 'NPZ round-trip changed the probe'
assert np.isfinite(probe.directions).all(), 'probe directions contain inf/nan'
if probe.scaler_scale is not None:
    assert np.isfinite(probe.scaler_scale).all(), 'probe scaler contains inf/nan'
    assert (probe.scaler_scale != 0).all(), 'probe scaler has a zero scale'
print('npz round-trip: identical; all weights finite')
PY"

say "probe ready"
echo "next:  cloud/run_overhead.sh"
