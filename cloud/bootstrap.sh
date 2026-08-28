#!/usr/bin/env bash
# Runs ON the instance, from the repo root. Idempotent.
#
# Two environments, kept apart exactly as RUNBOOK Phase 2 requires:
#   .venv         -- this project. Current torch + transformers; runs the evaluation.
#   .venv-apollo  -- Apollo's trainer only. torch~=2.2 / numpy<2, which would break
#                    the main stack. Lives inside the submodule.
# Both are Python 3.11: the only version that satisfies this project's >=3.11 and
# Apollo's torch~=2.2 pin, which has no wheels for 3.12+.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
: "${PYTHON_VERSION:=3.11}"

step() { printf '\n\033[1m-- %s\033[0m\n' "$*"; }

step "host"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | awk 'NR==2 {print "  host RAM: " $2 " GB"}'
df -h --output=avail / | tail -1 | awk '{print "  disk free: " $1}'

step "uv + Python $PYTHON_VERSION"
# uv rather than the distro's python3.10 plus a PPA: it supplies a standalone 3.11
# without adding an apt source, and it resolves Apollo's dependency set in seconds
# where pip backtracks for minutes (the RUNBOOK notes that wait).
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install "$PYTHON_VERSION"
uv --version

step "submodules"
for spec in "treecut-math|https://github.com/j-bagel/treecut-math" \
            "deception-detection|https://github.com/ApolloResearch/deception-detection"; do
    name=${spec%%|*}; url=${spec##*|}
    if [ -d "third_party/$name/.git" ]; then
        echo "  third_party/$name already cloned"
    else
        rm -rf "third_party/$name"
        git clone --depth 1 "$url" "third_party/$name"
    fi
done

step "main environment (.venv)"
uv venv --allow-existing --python "$PYTHON_VERSION" .venv
VIRTUAL_ENV="$ROOT/.venv" uv pip install -q -e ".[dev,model,plots]"
.venv/bin/python -c "
import torch, transformers
print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}  '
      f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
print(f'  transformers {transformers.__version__}')
"

step "Apollo environment (.venv-apollo)"
# --no-deps then an explicit list: plain `pip install -e .` drags in sae-lens,
# inspect-ai, together and streamlit, none of which the linear-probe path uses.
cd third_party/deception-detection
uv venv --allow-existing --python "$PYTHON_VERSION" .venv-apollo
export VIRTUAL_ENV="$PWD/.venv-apollo"
uv pip install -q -e . --no-deps
uv pip install -q "torch~=2.2.0" "torchvision~=0.17.0" "numpy<2" accelerate \
    "transformers>=4.45.2,<5" tokenizers datasets pandas einops jaxtyping pydantic \
    scikit-learn wandb fire tqdm pyyaml matplotlib python-dotenv \
    peft plotly seaborn circuitsvis goodfire anthropic openai together
# The last line is import-satisfaction only. Apollo imports its peripheral
# integrations at module scope -- LoRA (peft), plotting (plotly, seaborn,
# circuitsvis), and the rollout API clients (goodfire, anthropic, openai, together) --
# so importing the linear-probe trainer pulls all of them in even though the `lr` path
# calls none. These seven are exactly the set reachable from
# deception_detection.scripts.experiment; sae_lens, inspect_ai and streamlit are NOT
# on that chain, which is why `pip install -e . --no-deps` above is still the right
# call rather than installing Apollo's full dependency set.
.venv-apollo/bin/python -c "
import torch
print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'Apollo env cannot see the GPU'
"
unset VIRTUAL_ENV
cd "$ROOT"

step "Apollo CUDA loader patch"
python3 cloud/patch_apollo_cuda.py

step "Hugging Face token"
if bash cloud/install_hf_token.sh; then
    :
else
    echo "  token not installed yet -- run cloud/hf_token.sh after creating the secret"
fi

step "server extra (llama.cpp client)"
# `requests` + PySocks, for talking to a llama.cpp server. Cheap, and RQ2 needs it.
VIRTUAL_ENV="$ROOT/.venv" uv pip install -q -e ".[server]"

step "protocol tests (GPU-free)"
.venv/bin/python -m pytest -q

step "bootstrap complete"
