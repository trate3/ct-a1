#!/usr/bin/env bash
# A second Apollo environment for Gemma-4 training.
#
# `.venv-apollo` pins transformers<5 because Apollo pins torch~=2.2 and transformers 5
# requires torch>=2.4. Gemma 4 cannot be loaded by transformers 4.x at all -- its
# tokenizer config uses a shape 4.x mis-parses ("'list' object has no attribute 'keys'")
# -- and gemma4-port.patch is itself written against transformers 5, which is why it
# renames num_logits_to_keep to logits_to_keep. So Gemma-4 gets its own environment
# rather than one env that is subtly wrong for one of the two tracks.
set -euo pipefail
cd "$(dirname "$0")/../third_party/deception-detection"
export PATH="$HOME/.local/bin:$PATH"

uv venv --allow-existing --python 3.11 .venv-apollo4
export VIRTUAL_ENV="$PWD/.venv-apollo4"

uv pip install -q -e . --no-deps
# Newer torch + transformers 5, and the same peripheral imports Apollo does at module
# scope. numpy is unpinned here: the <2 pin belongs to the torch 2.2 stack.
uv pip install -q torch torchvision "transformers>=5.10" accelerate tokenizers datasets \
    pandas einops jaxtyping pydantic scikit-learn wandb fire tqdm pyyaml matplotlib \
    python-dotenv peft plotly seaborn circuitsvis goodfire anthropic openai together

.venv-apollo4/bin/python - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
assert torch.cuda.is_available(), "apollo4 env cannot see the GPU"
assert int(transformers.__version__.split(".")[0]) >= 5, "need transformers >= 5 for Gemma 4"
PY
echo "-- apollo4 env ready"
