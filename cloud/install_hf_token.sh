#!/usr/bin/env bash
# Runs ON the instance, from the repo root. Pulls the Hugging Face token out of Secret
# Manager using the instance's own service account and drops it where both
# environments expect it.
#
# The token never appears in the repo, in instance metadata, or on a command line, and
# whoever owns it is the only party that ever handled its plaintext.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PROJECT:=certified-taps}"
: "${HF_SECRET:=hf-token}"

if ! gcloud secrets versions access latest --secret="$HF_SECRET" --project="$PROJECT" \
        > /tmp/.hf_token_fetch 2>/tmp/.hf_token_err; then
    echo "  cannot read secret '$HF_SECRET' in project '$PROJECT':" >&2
    sed 's/^/    /' /tmp/.hf_token_err >&2
    rm -f /tmp/.hf_token_fetch /tmp/.hf_token_err
    exit 1
fi

umask 077
mv /tmp/.hf_token_fetch .hf_token
# Apollo's loader calls load_dotenv(), so its env needs the token as a .env line.
printf 'HF_TOKEN=%s\n' "$(cat .hf_token)" > third_party/deception-detection/.env
chmod 600 .hf_token third_party/deception-detection/.env
umask 022
rm -f /tmp/.hf_token_err

HF_TOKEN="$(cat .hf_token)" .venv/bin/python - <<'PY'
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

model = "google/gemma-2-2b-it"
token = os.environ["HF_TOKEN"]
info = HfApi().model_info(model, token=token)
print(f"  metadata visible: {info.id} @ {info.sha}  (gated={info.gated})")

# Metadata access is NOT the check that matters. A gated repo returns model_info to
# any authenticated caller while refusing file downloads with 403 until the license is
# granted AND the token carries gated-repo read scope -- so verifying metadata alone
# reports success and the trainer still fails on the first download. Fetch a real file.
try:
    hf_hub_download(model, filename="config.json", token=token)
except Exception as error:
    code = getattr(getattr(error, "response", None), "status_code", None)
    print(f"  DOWNLOAD REFUSED (http {code}): {type(error).__name__}", file=sys.stderr)
    if code == 403:
        who = HfApi().whoami(token=token)
        auth = (who.get("auth") or {}).get("accessToken") or {}
        print(f"  account {who.get('name')}, token role {auth.get('role')}",
              file=sys.stderr)
        if auth.get("fineGrained"):
            print("  this is a FINE-GRAINED token: it must carry 'Read access to the",
                  file=sys.stderr)
            print("  contents of all public gated repos you can access', or name this",
                  file=sys.stderr)
            print("  repo explicitly. A classic Read token avoids the whole question.",
                  file=sys.stderr)
        print(f"  also confirm the licence is granted: https://huggingface.co/{model}",
              file=sys.stderr)
    raise SystemExit(1)
print("  gated download works: the trainer and the evaluation can both fetch weights")
PY
