#!/usr/bin/env bash
# Install the Hugging Face token on the instance, after its secret has been created.
source "$(dirname "$0")/config.sh"

if ! gc secrets describe "$HF_SECRET" >/dev/null 2>&1; then
    cat <<MSG

  Secret '$HF_SECRET' does not exist in project '$PROJECT' yet.

  $MODEL is gated (HTTP 401 unauthenticated), so both the probe trainer and the
  evaluation need a Hugging Face read token that has been granted access to it:

    1. accept the license at https://huggingface.co/$MODEL
    2. create a Read token at https://huggingface.co/settings/tokens
    3. store it -- this command is the only place its plaintext appears:

       printf %s "hf_YOUR_TOKEN" | gcloud secrets create $HF_SECRET \\
           --project=$PROJECT --data-file=- --replication-policy=automatic

    4. re-run this script.

MSG
    exit 1
fi

service_account=$(gc compute instances describe "$INSTANCE" --zone "$ZONE" \
    --format="value(serviceAccounts[0].email)")
say "granting $service_account read access to $HF_SECRET"
gc secrets add-iam-policy-binding "$HF_SECRET" \
    --member "serviceAccount:${service_account}" \
    --role roles/secretmanager.secretAccessor >/dev/null

say "installing on the instance"
onvm "cd ~/$REMOTE_ROOT && PROJECT='$PROJECT' HF_SECRET='$HF_SECRET' bash cloud/install_hf_token.sh"
