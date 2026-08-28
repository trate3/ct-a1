#!/usr/bin/env python3
"""Exercise the financial probe path on the GPU without the gated checkpoint.

Everything in the real run except two things -- the gated Gemma-2 download and the
trained probe -- can be checked before either exists, and this does that:

  * a Gemma2ForCausalLM loads on CUDA with eager attention under this transformers;
  * generation still exposes one hidden-state tuple per emitted token, which is the
    contract `Gemma2Program.generate` depends on to hand the tap a per-token view;
  * the probe tap reads those activations, and the NPZ package crosses the trusted
    boundary intact;
  * the direct and certified arms agree on (m, m_fp) when the model runs on a device;
  * the new accelerator telemetry actually populates on CUDA.

It uses `hf-internal-testing/tiny-random-Gemma2ForCausalLM` -- Hugging Face's own
ungated test checkpoint for this architecture -- with a random-direction probe sized
to that model. So it says nothing about probe quality or about timings, and it is not
a measurement. It is the difference between finding a transformers incompatibility now
and finding it after paying for a probe training run.
"""

from __future__ import annotations

import sys

import numpy as np

from evals.financial.insider_trading_probe_components import (
    ATTN_IMPLEMENTATION,
    FinancialAgentDriver,
    FinancialTapDriver,
    build_environment,
    build_program,
    component_packages,
)
from evals.financial.probe_tap import ApolloProbe
from evals_runner.util import ExperimentConfig, detect_accelerator, run_comparison

MODEL = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
N_TRAJECTORIES = 2
SEED = 7

# The tiny test checkpoint ships no tokenizer.chat_template, so apply_chat_template
# raises on it. google/gemma-2-2b-it does ship one and needs nothing here. A plain
# stand-in is enough for a wiring check: what is being exercised is the generation and
# activation-capture path, not prompt fidelity.
STANDIN_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}{% if add_generation_prompt %}model: {% endif %}"
)


def synthetic_probe(model: str) -> ApolloProbe:
    """A probe shaped to the checkpoint, so the scoring math runs on real activations."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    hidden = config.hidden_size
    # hidden_states has num_hidden_layers + 1 entries; stay inside it.
    layer = max(1, config.num_hidden_layers // 2)
    rng = np.random.default_rng(SEED)
    print(f"  checkpoint: {model}")
    print(f"  layers={config.num_hidden_layers} hidden={hidden} -> probing layer {layer}")
    return ApolloProbe(
        layers=[layer],
        directions=rng.normal(size=(1, hidden)).astype(np.float32),
        scaler_mean=np.zeros((1, hidden), dtype=np.float32),
        scaler_scale=np.ones((1, hidden), dtype=np.float32),
        normalize=True,
    )


def main() -> int:
    accelerator = detect_accelerator()
    print(f"accelerator: {accelerator.describe()}")
    if not accelerator.available:
        print("no CUDA device -- this check is only meaningful on the GPU instance",
              file=sys.stderr)
        return 1

    print(f"attn implementation: {ATTN_IMPLEMENTATION}")
    probe = synthetic_probe(MODEL)

    # Built eagerly so the chat template can be installed before the drivers ask for
    # it: _program is lru_cached on (path, device, dtype, max_new_tokens, verbose,
    # attn), so the driver inside run_comparison receives this same instance -- which
    # is also what keeps the real run from loading the weights twice.
    program = build_program(MODEL, "cuda", 8, verbose=True)
    if getattr(program._tok, "chat_template", None) is None:
        program._tok.chat_template = STANDIN_CHAT_TEMPLATE
        print("  installed a stand-in chat template (test checkpoint ships none)")

    packages = component_packages(
        probe, model=MODEL, device="cuda", dtype="auto", max_new_tokens=8,
        threshold=0.0, verbose=True)
    print(f"  committed agent config: {packages[0].config}")

    result = run_comparison(
        ExperimentConfig(
            "financial-gpu-smoke", SEED, N_TRAJECTORIES, "apollo-insider-trading",
            parameters={"model": MODEL, "note": "synthetic probe; wiring check only"}),
        build_agent_package=lambda: packages[0],
        build_tap_package=lambda: packages[1],
        agent_drivers=(FinancialAgentDriver,),
        tap_drivers=(FinancialTapDriver,),
        build_environment=lambda: build_environment(N_TRAJECTORIES, SEED),
    )

    # run_comparison already raises if the arms disagree; assert the GPU-specific
    # columns are populated, since a silently-null accelerator block would mean the
    # real run records nothing about the device it ran on.
    device_metrics = result["direct"]["metrics"]["accelerator"]
    assert device_metrics, "accelerator telemetry is empty on a CUDA device"
    assert device_metrics["device_peak_allocated_bytes"] > 0, "no device memory recorded"
    assert result["runtime"]["accelerator"]["name"], "device name not recorded"
    assert result["runtime"]["instance"], "GCE instance metadata not recorded"

    peak = device_metrics["device_peak_allocated_bytes"] / 1024**2
    print(f"\ndirect measurement : {result['direct']['measurement']}")
    print(f"released           : m={result['certified']['certificate']['m']:.3f} "
          f"m_fp={result['certified']['certificate']['m_fp']:.3f}")
    print(f"device peak        : {peak:.1f} MiB")
    print(f"wire bytes         : {result['overhead']['wire_bytes']:,}")
    print(f"instance           : {result['runtime']['instance']['machine_type']} "
          f"{result['runtime']['instance']['zone']}")
    print("\nGPU path OK: eager Gemma-2 load, per-token activations, probe tap, "
          "certified == direct, telemetry populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
