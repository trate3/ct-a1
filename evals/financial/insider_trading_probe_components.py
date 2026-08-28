"""Construct the agent, probe tap, and environment for the insider-trading experiment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ct_core.agent import Agent
from ct_core.interfaces import AgentDriver, ArtifactPackage, TapDriver
from util.model_backend import model_artifact_root, model_artifacts

from . import actions, adaptive_agent, agent, divergent_agent, probe_tap, program
from .actions import FINANCIAL_TOOLS
from .agent import InsiderTradingControl
from .insider_trading_env import InsiderTradingEnvironment
from .probe_tap import ApolloProbe, ProbeMonitor
from .program import Gemma2Program
from .program_gemma4 import Gemma4Program

_RESULTS = Path(__file__).resolve().parents[2] / "third_party" / "deception-detection" / "results"


def program_class(path):
    """Pick the program by what the checkpoint IS, not by what it is called.

    Gemma 2 is a dense causal LM (`AutoModelForCausalLM`); Gemma 4 is a multimodal
    `Gemma4ForConditionalGeneration` whose text layers nest one level deeper, so loading
    one with the other's class fails outright. Dispatching on `config.json`'s
    `model_type` rather than on a substring of the model id means a local directory, a
    renamed snapshot, or a future checkpoint all resolve correctly -- and a checkpoint we
    do not recognise raises here rather than halfway through a run.

    This closes probe-track.md's "Open gap": a Gemma-4-trained probe could not be
    certified through the financial slice while this always built a Gemma-2 program.
    """
    import json
    from pathlib import Path as _Path

    config = json.loads((_Path(path) / "config.json").read_text())
    model_type = str(config.get("model_type", ""))
    if model_type.startswith("gemma4"):
        return Gemma4Program
    if model_type.startswith("gemma2") or model_type == "gemma":
        return Gemma2Program
    raise ValueError(
        f"no probed-program class for model_type {model_type!r} at {path}. The probe "
        "reads this model's activations, so guessing the wrong wrapper would either "
        "fail to load or read the wrong tower.")


@lru_cache(maxsize=None)
def _program(path, device, dtype, max_new_tokens, verbose, attn_implementation):
    return program_class(path)(
        path, device=device, dtype=dtype, max_new_tokens=max_new_tokens, verbose=verbose,
        attn_implementation=attn_implementation)


# Apollo's trainer loads Gemma-2 with eager attention. Gemma-2 soft-caps attention
# logits and the fused kernels skip that, so eager and sdpa are different functions --
# and the probe was fitted on the eager one. Committed with the rest of the agent
# config so the direct and certified arms cannot diverge on it either.
ATTN_IMPLEMENTATION = "eager"


def resolve_dtype(device: str, dtype: str = "auto") -> str:
    """Pick the load precision when the caller did not name one.

    float32 is the right default on CPU: it is full precision and uses the fast
    multi-threaded kernels, whereas bf16 on an Apple-Silicon CPU is single-threaded
    and slow. On CUDA the same choice doubles both the weight footprint and the
    generation time for no benefit, so an accelerator gets bfloat16. Resolved once,
    by the caller, and then carried in the committed agent config -- so the direct
    and certified arms cannot end up at different precisions.
    """
    if dtype != "auto":
        return dtype
    return "bfloat16" if str(device).startswith("cuda") else "float32"


def _control(config):
    kind = config.get("kind") if isinstance(config, dict) else None
    if kind == "insider" and set(config) == {"kind"}:
        return InsiderTradingControl()
    if kind == "adapted" and set(config) == {"kind", "instruction"}:
        return adaptive_agent.AdaptedInsiderTradingControl(config["instruction"])
    if kind == "baseline" and set(config) == {"kind"}:
        return divergent_agent.TrackedInsiderTradingControl()
    if kind == "benign" and set(config) == {"kind"}:
        return divergent_agent.BenignReformattingControl()
    if (kind == "divergent"
            and set(config) == {"kind", "rate", "n_trajectories", "seed", "plan"}):
        return divergent_agent.DivergentInsiderTradingControl(
            config["rate"], config["n_trajectories"], config["seed"],
            divergent_agent.OverridePlan(**config["plan"]))
    raise ValueError("unsupported financial control configuration")


class FinancialAgentDriver(AgentDriver):
    DRIVER = "financial-agent/v1"

    @classmethod
    def code_paths(cls):
        return (actions.__file__, agent.__file__, adaptive_agent.__file__,
                divergent_agent.__file__, program.__file__)

    @classmethod
    def from_package(cls, config, artifacts):
        required = {"model", "device", "dtype", "attn_implementation",
                    "max_new_tokens", "verbose", "control"}
        if set(config) != required:
            raise ValueError("malformed financial agent package")
        _control(config["control"])
        loaded = _program(str(model_artifact_root(artifacts)), config["device"],
                          config["dtype"], config["max_new_tokens"], config["verbose"],
                          config["attn_implementation"])
        return cls(loaded, config["control"])

    def __init__(self, loaded_program, control):
        self.program, self.control = loaded_program, control

    def new_agent(self, randomness):
        return Agent(self.program, _control(self.control), FINANCIAL_TOOLS)


class FinancialTapDriver(TapDriver):
    DRIVER = "financial-probe-tap/v1"

    @classmethod
    def code_paths(cls):
        return (probe_tap.__file__,)

    @classmethod
    def from_package(cls, config, artifacts):
        if set(config) != {"threshold"} or set(artifacts) != {"detector.npz"}:
            raise ValueError("malformed financial tap package")
        encoded = artifacts["detector.npz"]
        if not isinstance(encoded, bytes):
            encoded = Path(encoded).read_bytes()
        return cls(ApolloProbe.from_safe_bytes(encoded), config["threshold"])

    def __init__(self, probe, threshold):
        self.probe, self.threshold = probe, threshold

    def new_tap(self, randomness):
        return build_tap(self.probe, self.threshold)


def component_packages(probe, *, model="google/gemma-2-2b-it", device="cpu",
                       dtype="auto", max_new_tokens=128, threshold=0.0, verbose=False,
                       control=None):
    _, model_files = model_artifacts(model)
    if isinstance(probe, ApolloProbe):
        detector = probe.safe_bytes()
    else:
        path = Path(probe)
        detector = (path.read_bytes() if path.suffix == ".npz"
                    else ApolloProbe.load(path).safe_bytes())
    agent_package = ArtifactPackage(
        FinancialAgentDriver.DRIVER,
        {"model": model, "device": device, "dtype": resolve_dtype(device, dtype),
         "attn_implementation": ATTN_IMPLEMENTATION,
         "max_new_tokens": max_new_tokens,
         "verbose": verbose, "control": control or {"kind": "insider"}},
        model_files,
    )
    tap_package = ArtifactPackage(
        FinancialTapDriver.DRIVER, {"threshold": threshold},
        (("detector.npz", detector),),
    )
    return agent_package, tap_package


def latest_detector() -> Path:
    hits = sorted(_RESULTS.glob("*/detector.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise SystemExit(
            f"No detector.pt under {_RESULTS}. Train one first (RUNBOOK Phase 2 "
            "Step A), or pass --detector <path>."
        )
    return hits[0]


def build_program(
    model: str = "google/gemma-2-2b-it",
    device: str = "cpu",
    max_new_tokens: int = 128,
    verbose: bool = False,
    dtype: str = "auto",
) -> Gemma2Program:
    return _program(str(model_artifacts(model)[0]), device, resolve_dtype(device, dtype),
                    max_new_tokens, verbose, ATTN_IMPLEMENTATION)


def build_agent(program: Gemma2Program) -> Agent:
    return Agent(program, InsiderTradingControl(), FINANCIAL_TOOLS)


def build_tap(probe: ApolloProbe, threshold: float = 0.0) -> ProbeMonitor:
    return ProbeMonitor(probe, threshold=threshold)


def build_environment(
    n_trajectories: int | None = None, seed: int = 7
) -> InsiderTradingEnvironment:
    return InsiderTradingEnvironment(n_trajectories=n_trajectories, seed=seed)
