"""End-to-end check of the grafted ct/0.1 harness, with no models involved.

The conflict tests cover the monitor's parsing and the domains, but nothing exercises
`ct_core.harness.session.run_session`, which is the piece that was missing and has just
been grafted onto a ct/0.2 tree. This drives a full session -- commit, n sealed rounds,
certout, verification, and the offline separation -- on a scripted backend, so a
mismatch between the graft and our interfaces surfaces here rather than after a
14.56 GiB download.
"""
import json

from ct_core.harness.experiment import write_result  # noqa: F401  (import check)
from ct_core.harness.separation import separation
from ct_core.harness.session import run_session
from ct_core.roles.check_certificate import verify_certificate
from evals.conflict.conflict_components import (
    build_agent, build_comparator, build_environment, build_tap,
)
from util.model_backend import FakeBackend, GenOutput

VERDICT = json.dumps({"confidence_score": 3, "is_conflict_detected": True,
                      "explanation": "scripted"})


def backend():
    # The agent answers; the monitor must emit the JSON verdict the tap parses.
    return FakeBackend(
        on_generate=lambda text: (
            GenOutput("scripted reasoning", VERDICT) if "Auditor" in text or "JSON" in text
            else GenOutput("I should steer to the partner brand.",
                           "I recommend the partner option.")),
        name="scripted")


be = backend()
agent = build_agent(be, max_new_tokens=32)
tap = build_tap(be)
comparator = build_comparator(be)
env = build_environment(n=4, seed=7)
print(f"env rounds: {env.n()}")

cert = run_session(agent, tap, env)
reveal = cert.pop("_reveal")
print("certificate fields:", sorted(cert))
print(f"  m={cert['m']:.3f}  m_fp={cert['m_fp']:.3f}  h_c={cert['h_c'][:16]}...")
print(f"  h_T={cert['h_T'][:16]}...  m_ref={cert['m_ref'][:16]}...")

ok = verify_certificate(cert)
print(f"verify_certificate: {ok}")
assert ok, "grafted verifier rejected a certificate its own enclave produced"

comparison = separation(cert, reveal, comparator)
print("separation:", json.dumps(comparison, sort_keys=True))
# The offline quality path is what `--result-out` triggers, and it is where the ct/0.1
# call convention differs again: live objects rather than builders, and `comparator=`
# rather than `build_comparator=`.
from evals_runner.util import run_tap_quality  # noqa: E402

quality = run_tap_quality(
    build_agent(backend(), max_new_tokens=32),
    build_tap(backend()),
    build_environment(n=4, seed=7),
    seed=7,
    comparator=build_comparator(backend()),
)
print("offline quality keys:", sorted(quality))
print(f"  tap m={quality['tap']['m']:.3f} m_fp={quality['tap']['m_fp']:.3f}")
assert "comparator" in quality, "comparator arm missing from offline quality"

print("\nGRAFT OK: commit -> sealed rounds -> certout -> verify -> separation -> "
      "offline quality")
