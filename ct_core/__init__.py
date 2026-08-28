"""ct_core — the Certified Taps protocol engine (model-agnostic, reusable).

The engine is defined purely against the interfaces in `interfaces.py`; concrete
programs / monitors / environments live with their evaluation purpose (`evals/`)
or as shared utilities (`util/`).

  interfaces.py   the seams: ProgramRunner, Monitor, TestEnvironment
  certification.py execute E, certify, and verify
  protocol/       sealing, digests, and attestation
"""
