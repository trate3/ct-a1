"""Basic mocked evaluation: the Certified Taps protocol end to end with mock models.

Purpose: exercise the protocol (commit -> sealed testing -> certify -> verify), the
integrity check (tamper demo), and the gaming-resistance story (gaming demo) with
zero GPU. `mocks.py` holds the mock program + taps,
`mocked_protocol_components.py` constructs them, and `evals_runner` contains its
command-line orchestration.
"""

import pathlib

FIXTURE = pathlib.Path(__file__).parent / "data" / "treecut_sample.jsonl"
