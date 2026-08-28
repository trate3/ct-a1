"""Probe scoring math (numpy; no torch, no model, no Apollo data)."""

from __future__ import annotations

import pytest

from ct_core.interfaces import MonitorInput


def test_probe_monitor_scoring_and_digest():
    np = pytest.importorskip("numpy")
    from evals.financial.probe_tap import ApolloProbe, ProbeMonitor

    probe = ApolloProbe(
        layers=[0],
        directions=np.array([[1.0, 0.0]]),  # [n_layers, emb]
        scaler_mean=None,
        scaler_scale=None,
        normalize=False,
    )
    mon = ProbeMonitor(probe, threshold=1.0)
    assert mon.needs_activations and mon.probe_layers == [0]

    # acts [T, L, E]: two tokens, one layer; raw score = mean_t(dir . act) = mean(2,4) = 3.
    acts = np.array([[[2.0, 9.0]], [[4.0, 9.0]]])
    pred = mon.predict(MonitorInput(x="q", answer="a", activations=acts))
    assert pred.score == pytest.approx(3.0)
    assert pred.z_hat == 1  # 3.0 > threshold 1.0

    # Evaluation identity includes the actual weights and threshold.
    assert "weights=" in mon.digest() and "thr=1.0" in mon.digest()
    assert ProbeMonitor(probe, threshold=5.0).predict(
        MonitorInput(x="q", answer="a", activations=acts)
    ).z_hat == 0


def test_probe_monitor_requires_activations():
    np = pytest.importorskip("numpy")
    from evals.financial.probe_tap import ApolloProbe, ProbeMonitor

    mon = ProbeMonitor(ApolloProbe([0], np.array([[1.0]]), None, None, False))
    with pytest.raises(ValueError, match="activations"):
        mon.predict(MonitorInput(x="q", answer="a"))


def test_load_real_apollo_probe_if_present():
    # Guards the loader format: Apollo saves the linear probe with pickle.dump, not
    # torch.save, so ApolloProbe.load must pickle.load. Skips when the submodule
    # (or torch, needed to reconstruct the pickled tensors) is unavailable.
    import pathlib

    pytest.importorskip("torch")
    from evals.financial.probe_tap import ApolloProbe

    p = (pathlib.Path(__file__).resolve().parents[2] / "third_party" / "deception-detection"
         / "example_results" / "roleplaying" / "detector.pt")
    if not p.exists():
        pytest.skip("Apollo submodule example probe not checked out")

    probe = ApolloProbe.load(p)
    assert probe.directions.ndim == 2                          # [n_layers, emb]
    assert probe.directions.shape[0] == len(probe.layers)


def test_program_uses_generation_trace_without_a_second_model_run():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from evals.financial.program import Gemma2Program

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return Batch(input_ids=torch.tensor([[10, 11]]))

        def decode(self, tokens, skip_special_tokens=True):
            return "answer"

    class Model:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs["return_dict_in_generate"]
            assert kwargs["output_hidden_states"]
            return SimpleNamespace(
                sequences=torch.tensor([[10, 11, 12, 13]]),
                hidden_states=(
                    (torch.zeros(1, 2, 2), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])),
                    (torch.zeros(1, 1, 2), torch.tensor([[[5.0, 6.0]]])),
                ),
            )

        def __call__(self, *args, **kwargs):
            pytest.fail("tap activations were recomputed in a second model run")

    program = Gemma2Program.__new__(Gemma2Program)
    program._torch, program._tok, program._model = torch, Tokenizer(), Model()
    program._device, program._max_new_tokens, program._verbose = "cpu", 2, False
    generation = program.generate("query", capture_layers=[1])

    assert generation.answer == "answer"
    assert generation.activations.tolist() == [[[3.0, 4.0]], [[5.0, 6.0]]]
    assert program._model.calls == 1
