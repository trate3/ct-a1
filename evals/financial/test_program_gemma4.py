"""Layer-coordinate tests for the Gemma-4 program.

These cover the part that fails silently. A wrong load class raises immediately; a
wrong layer index does not -- it yields a probe score that looks reasonable and means
something else. All of this runs without weights.
"""

from __future__ import annotations

import pytest

from evals.financial.program_gemma4 import (
    full_attention_layers,
    text_layer_count,
    validate_layers,
)


class _Text:
    def __init__(self, layers, kinds):
        self.num_hidden_layers = layers
        self.layer_types = kinds


class _Config:
    def __init__(self, layers, kinds):
        self.text_config = _Text(layers, kinds)


# google/gemma-4-E4B-it, read from its published config.json.
E4B_KINDS = ["sliding_attention"] * 42
for _block in (5, 11, 17, 23, 29, 35, 41):
    E4B_KINDS[_block] = "full_attention"
E4B = _Config(42, E4B_KINDS)


def test_layer_count_reads_the_nested_text_config():
    assert text_layer_count(E4B) == 42


def test_full_attention_layers_are_offset_into_hidden_states():
    # layer_types[i] describes block i; the stream after block i is hidden_states[i+1].
    assert full_attention_layers(E4B) == [6, 12, 18, 24, 30, 36, 42]


def test_validate_accepts_embeddings_and_final_block():
    validate_layers([0, 42], E4B)


@pytest.mark.parametrize("layer", [-1, 43, 60])
def test_validate_rejects_indices_the_checkpoint_cannot_supply(layer):
    with pytest.raises(ValueError, match="outside this checkpoint"):
        validate_layers([layer], E4B)


def test_a_gemma2_layer_index_is_not_silently_valid_on_a_smaller_model():
    """A probe fitted on gemma-2-9b (42 layers) names layers a 26-layer model lacks."""
    gemma2_2b = _Config(26, [])
    with pytest.raises(ValueError):
        validate_layers([42], gemma2_2b)
    # ...and repe.yaml's default IS in range for 2b, which is why the mistake is quiet.
    validate_layers([22], gemma2_2b)


def test_no_layer_types_means_no_sliding_claim():
    """Gemma-2 has no layer_types; we must not invent a full-attention set for it."""
    assert full_attention_layers(_Config(26, [])) == []


def test_program_class_dispatches_on_model_type(tmp_path):
    """Loading Gemma 4 with Gemma 2's class fails; loading Gemma 2 with Gemma 4's reads
    a tower that is not there. Dispatch must key on the checkpoint, not the name."""
    import json

    from evals.financial.insider_trading_probe_components import program_class
    from evals.financial.program import Gemma2Program
    from evals.financial.program_gemma4 import Gemma4Program

    def checkpoint(model_type):
        directory = tmp_path / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
        return directory

    assert program_class(checkpoint("gemma4")) is Gemma4Program
    assert program_class(checkpoint("gemma4_text")) is Gemma4Program
    assert program_class(checkpoint("gemma2")) is Gemma2Program
    assert program_class(checkpoint("gemma")) is Gemma2Program


def test_program_class_refuses_an_unknown_family(tmp_path):
    import json

    from evals.financial.insider_trading_probe_components import program_class

    directory = tmp_path / "mystery"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="no probed-program class"):
        program_class(directory)
