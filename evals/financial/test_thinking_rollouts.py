"""Detection-mask correctness for build_thinking_tokenized_dataset.

The whole point of the module is WHERE the mask is True: exactly the generated thought
channel + response, never the prompt and never the left-padding. If that slips, the probe
silently trains on the scenario text (which states the deception) instead of on the
model's own tokens, and the resulting numbers are worthless but look fine.

Runs with a stub tokenizer -- no weights, no network, no GPU -- so it is a real guard
rather than something that only runs on the training box. Needs Apollo's package for
TokenizedDataset, so it skips when the submodule is not installed.
"""

from __future__ import annotations

import pytest


class StubTokenizer:
    """Minimal stand-in: the module only needs bos/pad ids and decode().

    Deliberately not named *llama*: Apollo's decode_single_token special-cases that.
    """

    bos_token_id = 2
    bos_token = "<bos>"
    pad_token_id = 2

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(f"t{int(i)}" for i in ids)


def _records():
    # Deliberately different total lengths so left-padding is actually exercised.
    return [
        {"idx": 0, "label": "honest", "prompt_ids": [2, 10, 11, 12], "generation_ids": [20, 21]},
        {"idx": 1, "label": "deceptive", "prompt_ids": [2, 10], "generation_ids": [30, 31, 32, 33]},
        {"idx": 2, "label": "deceptive", "prompt_ids": [2, 10, 11, 12, 13, 14],
         "generation_ids": [40]},
    ]


def _build(records=None, **kw):
    pytest.importorskip("torch")
    pytest.importorskip("deception_detection")
    from evals.financial.thinking_rollouts import build_thinking_tokenized_dataset

    # `is None`, not `or`: an empty list is a case under test, not "use the default".
    return build_thinking_tokenized_dataset(
        _records() if records is None else records, StubTokenizer(), **kw
    )


def test_detection_mask_covers_exactly_the_generated_tokens():
    torch = pytest.importorskip("torch")
    recs = _records()
    td = _build(recs)

    assert len(td) == len(recs)
    seq_len = td.tokens.shape[1]
    assert seq_len == max(len(r["prompt_ids"]) + len(r["generation_ids"]) for r in recs)

    for row, rec in enumerate(recs):
        gen_len = len(rec["generation_ids"])
        full_len = len(rec["prompt_ids"]) + gen_len
        left_pad = seq_len - full_len
        dmask, amask = td.detection_mask[row], td.attention_mask[row]

        # exactly gen_len positions, contiguous, at the tail (left-padding puts it there)
        assert int(dmask.sum()) == gen_len
        true_idx = torch.nonzero(dmask, as_tuple=False).flatten().tolist()
        assert true_idx == list(range(seq_len - gen_len, seq_len))

        # the masked tokens ARE the generation, not the prompt
        assert td.tokens[row][dmask].tolist() == rec["generation_ids"]

        # attention covers the real tokens only; padding is masked out of both
        assert int(amask.sum()) == full_len
        assert not bool(dmask[:left_pad].any())
        assert int(amask[:left_pad].sum()) == 0


def test_prompt_tokens_are_never_detected():
    recs = _records()
    td = _build(recs)
    for row, rec in enumerate(recs):
        prompt_len = len(rec["prompt_ids"])
        left_pad = td.tokens.shape[1] - (prompt_len + len(rec["generation_ids"]))
        prompt_span = td.detection_mask[row][left_pad : left_pad + prompt_len]
        assert not bool(prompt_span.any()), f"row {row}: prompt token marked for detection"


def test_slicing_preserves_masks():
    torch = pytest.importorskip("torch")
    td = _build()

    sub = td[0:2]
    assert len(sub) == 2
    assert torch.equal(sub.tokens, td.tokens[0:2])
    assert torch.equal(sub.detection_mask, td.detection_mask[0:2])
    assert torch.equal(sub.attention_mask, td.attention_mask[0:2])
    assert len(sub.dialogues) == len(sub.formatted_dialogues) == len(sub.str_tokens) == 2

    one = td[1]
    assert len(one) == 1
    assert torch.equal(one.detection_mask[0], td.detection_mask[1])


def test_pad_to_and_partial_id():
    recs = _records()
    longest = max(len(r["prompt_ids"]) + len(r["generation_ids"]) for r in recs)

    td = _build(recs, pad_to=longest + 5, dataset_partial_id="insider")
    assert td.tokens.shape[1] == longest + 5
    assert td.dataset_partial_id == "insider"
    # padding further left must not shift the detection mask off the generation
    for row, rec in enumerate(recs):
        assert td.tokens[row][td.detection_mask[row]].tolist() == rec["generation_ids"]

    with pytest.raises(AssertionError, match="shorter than the longest"):
        _build(recs, pad_to=longest - 1)


def test_empty_records_rejected():
    with pytest.raises(AssertionError, match="empty record list"):
        _build([])
