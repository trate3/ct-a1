"""Build an Apollo `TokenizedDataset` from pre-tokenized Gemma-4 "thinking rollout" records.

This is OUR code, not Apollo's: it contains none of their source, so unlike the Gemma-4
loader changes (`third_party/patches/gemma4-port.patch`) it lives here under version
control where it can be tested. It is used at probe-TRAINING time, in the Apollo env,
and needs `deception_detection` importable -- hence the lazy imports below, matching
`program.py` and `probe_tap.py`.

Why it exists. Apollo's shipped datasets score the "detect tokens" they define inside a
fully rendered dialogue. A Gemma-4 rollout with thinking on is different: the model emits
a thought channel and then a response, and the thing we want the probe to read is exactly
those generated tokens -- not the scenario, not the instruction, not the prompt. Each
record supplies the prompt tokens (system+user turns plus the generation prompt ending at
`<|turn>model\\n`) and the raw generated tokens (thought channel then response, possibly
ending in `<turn|>`). We form `full_ids = prompt_ids + generation_ids`, left-pad the batch
with the pad token (bos), and build a detection mask True over exactly the generation
tokens and False over the prompt and the padding.

The result is drop-in compatible with `Activations.from_model` and the `DirectionDetector`
scoring path, so no Apollo file needs to change to consume it.

Note this makes the probe read the model's hidden reasoning as well as its visible answer,
which is a different measurement from the one `ProbeMonitor` performs at certification time
(see probe-track.md); the two must be reconciled before the numbers are comparable.
"""

from __future__ import annotations

from typing import Any


def build_thinking_tokenized_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    pad_to: int | None = None,
    dataset_partial_id: str = "thinking_rollouts",
) -> Any:
    """Build a left-padded `TokenizedDataset` from thinking-rollout records.

    Args:
        records: dicts with keys ``label`` ("honest"|"deceptive"), ``prompt_ids``
            (list[int]) and ``generation_ids`` (list[int]); ``idx``, ``scenario`` and
            ``question`` are carried by callers but not required here.
        tokenizer: the Gemma-4 tokenizer. We set ``pad_token_id = bos_token_id`` and pad
            with it, consistent with the rest of Apollo's repo (bos as pad, left-padding).
        pad_to: optional fixed sequence length. Defaults to the longest ``full_ids`` in
            the batch; must be >= that length if given.
        dataset_partial_id: metadata tag on the returned dataset. Set it per domain
            (roleplaying, insider, sandbagging, werewolf, repe) so mixed-domain training
            runs remain attributable.

    Returns:
        TokenizedDataset with ``tokens`` [n, seq], ``attention_mask`` [n, seq] (0 on the
        left-pad positions), and ``detection_mask`` [n, seq] True over exactly the
        ``generation_ids`` positions.
    """
    # Lazy heavy imports: torch, and Apollo's package (only present in the Apollo env).
    import torch

    from deception_detection.tokenized_data import TokenizedDataset, get_str_tokens
    from deception_detection.types import Message

    assert records, "Cannot build a TokenizedDataset from an empty record list"

    # Use bos as the pad token, matching the repo convention.
    tokenizer.pad_token_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no bos/pad token id"

    full_ids_list: list[list[int]] = []
    gen_lens: list[int] = []
    for rec in records:
        prompt_ids = list(rec["prompt_ids"])
        generation_ids = list(rec["generation_ids"])
        full_ids_list.append(prompt_ids + generation_ids)
        gen_lens.append(len(generation_ids))

    max_len = max(len(f) for f in full_ids_list)
    seq_len = pad_to if pad_to is not None else max_len
    assert seq_len >= max_len, (
        f"pad_to={pad_to} is shorter than the longest full_ids length ({max_len})"
    )

    n = len(records)
    tokens = torch.full((n, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    detection_mask = torch.zeros((n, seq_len), dtype=torch.bool)

    formatted_dialogues: list[str] = []
    dialogues: list[list[Message]] = []

    for row, (rec, full_ids, gen_len) in enumerate(
        zip(records, full_ids_list, gen_lens, strict=True)
    ):
        length = len(full_ids)
        left_pad = seq_len - length
        # place the real tokens on the right (left-padding)
        tokens[row, left_pad:] = torch.tensor(full_ids, dtype=torch.long)
        attention_mask[row, left_pad:] = 1
        # detection mask: True over exactly the final gen_len real tokens, i.e. the
        # generation_ids. Left-padding makes the generation the tail of every row.
        if gen_len > 0:
            detection_mask[row, seq_len - gen_len : seq_len] = True

        formatted_dialogues.append(tokenizer.decode(full_ids))

        # Minimal placeholder dialogue so the field has length n and slicing works. The
        # mask is built explicitly above and does not depend on this.
        dialogues.append(
            [
                Message("user", tokenizer.decode(list(rec["prompt_ids"])), False),
                Message("assistant", tokenizer.decode(list(rec["generation_ids"])), True),
            ]
        )

    # str_tokens: reuse Apollo's helper (it strips pad/bos and prepends bos) when it
    # works, else decode per token. Either way it is length n, so slicing works.
    try:
        str_tokens = get_str_tokens(tokens, tokenizer)
        assert len(str_tokens) == n
    except Exception:
        str_tokens = [
            [tokenizer.decode([int(t)]) for t in tokens[row].tolist()] for row in range(n)
        ]

    return TokenizedDataset(
        dialogues=dialogues,
        formatted_dialogues=formatted_dialogues,
        tokens=tokens,
        str_tokens=str_tokens,
        detection_mask=detection_mask,
        attention_mask=attention_mask,
        dataset_partial_id=dataset_partial_id,
    )
