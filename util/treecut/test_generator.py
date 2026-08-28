"""Tests for the shared TreeCut generator."""

from __future__ import annotations

import random

from util.treecut import generate_items, generate_qa


def test_generate_items_deterministic_and_balanced():
    a = generate_items(8, seed=7)
    assert generate_items(8, seed=7) == a          # reproducible from the seed
    labels = [it["z"] for it in a]
    assert set(labels) <= {0, 1} and 0 in labels and 1 in labels


def test_generate_qa_answerable_has_number_cut_is_unknown():
    random.seed(0)
    ans = generate_qa("food", False, 7, 3, "random", False)
    cut = generate_qa("food", False, 7, 3, "random", True, 1)
    assert ans["answer"] != "unknown"       # answerable -> a number
    assert cut["answer"] == "unknown"        # solution path cut -> unanswerable
