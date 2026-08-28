"""Guard: the Gemma-4 patch still matches the pinned deception-detection submodule.

The patch is how we get Gemma-4 support without redistributing Apollo's unlicensed
source (see third_party/patches/gemma4-port.patch). Its failure mode is silent: bump the
submodule pin and the patch stops applying, which nobody notices until someone tries to
train a probe. This test fails loudly instead.

It passes if the patch applies cleanly OR is already applied (reverse-applies cleanly),
so it is green both before and after a developer runs `git apply`.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH = ROOT / "third_party" / "patches" / "gemma4-port.patch"
SUBMODULE = ROOT / "third_party" / "deception-detection"


def _git_apply(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", "--check", "--directory=third_party/deception-detection",
         *args, str(PATCH)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_gemma4_patch_matches_pinned_submodule():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    if not (SUBMODULE / "deception_detection" / "models.py").exists():
        pytest.skip("deception-detection submodule not checked out")
    assert PATCH.exists(), f"missing patch: {PATCH}"

    forward = _git_apply()
    if forward.returncode == 0:
        return  # applies cleanly to a pristine submodule
    reverse = _git_apply("-R")
    if reverse.returncode == 0:
        return  # already applied in this working tree

    pytest.fail(
        "third_party/patches/gemma4-port.patch neither applies nor reverse-applies to "
        "the pinned submodule -- the pin probably moved and the patch needs rebasing.\n"
        f"forward: {forward.stderr.strip()}\nreverse: {reverse.stderr.strip()}"
    )


def test_patch_touches_only_apollo_files_it_documents():
    # The split matters: Apollo's files go through the patch (not redistributed), our own
    # code lives in the repo. A new file appearing here would mean we started shipping
    # code that no test can reach.
    assert PATCH.exists(), f"missing patch: {PATCH}"
    targets = sorted(
        line.split(" b/")[-1].strip()
        for line in PATCH.read_text().splitlines()
        if line.startswith("diff --git ")
    )
    assert targets == [
        "deception_detection/activations.py",
        "deception_detection/data/base.py",
        # repe.py overrides DialogueDataset.padding with its own model_type list, so the
        # base-class fix does not reach it and a Gemma-4 run dies on KeyError('gemma4').
        "deception_detection/data/repe.py",
        "deception_detection/models.py",
        "deception_detection/tokenized_data.py",
    ], targets


def test_patch_registers_e4b_with_its_own_layer_count():
    """E4B must be selectable, and its depth must not be inherited from 31B.

    A probe's layer index only means something on the checkpoint whose activations
    fitted it, so a wrong n_layers here yields a probe that trains and scores and is
    quietly reading the wrong depth.
    """
    text = PATCH.read_text()
    assert 'GEMMA_4_E4B = "gemma4-4b"' in text, "E4B is not registered"
    assert "n_layers = {4: 42, 31: 60}" in text, "E4B layer count missing or wrong"
    assert '"google/gemma-4-E4B-it"' in text, "E4B checkpoint id not mapped"


def test_model_name_size_parses_for_every_gemma4_entry():
    """ModelName.size does value.split('-')[1].replace('b', '') and int()s the result,
    so a name like 'gemma4-e4b' would raise. Guard the naming convention itself."""
    for value, expected in (("gemma4-31b", 31), ("gemma4-4b", 4)):
        assert int(value.split("-")[1].replace("b", "")) == expected


def test_padding_override_datasets_know_about_gemma4():
    """A dataset that overrides `padding` shadows the base class entirely.

    Adding gemma4 to DialogueDataset.padding is not enough: repe.py (the training set)
    rebuilds the dict from an explicit model_type list, so the lookup raises
    KeyError('gemma4') rather than falling back to anything.
    """
    text = PATCH.read_text()
    assert '"gemma", "gemma4", "mistral", "llama"' in text, (
        "repe.py's padding override does not list gemma4")


def test_patch_does_not_force_apollo_cluster_paths_for_gemma4():
    """local_files_only=True with a /data/huggingface default makes a first download
    impossible anywhere but Apollo's cluster."""
    text = PATCH.read_text()
    assert 'os.environ.get("GEMMA4_LOCAL_ONLY", "0") == "1"' in text
    assert '+    cache_dir = os.environ.get("GEMMA4_CACHE_DIR") or None' in text
