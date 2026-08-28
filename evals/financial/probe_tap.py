"""Linear-probe tap: the white-box probe monitor (Apollo probe track).

This is the probe instantiation of a tap: instead of reading text, it reads the
program's residual-stream activations and applies a trained linear probe.

    SOURCE (referenced, driven as-published; NOT copied):
      git submodule third_party/deception-detection
      repo:  https://github.com/ApolloResearch/deception-detection  (no OSS license;
             run as-published, do not redistribute -- see probe-track.md at the root)
      paper: Goldowsky-Dill et al., "Detecting Strategic Deception with Linear
             Probes", ICML 2025 (arXiv:2502.03407)

We do not import Apollo's package (heavy deps, and its models.py hardcodes
/data/huggingface). Instead we consume its documented artifact: `detector.pt` is
a pickle of a dict
    {layers: list[int], directions: Tensor[n_layers, emb],
     scaler_mean: Tensor[n_layers, emb] | None, scaler_scale: ... | None,
     normalize: bool, reg_coeff: float}
written by deception_detection/detectors.py::LogisticRegressionDetector.save. The
scoring below mirrors that file's `.score` (standardize -> per-layer dot with the
direction -> mean over layers) plus MeanPromptScorer (mean over tokens). Train a
probe per RUNBOOK Phase 2 (probe-track.md at the repo root has the provenance and
usage), then point ProbeMonitor at the resulting detector.pt. Torch is needed only to
unpickle; the math is numpy.
"""

from __future__ import annotations

import io
import pathlib
from dataclasses import dataclass

from ct_core.interfaces import Monitor, MonitorInput, Prediction


@dataclass
class ApolloProbe:
    layers: list[int]
    directions: "object"     # numpy array [n_layers, emb]
    scaler_mean: "object"    # numpy array [n_layers, emb] or None
    scaler_scale: "object"   # numpy array [n_layers, emb] or None
    normalize: bool

    def content_hash(self) -> str:
        """Evaluation identity for the probe weights; h_c hashes their safe package."""
        import hashlib

        import numpy as np

        h = hashlib.sha256()
        for arr in (self.directions, self.scaler_mean, self.scaler_scale):
            h.update(b"None" if arr is None else np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(str(self.layers).encode())
        h.update(str(self.normalize).encode())
        return h.hexdigest()

    def safe_bytes(self) -> bytes:
        """Canonical non-executable package consumed by the trusted workload."""
        import numpy as np

        output = io.BytesIO()
        np.savez(
            output,
            layers=np.asarray(self.layers, dtype=np.int64),
            directions=np.asarray(self.directions),
            scaler_mean=(np.asarray(self.scaler_mean) if self.scaler_mean is not None
                         else np.asarray([], dtype=np.float64)),
            scaler_scale=(np.asarray(self.scaler_scale) if self.scaler_scale is not None
                          else np.asarray([], dtype=np.float64)),
            normalize=np.asarray([self.normalize], dtype=np.bool_),
        )
        return output.getvalue()

    @classmethod
    def from_safe_bytes(cls, encoded: bytes) -> "ApolloProbe":
        """Parse the inert NPZ form; pickle artifacts never cross the trusted boundary."""
        import numpy as np

        with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
            required = {"layers", "directions", "scaler_mean", "scaler_scale", "normalize"}
            if set(data.files) != required:
                raise ValueError("malformed probe package")
            mean, scale = data["scaler_mean"], data["scaler_scale"]
            return cls(
                layers=data["layers"].astype(int).tolist(),
                directions=data["directions"],
                scaler_mean=None if mean.size == 0 else mean,
                scaler_scale=None if scale.size == 0 else scale,
                normalize=bool(data["normalize"][0]),
            )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ApolloProbe":
        """Load Apollo's detector.pt into numpy arrays.

        LogisticRegressionDetector.save (deception_detection/detectors.py) writes a
        plain `pickle.dump` of a dict of torch tensors, NOT torch.save, so read it
        with pickle.load. torch must be importable so the pickled tensors reconstruct.
        """
        import pickle

        import torch  # noqa: F401  needed so the pickled torch tensors reconstruct

        with open(path, "rb") as f:
            d = pickle.load(f)
        to_np = lambda t: None if t is None else t.float().cpu().numpy()
        return cls(
            layers=list(d["layers"]),
            directions=to_np(d["directions"]),
            scaler_mean=to_np(d.get("scaler_mean")),
            scaler_scale=to_np(d.get("scaler_scale")),
            normalize=bool(d.get("normalize", True)),
        )

    def raw_score(self, acts) -> float:
        """Apollo LogisticRegressionDetector.score + MeanPromptScorer, in numpy.

        acts: [num_tokens, n_layers, emb]. Standardize per layer (if normalize),
        dot each layer's activation with that layer's direction, mean over layers
        (per-token score), then mean over tokens. Raw logit-like scalar (no sigmoid),
        exactly as upstream; threshold is calibrated on a control set by the caller.
        """
        import numpy as np

        x = np.asarray(acts, dtype=np.float64)  # [T, L, E]
        if self.normalize and self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale
        per_token_layer = np.einsum("tle,le->tl", x, self.directions)  # [T, L]
        per_token = per_token_layer.mean(axis=1)                        # [T]
        return float(per_token.mean())                                  # scalar


class ProbeMonitor(Monitor):
    """A tap that scores the program's activations with an Apollo linear probe.

    z_hat = 1[raw_score > threshold]; `threshold` is the control-calibrated cutoff
    (Apollo calibrates on alpaca at a target FPR -- see probe-track.md). This is
    a white-box tap: it reads layer states retained from the generation calls that
    produced the response (`inp.activations`). `needs_activations` / `probe_layers`
    tell the enclave which layers to capture.
    """

    needs_activations = True

    def __init__(self, probe: ApolloProbe, *, threshold: float = 0.0):
        self._probe = probe
        self.probe_layers = list(probe.layers)
        self._threshold = threshold

    def predict(self, inp: MonitorInput) -> Prediction:
        if inp.activations is None:
            raise ValueError(
                "ProbeMonitor needs activations from the program's execution; run it "
                "with a program that captures them (Gemma2Program), not the mock."
            )
        raw = self._probe.raw_score(inp.activations)
        return Prediction(z_hat=int(raw > self._threshold), score=raw)

    def digest(self) -> str:
        # Evaluation identity; the certified package separately carries these values.
        return (f"apollo-probe-v1:layers={self._probe.layers}:thr={self._threshold}:"
                f"weights={self._probe.content_hash()[:16]}")
