"""Runtime telemetry: accelerator peaks and host provenance for a measured arm.

The local comparison recorded host wall time, host CPU time, and the process RSS
high-water mark. On an accelerator those three stop describing where the work
happened: the weights live in device memory rather than in RSS, the generation
kernels are asynchronous so the host clock can stop before the device is done, and
the machine is a rented instance whose type belongs in the record.

This module supplies the missing columns and keeps `comparison.py` free of a hard
torch dependency -- the mocked experiment still runs with numpy alone, and every
lookup here degrades to `None`/`{}` rather than raising.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import resource
import sys
import urllib.error
import urllib.request

_METADATA = "http://metadata.google.internal/computeMetadata/v1"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def _torch():
    """Return the imported torch module, or None when it is not installed."""
    return sys.modules.get("torch") or _import_torch()


def _import_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


class NullAccelerator:
    """Stands in when no accelerator is present, so callers need no branches."""

    available = False
    device = "cpu"

    def reset(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def peak(self) -> dict | None:
        return None

    def describe(self) -> dict:
        return {"kind": "cpu", "available": False}


class CudaAccelerator:
    """CUDA peak-memory and synchronization hooks for one timed arm."""

    available = True
    device = "cuda"

    def __init__(self, torch_module, index: int = 0):
        self._torch = torch_module
        self._index = index
        self._baseline_bytes: int | None = None

    def reset(self) -> None:
        # Sets the recorded peak to the *current* allocation, so the peak reported
        # for this arm includes the already-resident weights and adds whatever the
        # arm itself allocates on top. The baseline is kept so the arm's own
        # footprint (KV cache, retained layer states) stays separable.
        self._torch.cuda.synchronize(self._index)
        self._torch.cuda.reset_peak_memory_stats(self._index)
        self._baseline_bytes = int(self._torch.cuda.memory_allocated(self._index))

    def synchronize(self) -> None:
        # Generation launches asynchronously; without this the host clock can stop
        # while device work is still queued, understating the arm.
        self._torch.cuda.synchronize(self._index)

    def peak(self) -> dict:
        peak_allocated = int(self._torch.cuda.max_memory_allocated(self._index))
        added = (None if self._baseline_bytes is None
                 else peak_allocated - self._baseline_bytes)
        return {
            "device_peak_allocated_bytes": peak_allocated,
            "device_peak_reserved_bytes": int(
                self._torch.cuda.max_memory_reserved(self._index)),
            "device_resident_at_arm_start_bytes": self._baseline_bytes,
            "device_added_by_arm_bytes": added,
        }

    def describe(self) -> dict:
        torch = self._torch
        properties = torch.cuda.get_device_properties(self._index)
        return {
            "kind": "cuda",
            "available": True,
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": getattr(properties, "multi_processor_count", None),
            "device_count": int(torch.cuda.device_count()),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "driver": _nvidia_driver_version(),
            # Identifies the physical card. A stopped and restarted instance can be
            # scheduled onto different hardware, which would show up as unexplained
            # variance when timing data from two sessions is pooled. Recording the UUID
            # makes that checkable instead of assumed.
            "uuid": _nvidia_query("uuid"),
        }


def detect_accelerator():
    """Pick the accelerator actually in use, without importing torch on the mock path."""
    torch = _torch()
    if torch is None:
        return NullAccelerator()
    try:
        if torch.cuda.is_available():
            return CudaAccelerator(torch, torch.cuda.current_device())
    except Exception:
        return NullAccelerator()
    # MPS exposes no peak-memory counter comparable to CUDA's, so it is recorded as
    # an accelerator without memory columns rather than silently reported as CPU.
    try:
        if torch.backends.mps.is_available():
            return NullAccelerator()
    except Exception:
        pass
    return NullAccelerator()


def _nvidia_query(field: str) -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return None
    return out.stdout.strip().splitlines()[0].strip() or None


def _nvidia_driver_version() -> str | None:
    return _nvidia_query("driver_version")


def _cpu_model() -> str | None:
    """The host CPU, for the same reason as the GPU UUID."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _metadata(path: str) -> str | None:
    """One best-effort GCE metadata-server read; None anywhere but a GCE instance."""
    request = urllib.request.Request(f"{_METADATA}/{path}", headers=_METADATA_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.read().decode().strip()
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def gce_instance() -> dict | None:
    """Identify the Compute Engine instance this measurement ran on, if any."""
    name = _metadata("instance/name")
    if name is None:
        return None
    scheduling = _metadata("instance/scheduling/?recursive=true")
    try:
        scheduling = json.loads(scheduling) if scheduling else {}
    except json.JSONDecodeError:
        scheduling = {}
    tail = lambda value: value.rsplit("/", 1)[-1] if value else None
    return {
        "provider": "gce",
        "instance_name": name,
        "instance_id": _metadata("instance/id"),
        "machine_type": tail(_metadata("instance/machine-type")),
        "zone": tail(_metadata("instance/zone")),
        "project": _metadata("project/project-id"),
        "image": tail(_metadata("instance/image")),
        "preemptible": scheduling.get("preemptible"),
        "provisioning_model": scheduling.get("provisioningModel"),
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def host_provenance(accelerator=None) -> dict:
    """Where and on what this measurement ran -- recorded, never used as a control."""
    accelerator = accelerator or detect_accelerator()
    page = resource.getpagesize()
    try:
        total_ram = os.sysconf("SC_PHYS_PAGES") * page
    except (ValueError, OSError, AttributeError):
        total_ram = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "total_ram_bytes": total_ram,
        "rss_unit_note": (
            "process_lifetime_max_rss_bytes is host RSS and excludes device memory; "
            "read the accelerator block for weights and activations on GPU"),
        "versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "numpy", "accelerate")
        },
        "accelerator": accelerator.describe(),
        "instance": gce_instance(),
    }
