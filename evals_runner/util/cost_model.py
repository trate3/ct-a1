"""Turn measured direct-versus-certified resources into dollars (RQ1).

RQ1 asks whether certifying an evaluation costs about the same as running it
normally. For an LLM eval that must run on a GPU-capable TEE, that is three
charges, and collapsing them hides which one dominates:

  1. protocol overhead -- extra wall-clock the certified arm spends on commit,
        certify, sealed-box transport and verify. MEASURED, by `run_comparison`,
        as certified minus direct. Fully computable from a local run.
  2. confidential rate  -- confidential GPU/CPU/RAM are separate SKUs, not a
        published percentage premium. From the pricing table.
  3. TEE slowdown       -- CC mode is also slower (encrypted PCIe bounce
        buffers), so it costs extra wall-clock as well as extra dollars per
        hour. Workload dependent. NOT measurable on hardware without CC mode.

All three are charged on wall-clock, which is what clouds bill, so wall_seconds
is the basis; cpu_seconds is carried for reference only.

Design rule: an unknown input is reported as unavailable with the reason, never
silently defaulted. A missing TEE slowdown does not quietly become 1.0, and a
missing SKU rate does not quietly become zero -- either would understate the
cost and make RQ1 look better than the evidence supports.

Caution the ratios cannot express: protocol cost is essentially FIXED per round
(hash, sign, fixed-size transport) while the denominator scales with inference.
A ratio from a mock workload divides a real cost by nearly zero, so results whose
direct arm did no real inference are flagged and must not be quoted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "ct-cost/2"
RESULT_SCHEMA = "ct-eval/1"
DEFAULT_PRICING = Path(__file__).with_name("pricing_gcp.json")

SECONDS_PER_HOUR = 3600.0
# Below this, the direct arm cannot have done meaningful model inference.
TRIVIAL_DENOMINATOR_SECONDS = 0.5


class PricingError(ValueError):
    """Raised when the pricing table cannot price the requested instance."""


@dataclass(frozen=True)
class Instance:
    name: str
    vcpus: int
    memory_gib: float
    gpu: str | None
    gpu_count: int
    confidential_tech: str | None
    base_usd_per_hour: float | None
    base_usd_per_hour_spot: float | None


def load_pricing(path: str | Path | None = None) -> dict:
    table = json.loads(Path(path or DEFAULT_PRICING).read_text())
    if table.get("schema_version") != SCHEMA_VERSION:
        raise PricingError(
            f"pricing schema {table.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    return table


def get_instance(pricing: dict, name: str) -> Instance:
    spec = pricing.get("instances", {}).get(name)
    if spec is None:
        available = ", ".join(sorted(pricing.get("instances", {})))
        raise PricingError(f"unknown instance {name!r}; known: {available}")
    return Instance(
        name=name,
        vcpus=int(spec["vcpus"]),
        memory_gib=float(spec["memory_gib"]),
        gpu=spec.get("gpu"),
        gpu_count=int(spec.get("gpu_count") or 0),
        confidential_tech=spec.get("confidential_tech"),
        base_usd_per_hour=spec.get("base_usd_per_hour"),
        base_usd_per_hour_spot=spec.get("base_usd_per_hour_spot"),
    )


def base_rate(instance: Instance, use_spot: bool) -> tuple[float | None, str]:
    """Ordinary (non-confidential) $/hour, and which basis it is."""
    if use_spot:
        return instance.base_usd_per_hour_spot, "spot"
    return instance.base_usd_per_hour, "on-demand"


def confidential_rate(
    pricing: dict, instance: Instance, use_spot: bool
) -> tuple[float | None, list[str]]:
    """Confidential $/hour for this shape, or None with the reasons it is unknown."""
    missing: list[str] = []
    tech = instance.confidential_tech
    if tech is None:
        return None, [f"{instance.name} has no confidential-computing mode"]

    gpu_rates = pricing.get("confidential_gpu_skus", {}).get(tech)
    if gpu_rates is None:
        return None, [f"no confidential SKU entry for {tech!r}"]

    gpu_key = "usd_per_gpu_hour_spot" if use_spot else "usd_per_gpu_hour_on_demand"
    gpu_rate = gpu_rates.get(gpu_key)
    if gpu_rate is None and instance.gpu_count:
        missing.append(f"{tech}.{gpu_key} (SKU {gpu_rates.get('spot_sku' if use_spot else 'on_demand_sku')})")

    machine = pricing.get("confidential_gpu_skus", {}).get("a3-intel-tdx-instance", {})
    vcpu_key = "usd_per_vcpu_hour_spot" if use_spot else "usd_per_vcpu_hour"
    gib_key = "usd_per_gib_hour_spot" if use_spot else "usd_per_gib_hour"
    vcpu_rate, gib_rate = machine.get(vcpu_key), machine.get(gib_key)
    if vcpu_rate is None:
        missing.append(f"a3-intel-tdx-instance.{vcpu_key} (SKU {machine.get('core_sku')})")
    if gib_rate is None:
        missing.append(f"a3-intel-tdx-instance.{gib_key} (SKU {machine.get('ram_sku')})")

    if missing:
        return None, missing
    return (
        instance.gpu_count * (gpu_rate or 0.0)
        + instance.vcpus * vcpu_rate
        + instance.memory_gib * gib_rate
    ), []


def _usd(rate_per_hour: float | None, seconds: float) -> float | None:
    if rate_per_hour is None:
        return None
    return rate_per_hour * seconds / SECONDS_PER_HOUR


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def cost_breakdown(
    result: dict, pricing: dict, instance_name: str, use_spot: bool = True
) -> dict:
    """Price one `run_comparison` result.

    `use_spot` defaults True because confidential H100 on A3 is Spot/flex-start
    only, so Spot is the honest baseline to compare a confidential run against.
    """
    if result.get("schema_version") != RESULT_SCHEMA:
        raise PricingError(f"unexpected result schema {result.get('schema_version')!r}")
    instance = get_instance(pricing, instance_name)

    direct_wall = float(result["direct"]["metrics"]["wall_seconds"])
    certified_wall = float(result["certified"]["metrics"]["wall_seconds"])
    overhead_wall = certified_wall - direct_wall

    rate, basis = base_rate(instance, use_spot)
    conf_rate, conf_missing = confidential_rate(pricing, instance, use_spot)

    slowdown_spec = pricing.get("tee_slowdown", {})
    scenarios = slowdown_spec.get("scenarios") or {}
    unavailable: list[str] = []
    if rate is None:
        unavailable.append(f"{instance.name}.base_usd_per_hour[{basis}] is unset")
    unavailable.extend(f"confidential rate unknown: {m}" for m in conf_missing)
    if not scenarios:
        unavailable.append(
            "tee_slowdown.scenarios is unset (CC-mode wall-clock multiplier); "
            "not measurable on hardware without CC mode"
        )

    # Term 1 is measurable now, independent of the unresolved TEE inputs.
    baseline_usd = _usd(rate, direct_wall)
    protocol_overhead_usd = _usd(rate, overhead_wall)

    # Terms 2 and 3 need the TEE inputs; stay None if any is missing. One wall-clock
    # estimate and one total per published slowdown scenario, because the true value is
    # workload dependent and a single number would be a false point estimate.
    tee_wall = {k: certified_wall * v for k, v in scenarios.items()}
    tee_total_usd = {
        k: _usd(conf_rate, w) for k, w in tee_wall.items()
    } if conf_rate is not None else {k: None for k in tee_wall}

    warnings: list[str] = []
    if direct_wall < TRIVIAL_DENOMINATOR_SECONDS:
        warnings.append(
            f"direct arm ran {direct_wall:.4f}s (< {TRIVIAL_DENOMINATOR_SECONDS}s): the "
            "denominator contains little or no model inference, so these ratios are "
            "not RQ1 evidence"
        )
    if not pricing.get("verified_against_invoice", False):
        warnings.append(
            f"pricing is list rates retrieved {pricing.get('retrieved')}, "
            "not verified against an invoice"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "instance": asdict(instance),
        "pricing_basis": basis,
        "pricing_retrieved": pricing.get("retrieved"),
        "experiment": result.get("config", {}).get("experiment"),
        "n_rounds": result.get("config", {}).get("n_rounds"),
        "seconds": {
            "direct_wall": direct_wall,
            "certified_wall": certified_wall,
            "protocol_overhead_wall": overhead_wall,
            "tee_wall_estimate_by_scenario": tee_wall or None,
            "direct_cpu": result["direct"]["metrics"]["cpu_seconds"],
            "certified_cpu": result["certified"]["metrics"]["cpu_seconds"],
        },
        "wire_bytes": result["certified"]["metrics"]["wire_bytes"],
        "usd": {
            "baseline_uncertified": baseline_usd,
            "protocol_overhead": protocol_overhead_usd,
            "certified_on_tee_total_by_scenario": tee_total_usd or None,
        },
        "ratios_vs_baseline": {
            # Rate-free: overhead/direct wall-clock. This is RQ1's "fraction of
            # computational resources" and needs NO pricing input at all, so it stays
            # available even when every SKU rate is unresolved.
            "protocol_overhead": _ratio(overhead_wall, direct_wall),
            "total_on_tee_by_scenario": {
                k: _ratio(v, baseline_usd) for k, v in tee_total_usd.items()
            } if any(v is not None for v in tee_total_usd.values()) else None,
        },
        "unavailable": unavailable,
        "warnings": warnings,
    }


def format_table(breakdowns: list[dict]) -> str:
    """Human-readable summary, one row per priced result."""
    header = (
        f"{'experiment':<30} {'rnds':>5} {'direct_s':>9} {'cert_s':>9} "
        f"{'ovh_s':>8} {'ovh_$':>10} {'ovh_%':>8}"
    )
    lines = [header, "-" * len(header)]
    for b in breakdowns:
        usd, ratios, secs = b["usd"], b["ratios_vs_baseline"], b["seconds"]
        ovh_usd = "n/a" if usd["protocol_overhead"] is None else f"{usd['protocol_overhead']:.6f}"
        ovh_pct = (
            "n/a" if ratios["protocol_overhead"] is None
            else f"{100 * ratios['protocol_overhead']:.2f}%"
        )
        lines.append(
            f"{str(b['experiment'])[:30]:<30} {b['n_rounds']:>5} "
            f"{secs['direct_wall']:>9.3f} {secs['certified_wall']:>9.3f} "
            f"{secs['protocol_overhead_wall']:>8.3f} {ovh_usd:>10} {ovh_pct:>8}"
        )

    blocked = sorted({u for b in breakdowns for u in b["unavailable"]})
    if blocked:
        lines += ["", "TEE terms NOT computed -- missing inputs:"]
        lines += [f"  - {u}" for u in blocked]
    flagged = [b for b in breakdowns if b["warnings"]]
    if flagged:
        lines += ["", "Warnings:"]
        for warning in sorted({w for b in flagged for w in b["warnings"]}):
            lines.append(f"  - {warning}")
    return "\n".join(lines)
