"""Cost model: the dollar conversion behind RQ1.

The property that matters most here is what the model does with MISSING inputs.
An unresolved SKU rate silently becoming 0, or an unmeasured TEE slowdown
silently becoming 1.0, would understate certified cost and make RQ1 look better
than the evidence supports. Those cases are tested harder than the arithmetic.
"""

from __future__ import annotations

import json

import pytest

from evals_runner.util.cost_model import (
    PricingError,
    cost_breakdown,
    format_table,
    load_pricing,
)


def _result(direct_wall: float, certified_wall: float, n_rounds: int = 8) -> dict:
    return {
        "schema_version": "ct-eval/1",
        "config": {"experiment": "test", "n_rounds": n_rounds},
        "direct": {"metrics": {"wall_seconds": direct_wall, "cpu_seconds": direct_wall}},
        "certified": {
            "metrics": {
                "wall_seconds": certified_wall,
                "cpu_seconds": certified_wall,
                "wire_bytes": 9400,
            }
        },
    }


def _pricing(gpu_rate=None, vcpu=None, gib=None, slowdown=None) -> dict:
    return {
        "schema_version": "ct-cost/2",
        "retrieved": "2026-08-12",
        "verified_against_invoice": False,
        "confidential_gpu_skus": {
            "h100-80gb-intel-tdx": {
                "on_demand_sku": "OD", "spot_sku": "SPOT",
                "usd_per_gpu_hour_on_demand": gpu_rate,
                "usd_per_gpu_hour_spot": gpu_rate,
            },
            "a3-intel-tdx-instance": {
                "core_sku": "CORE", "ram_sku": "RAM",
                "usd_per_vcpu_hour": vcpu, "usd_per_gib_hour": gib,
                "usd_per_vcpu_hour_spot": vcpu, "usd_per_gib_hour_spot": gib,
            },
        },
        "tee_slowdown": {"scenarios": ({"central": slowdown} if slowdown else {})},
        "instances": {
            "gpu-box": {
                "vcpus": 10, "memory_gib": 100, "gpu": "1x H100", "gpu_count": 1,
                "base_usd_per_hour": 36.0, "base_usd_per_hour_spot": 3.6,
                "confidential_tech": "h100-80gb-intel-tdx",
            },
            "no-cc-box": {
                "vcpus": 4, "memory_gib": 16, "gpu": "1x RTX 3090", "gpu_count": 1,
                "base_usd_per_hour": 1.0, "base_usd_per_hour_spot": 1.0,
                "confidential_tech": None,
            },
        },
    }


def test_protocol_overhead_is_computable_without_any_tee_inputs():
    # Term 1 is measured locally and must not be blocked by unresolved SKUs.
    b = cost_breakdown(_result(100.0, 110.0), _pricing(), "gpu-box")
    assert b["seconds"]["protocol_overhead_wall"] == pytest.approx(10.0)
    # 10s at $3.60/hr spot = $0.01
    assert b["usd"]["protocol_overhead"] == pytest.approx(0.01)
    assert b["usd"]["baseline_uncertified"] == pytest.approx(0.1)
    # the FRACTION is rate-free: overhead wall / direct wall
    assert b["ratios_vs_baseline"]["protocol_overhead"] == pytest.approx(0.1)


def test_missing_tee_inputs_yield_none_not_zero():
    b = cost_breakdown(_result(100.0, 110.0), _pricing(), "gpu-box")
    assert b["usd"]["certified_on_tee_total_by_scenario"] is None
    assert b["ratios_vs_baseline"]["total_on_tee_by_scenario"] is None
    blocked = " ".join(b["unavailable"])
    for expected in ("usd_per_gpu_hour_spot", "usd_per_vcpu_hour_spot",
                     "usd_per_gib_hour_spot", "tee_slowdown"):
        assert expected in blocked, blocked
    # the SKU id must be surfaced so the number can actually be looked up
    assert "SPOT" in blocked and "CORE" in blocked and "RAM" in blocked


def test_missing_slowdown_alone_still_blocks_the_total():
    # Rates known, slowdown unknown -> still refuse. 1.0 is not a safe default.
    b = cost_breakdown(_result(100.0, 110.0), _pricing(gpu_rate=2.0, vcpu=0.01, gib=0.001),
                       "gpu-box")
    assert b["usd"]["certified_on_tee_total_by_scenario"] is None
    assert any("tee_slowdown" in u for u in b["unavailable"])


def test_full_inputs_produce_the_total_and_apply_the_slowdown():
    pricing = _pricing(gpu_rate=2.0, vcpu=0.01, gib=0.001, slowdown=1.5)
    b = cost_breakdown(_result(100.0, 110.0), pricing, "gpu-box")
    assert b["unavailable"] == []
    # confidential rate = 1*2.0 + 10*0.01 + 100*0.001 = 2.2 $/hr
    # tee wall = 110 * 1.5 = 165s -> 165/3600 * 2.2
    assert b["seconds"]["tee_wall_estimate_by_scenario"]["central"] == pytest.approx(165.0)
    assert b["usd"]["certified_on_tee_total_by_scenario"]["central"] == pytest.approx(
        165.0 / 3600.0 * 2.2
    )
    assert b["ratios_vs_baseline"]["total_on_tee_by_scenario"]["central"] == pytest.approx(
        (165.0 / 3600.0 * 2.2) / 0.1
    )


def test_instance_without_cc_mode_is_reported_as_such():
    b = cost_breakdown(_result(100.0, 110.0), _pricing(), "no-cc-box")
    assert b["usd"]["certified_on_tee_total_by_scenario"] is None
    assert any("no confidential-computing mode" in u for u in b["unavailable"])
    # but its protocol overhead is still priced
    assert b["usd"]["protocol_overhead"] is not None


def test_trivial_denominator_is_flagged():
    # The mock workload case: real protocol cost divided by ~zero inference.
    b = cost_breakdown(_result(0.0001, 0.0142), _pricing(), "gpu-box")
    assert any("not RQ1 evidence" in w for w in b["warnings"])
    # a real-inference run is not flagged for that reason
    real = cost_breakdown(_result(100.0, 110.0), _pricing(), "gpu-box")
    assert not any("not RQ1 evidence" in w for w in real["warnings"])


def test_spot_is_the_default_basis_and_on_demand_is_opt_in():
    r = _result(100.0, 110.0)
    assert cost_breakdown(r, _pricing(), "gpu-box")["pricing_basis"] == "spot"
    on_demand = cost_breakdown(r, _pricing(), "gpu-box", use_spot=False)
    assert on_demand["pricing_basis"] == "on-demand"
    # on-demand is 10x the spot rate in the fixture
    assert on_demand["usd"]["baseline_uncertified"] == pytest.approx(1.0)


def test_unverified_pricing_is_always_warned():
    b = cost_breakdown(_result(100.0, 110.0), _pricing(), "gpu-box")
    assert any("not verified against an invoice" in w for w in b["warnings"])


def test_rejects_unknown_instance_and_wrong_schemas(tmp_path):
    with pytest.raises(PricingError, match="unknown instance"):
        cost_breakdown(_result(1.0, 2.0), _pricing(), "nope")
    with pytest.raises(PricingError, match="unexpected result schema"):
        cost_breakdown({"schema_version": "other"}, _pricing(), "gpu-box")
    wrong = tmp_path / "pricing.json"
    wrong.write_text(json.dumps({"schema_version": "ct-cost/1"}))
    with pytest.raises(PricingError, match="pricing schema"):
        load_pricing(wrong)


def test_shipped_pricing_table_loads_and_is_gpu_capable():
    pricing = load_pricing()
    assert "a3-highgpu-1g" in pricing["instances"]
    gpu_box = pricing["instances"]["a3-highgpu-1g"]
    assert gpu_box["gpu_count"] >= 1 and gpu_box["confidential_tech"]
    # every SKU we could not resolve must still carry its SKU id
    for entry in pricing["confidential_gpu_skus"].values():
        assert any(k.endswith("_sku") and entry[k] for k in entry)


def test_format_table_surfaces_blocked_terms_and_warnings():
    text = format_table([cost_breakdown(_result(0.0001, 0.0142), _pricing(), "gpu-box")])
    assert "TEE terms NOT computed" in text
    assert "not RQ1 evidence" in text


def test_overhead_fraction_survives_with_no_pricing_at_all():
    # RQ1's "fraction of computational resources" needs no rate: it is a ratio of
    # measured wall-clock. It must stay available on a box with no priced rate.
    pricing = _pricing()
    pricing["instances"]["gpu-box"]["base_usd_per_hour_spot"] = None
    b = cost_breakdown(_result(200.0, 210.0), pricing, "gpu-box")
    assert b["usd"]["baseline_uncertified"] is None  # no rate at all
    assert b["ratios_vs_baseline"]["protocol_overhead"] == pytest.approx(0.05)


def test_scenarios_give_one_wall_estimate_each():
    pricing = _pricing()
    pricing["tee_slowdown"]["scenarios"] = {"low": 1.02, "central": 1.06, "high": 1.25}
    b = cost_breakdown(_result(100.0, 110.0), pricing, "gpu-box")
    walls = b["seconds"]["tee_wall_estimate_by_scenario"]
    assert set(walls) == {"low", "central", "high"}
    assert walls["high"] == pytest.approx(137.5)
    assert walls["low"] < walls["central"] < walls["high"]
