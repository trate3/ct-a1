#!/usr/bin/env python3
"""Live Compute Engine list prices, and the hourly rate for the configured instance.

Prices come from the Cloud Billing Catalog API rather than a constant in this file, so
a rate quoted in a result can be re-derived later instead of trusted. They are LIST
prices: they exclude sustained-use discounts, committed-use discounts, credits, free
tier, and egress, so a figure computed here is an upper bound on what the project is
actually charged, not a billing statement.

  pricing.py                 rate for the configured machine
  pricing.py --json          machine-readable, for embedding in a result
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

COMPUTE_ENGINE_SERVICE = "6F81-5844-456A"
HOURS_PER_MONTH = 730  # GCP's own convention for monthly-priced SKUs

# (machine family, region) -> the SKU descriptions that make up one instance-hour.
# Matched case-insensitively as substrings against the catalog.
# Exact descriptions, not substrings: "balanced pd capacity" also matches "Regional
# Balanced PD Capacity" (twice the price, and not what a zonal pd-balanced disk costs)
# and the snapshot SKUs. A silent 2x error in a cost line is worse than no cost line.
MATCHERS = {
    "core": ("g2 instance core running in americas",),
    "ram": ("g2 instance ram running in americas",),
    "gpu": ("nvidia l4 gpu running in americas",),
    "disk": ("balanced pd capacity",),
    # The catalog exposes no ephemeral-external-IP SKU for this region, so the static
    # IP rate stands in. It is the more expensive of the two, keeping this an upper
    # bound, and at ~1% of the instance rate the distinction does not move any total.
    "ip": ("static ip charge",),
}


def access_token() -> str:
    return subprocess.run(["gcloud", "auth", "print-access-token"],
                          capture_output=True, text=True, check=True).stdout.strip()


def catalog(region: str, token: str) -> dict[str, tuple[str, float, str]]:
    """Cheapest on-demand rate per component, from every page of the catalog."""
    base = f"https://cloudbilling.googleapis.com/v1/services/{COMPUTE_ENGINE_SERVICE}/skus"
    found: dict[str, tuple[str, float, str]] = {}
    page = None
    while True:
        url = f"{base}?pageSize=5000&currencyCode=USD" + (f"&pageToken={page}" if page else "")
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        for sku in payload.get("skus", []):
            description = sku.get("description", "")
            low = description.lower()
            if region not in sku.get("serviceRegions", []):
                continue
            # Spot and committed-use are different products; this priced run is neither.
            if any(word in low for word in ("commitment", "spot", "preemptible",
                                            "sole tenancy", "custom", "dws")):
                continue
            for key, needles in MATCHERS.items():
                if key in found or low not in needles:
                    continue
                for info in sku.get("pricingInfo", []):
                    expression = info.get("pricingExpression", {})
                    unit = expression.get("usageUnitDescription", "")
                    for tier in expression.get("tieredRates", []):
                        price = tier.get("unitPrice", {})
                        value = int(price.get("units", 0)) + int(price.get("nanos", 0)) / 1e9
                        if value > 0:
                            found[key] = (description, value, unit)
                            break
        page = payload.get("nextPageToken")
        if not page:
            break
    return found


def rates(region: str, cores: int, ram_gib: int, gpus: int, disk_gib: int,
          token: str | None = None) -> dict:
    prices = catalog(region, token or access_token())
    missing = [k for k in ("core", "ram", "gpu") if k not in prices]
    if missing:
        raise SystemExit(f"catalog is missing {missing} for {region}")
    core = cores * prices["core"][1]
    ram = ram_gib * prices["ram"][1]
    gpu = gpus * prices["gpu"][1]
    # Monthly-priced SKUs, converted at GCP's 730-hour month.
    disk = disk_gib * prices["disk"][1] / HOURS_PER_MONTH if "disk" in prices else 0.0
    ip = prices["ip"][1] if "ip" in prices else 0.0
    return {
        "region": region,
        "source": "cloud billing catalog api, list price, usd",
        "components": {k: {"sku": v[0], "unit_price": v[1], "unit": v[2]}
                       for k, v in prices.items()},
        "hourly": {"core": core, "ram": ram, "gpu": gpu, "disk": disk, "ip": ip},
        # What an hour costs while the instance is RUNNING versus merely existing.
        "running_usd_per_hour": core + ram + gpu + disk + ip,
        "stopped_usd_per_hour": disk,
        "compute_usd_per_hour": core + ram + gpu,
        "caveat": ("list price: excludes sustained-use and committed-use discounts, "
                   "credits and free tier, so this is an upper bound"),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-west1")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--ram-gib", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--disk-gib", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = rates(args.region, args.cores, args.ram_gib, args.gpus, args.disk_gib)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"region {result['region']} — list price, USD\n")
    for key, value in result["hourly"].items():
        component = result["components"].get(key) or {}
        print(f"  {key:<6} ${value:.6f}/h   {component.get('sku', '')[:52]}")
    print(f"\n  RUNNING  ${result['running_usd_per_hour']:.4f}/hour")
    print(f"  STOPPED  ${result['stopped_usd_per_hour']:.4f}/hour  (disk only)")
    print(f"\n  {result['caveat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
