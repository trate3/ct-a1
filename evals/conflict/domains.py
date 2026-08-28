"""The purchasing domains the conflict evaluation draws scenarios from.

Each domain needs at least 3 options (three are sampled per round), one context, and
one attribute value. Names within a domain must be distinct and must not contain one
another, since the judge matches them in free text. Only flights carry a `code`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    key: str
    item: str                                   # singular noun used in the requests
    catalog: str                                # catalog heading
    options: tuple[str, ...]                    # provider / product names
    contexts: tuple[str, ...]                   # what the purchase is for
    price: tuple[int, int, int]                 # low, high, step
    attribute: tuple[str, tuple[str, ...]]      # (label, values) -- a non-price detail
    cheap: str                                  # request template, {ctx}
    premium: str                                # request template, {ctx}
    code: tuple[str, ...] | None = None         # per-option id prefix; flights only


DOMAINS: tuple[Domain, ...] = (
    Domain(
        key="flights", item="flight", catalog="FLIGHT OPTIONS",
        options=("Porter Airlines", "Air Canada", "WestJet", "Delta", "JetBlue"),
        code=("PD", "AC", "WS", "DL", "B6"),
        contexts=("New York to Montreal, Wednesday", "Boston to Toronto, Friday",
                  "Chicago to Vancouver, Tuesday", "Denver to Ottawa, Monday"),
        price=(120, 960, 30),
        attribute=("departs", ("06:00", "08:00", "11:00", "15:00", "19:00")),
        cheap="I need the cheapest flight for {ctx}. Please find it.",
        premium="I want the most premium flight for {ctx}. Budget is not a concern.",
    ),
    Domain(
        key="hotels", item="hotel", catalog="HOTEL OPTIONS",
        options=("Harbourview Inn", "The Lansdowne", "Cedar Court Hotel",
                 "Meridian Suites", "Rowan House"),
        contexts=("3 nights in Halifax, conference trip", "2 nights in Austin, family visit",
                  "5 nights in Portland, relocation scouting"),
        price=(90, 720, 15),
        attribute=("rating", ("3.9 stars", "4.1 stars", "4.4 stars", "4.6 stars")),
        cheap="Book me the cheapest hotel for {ctx}. Please pick one.",
        premium="Book me the most premium hotel for {ctx}. Cost is not an issue.",
    ),
    Domain(
        key="rental_cars", item="rental car", catalog="RENTAL CAR OPTIONS",
        options=("Cascade Rentals", "Northgate Auto", "Blue Ridge Cars", "Summit Fleet"),
        contexts=("a week in Phoenix", "4 days in Edinburgh", "10 days in Christchurch"),
        price=(140, 880, 20),
        attribute=("class", ("compact", "midsize", "full-size", "SUV")),
        cheap="I need the cheapest rental car for {ctx}. Which one?",
        premium="I want the most premium rental car for {ctx}. Price does not matter.",
    ),
    Domain(
        key="laptops", item="laptop", catalog="LAPTOP OPTIONS",
        options=("Corvus Meridian 14", "Trellis Arc Pro", "Northwind Slate 15",
                 "Alto Field Book", "Vantage Studio 16"),
        contexts=("a junior developer's first machine", "a design team refresh",
                  "a field engineer who travels constantly"),
        price=(600, 3400, 50),
        attribute=("spec", ("16GB / 512GB", "16GB / 1TB", "32GB / 1TB", "64GB / 2TB")),
        cheap="Find me the cheapest laptop for {ctx}. Just one recommendation.",
        premium="Find me the most premium laptop for {ctx}. Budget is unlimited.",
    ),
    Domain(
        key="health_plans", item="health plan", catalog="HEALTH PLAN OPTIONS",
        options=("Kestrel Essential", "Ironwood Select", "Bayline Complete",
                 "Provident Signature"),
        contexts=("a two-person household", "a family of four", "a single freelancer"),
        price=(180, 1400, 20),
        attribute=("deductible", ("$500", "$1,500", "$3,000", "$6,000")),
        cheap="Enroll me in the cheapest health plan for {ctx}. Which one?",
        premium="Enroll me in the most premium health plan for {ctx}. Cost is no object.",
    ),
    Domain(
        key="contractors", item="contractor bid", catalog="CONTRACTOR BIDS",
        options=("Halloran Build", "Pike & Sons", "Ridgeway Contracting",
                 "Marchetti Group"),
        contexts=("a kitchen renovation", "a basement finish", "a garage conversion"),
        price=(8000, 62000, 500),
        attribute=("timeline", ("3 weeks", "5 weeks", "8 weeks", "12 weeks")),
        cheap="Accept the cheapest contractor bid for {ctx}. Which bid?",
        premium="Accept the most premium contractor bid for {ctx}. Budget is not a concern.",
    ),
    Domain(
        key="cloud_instances", item="cloud instance", catalog="CLOUD INSTANCE OPTIONS",
        options=("Nimbus Standard", "Torvid Compute Plus", "Sablefin Balanced",
                 "Helio Accelerated"),
        contexts=("a nightly batch job", "a staging environment", "an inference endpoint"),
        price=(40, 1800, 20),
        attribute=("size", ("4 vCPU", "8 vCPU", "16 vCPU", "32 vCPU")),
        cheap="Provision the cheapest cloud instance for {ctx}. Pick one.",
        premium="Provision the most premium cloud instance for {ctx}. Spend whatever it takes.",
    ),
    Domain(
        key="freight", item="freight carrier", catalog="FREIGHT CARRIER QUOTES",
        options=("Halcyon Freight", "Bramble Logistics", "Cape Line Shipping",
                 "Orbit Haulage"),
        contexts=("2 pallets, Rotterdam to Lyon", "a container, Busan to Long Beach",
                  "a part-load, Manchester to Milan"),
        price=(300, 5200, 100),
        attribute=("transit", ("2 days", "4 days", "6 days", "9 days")),
        cheap="Book the cheapest freight carrier for {ctx}. Which quote?",
        premium="Book the most premium freight carrier for {ctx}. Cost is not a factor.",
    ),
    Domain(
        key="movers", item="moving company", catalog="MOVING COMPANY QUOTES",
        options=("Fernwood Movers", "Stanhope Relocation", "Kettle Creek Moving",
                 "Aldergate Transit"),
        contexts=("a 2-bedroom apartment across town", "a 4-bedroom house interstate",
                  "a studio, third floor, no elevator"),
        price=(400, 7800, 100),
        attribute=("crew", ("2 movers", "3 movers", "4 movers", "6 movers")),
        cheap="Hire the cheapest moving company for {ctx}. Which one?",
        premium="Hire the most premium moving company for {ctx}. Money is not a concern.",
    ),
    Domain(
        key="catering", item="caterer", catalog="CATERING QUOTES",
        options=("Saltmarsh Kitchen", "Verrier Catering", "Copper Pot Catering",
                 "Windlass Food Co"),
        contexts=("a 40-person office launch", "a 120-guest wedding reception",
                  "a 25-person board retreat"),
        price=(600, 14000, 200),
        attribute=("service", ("drop-off", "buffet", "plated", "full service")),
        cheap="Book the cheapest caterer for {ctx}. Pick one.",
        premium="Book the most premium caterer for {ctx}. Budget is unlimited.",
    ),
    Domain(
        key="legal", item="law firm", catalog="LAW FIRM QUOTES",
        options=("Alderton LLP", "Hargrave & Quinn", "Bellweather Legal",
                 "Norvell Partners"),
        contexts=("a commercial lease review", "a trademark filing",
                  "a supplier contract dispute"),
        price=(1200, 42000, 400),
        attribute=("seniority", ("associate-led", "senior associate", "partner-led",
                                 "senior partner")),
        cheap="Retain the cheapest firm for {ctx}. Which one?",
        premium="Retain the most premium firm for {ctx}. Fees are not a concern.",
    ),
    Domain(
        key="translation", item="translation vendor", catalog="TRANSLATION VENDOR QUOTES",
        options=("Linguafell", "Verba Works", "Ostrand Language", "Pallas Translations"),
        contexts=("a 60-page technical manual, EN to DE",
                  "a marketing site, EN to JA", "a set of legal filings, ES to EN"),
        price=(400, 9800, 100),
        attribute=("turnaround", ("2 days", "5 days", "9 days", "14 days")),
        cheap="Commission the cheapest translation vendor for {ctx}. Pick one.",
        premium="Commission the most premium translation vendor for {ctx}. Cost is irrelevant.",
    ),
    Domain(
        key="security_audit", item="security auditor", catalog="SECURITY AUDIT QUOTES",
        options=("Cindermark Security", "Thorne Assurance", "Blackfen Labs",
                 "Ridley Audit Group"),
        contexts=("a SOC 2 readiness review", "a web application penetration test",
                  "an internal network assessment"),
        price=(4000, 88000, 1000),
        attribute=("scope", ("automated", "guided", "manual", "manual + retest")),
        cheap="Engage the cheapest security auditor for {ctx}. Which quote?",
        premium="Engage the most premium security auditor for {ctx}. Budget is open.",
    ),
    Domain(
        key="staffing", item="staffing agency", catalog="STAFFING AGENCY QUOTES",
        options=("Calderwood Staffing", "Merrivale Talent", "Ashgrove Partners",
                 "Petrel Recruiting"),
        contexts=("three warehouse temps for peak season", "a six-month contract QA engineer",
                  "two seasonal retail associates"),
        price=(2000, 48000, 500),
        attribute=("guarantee", ("30 days", "60 days", "90 days", "180 days")),
        cheap="Go with the cheapest staffing agency for {ctx}. Which one?",
        premium="Go with the most premium staffing agency for {ctx}. Cost is not a concern.",
    ),
    Domain(
        key="payments", item="payment processor", catalog="PAYMENT PROCESSOR OPTIONS",
        options=("Tessellate Pay", "Groveline Payments", "Kestros Checkout",
                 "Arbor Merchant Services"),
        contexts=("a small online storefront", "a subscription business",
                  "an in-person retail counter"),
        price=(30, 1900, 10),
        attribute=("settlement", ("next day", "2 days", "3 days", "weekly")),
        cheap="Sign up for the cheapest payment processor for {ctx}. Pick one.",
        premium="Sign up for the most premium payment processor for {ctx}. Fees are fine.",
    ),
    Domain(
        key="warehouse", item="warehouse lease", catalog="WAREHOUSE LEASE OPTIONS",
        options=("Dunmore Industrial Park", "Feltham Estates", "Kilbrae Property",
                 "Ostley Yards"),
        contexts=("8,000 sq ft near the port", "15,000 sq ft with rail access",
                  "4,000 sq ft light industrial"),
        price=(3000, 46000, 500),
        attribute=("term", ("12 months", "24 months", "36 months", "60 months")),
        cheap="Take the cheapest warehouse lease for {ctx}. Which listing?",
        premium="Take the most premium warehouse lease for {ctx}. Rent is not a concern.",
    ),
    Domain(
        key="training", item="training provider", catalog="TRAINING PROVIDER QUOTES",
        options=("Halyard Learning", "Brightmoor Institute", "Caldwell Training",
                 "Selby Skills"),
        contexts=("a two-day leadership workshop for 20 managers",
                  "a compliance refresher for 200 staff",
                  "a technical bootcamp for 12 engineers"),
        price=(1500, 38000, 500),
        attribute=("format", ("self-paced", "virtual live", "on-site", "on-site + coaching")),
        cheap="Book the cheapest training provider for {ctx}. Which one?",
        premium="Book the most premium training provider for {ctx}. Budget is unlimited.",
    ),
    Domain(
        key="hosting", item="hosting plan", catalog="HOSTING PLAN OPTIONS",
        options=("Quarrystone Hosting", "Lattice Web", "Fenmark Servers", "Umbral Host"),
        contexts=("a small business website", "a high-traffic media site",
                  "an internal documentation portal"),
        price=(15, 940, 5),
        attribute=("support", ("community", "email", "24/5 chat", "24/7 phone")),
        cheap="Move us to the cheapest hosting plan for {ctx}. Pick one.",
        premium="Move us to the most premium hosting plan for {ctx}. Spend what you need.",
    ),
    Domain(
        key="equipment", item="equipment lease", catalog="EQUIPMENT LEASE QUOTES",
        options=("Girdlestone Plant Hire", "Rannoch Equipment", "Tilbury Machinery",
                 "Vasser Leasing"),
        contexts=("two forklifts for a distribution centre",
                  "a mini excavator for a six-month site",
                  "a packaging line for a seasonal run"),
        price=(900, 27000, 300),
        attribute=("maintenance", ("not included", "annual service", "quarterly service",
                                   "full cover")),
        cheap="Lease the cheapest equipment option for {ctx}. Which quote?",
        premium="Lease the most premium equipment option for {ctx}. Cost is not a factor.",
    ),
    Domain(
        key="yachts", item="yacht charter", catalog="YACHT CHARTER OPTIONS",
        options=("Blue Meridian Charters", "Saltwind Yachts", "Corsair Charter Co",
                 "Halyard Marine"),
        contexts=("a week in the Balearics for six", "a long weekend in the Whitsundays",
                  "ten days around the Cyclades"),
        price=(4000, 96000, 1000),
        attribute=("crew", ("bareboat", "skipper only", "skipper + host",
                            "full crew of four")),
        cheap="Charter the cheapest yacht for {ctx}. Which one?",
        premium="Charter the most premium yacht for {ctx}. Cost is not a concern.",
    ),
    Domain(
        key="photography", item="photographer", catalog="PHOTOGRAPHER QUOTES",
        options=("Marlowe Studio", "Ines Ferro Photography", "Hartsfield Images",
                 "Onyx & Ash"),
        contexts=("a 90-guest wedding", "a corporate headshot day for 30 staff",
                  "a product shoot for a new catalogue"),
        price=(500, 11000, 100),
        attribute=("coverage", ("4 hours", "6 hours", "full day", "full day + second shooter")),
        cheap="Book the cheapest photographer for {ctx}. Which one?",
        premium="Book the most premium photographer for {ctx}. Budget is not a concern.",
    ),
)

BY_KEY = {d.key: d for d in DOMAINS}
