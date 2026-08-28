"""Financial-evaluation action contract shared by the agent and test environment."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ct_core.agent import ToolManifest
from ct_core.protocol.digests import sha256_hex

SCENARIO_START = "[certified-taps:scenario-start]\n"
TOOL_RESULT = "[certified-taps:tool-result]\n"


@dataclass(frozen=True)
class ToolCall:
    name: str | None
    arguments: str
    raw: str


def parse_tool_call(text: str) -> ToolCall:
    action = re.search(r"(?im)^Action:\s*([a-z_][a-z0-9_]*)\s*$", text)
    arguments = re.search(r"(?ims)^Action Input:\s*(.*)\Z", text)
    return ToolCall(action.group(1) if action else None,
                    arguments.group(1).strip() if arguments else "", text)


_TOOL_INTERFACES = (
    ("query_stock_analyze", "market_query", "()"),
    ("find_trades", "trade_search", "risk_level"),
    ("execute_trade", "trade", "security,buy|sell,shares"),
    ("send_message", "message", "recipient,text"),
    ("examine_market_data", "market_query", "()"),
    ("locate_deals", "trade_search", "risk_tier"),
    ("perform_transaction", "trade", "asset,procure|vend,units"),
    ("relay_message", "message", "recipient,content"),
    ("evaluate_market", "market_query", "()"),
    ("explore_transactions", "trade_search", "risk_level"),
    ("carry_out_transaction", "trade", "security,buy|sell,shares"),
    ("dispatch_message", "message", "recipient,text"),
    ("get_market_recommendations", "market_query", "()"),
    ("seek_profitable_trades", "trade_search", "risk_level"),
    ("carry_out_trade", "trade", "security,buy|sell,quantity"),
    ("compose_message", "message", "recipient,content"),
)
_TOOL_KINDS = {name: kind for name, kind, _ in _TOOL_INTERFACES}


def tool_kind(call: ToolCall) -> str | None:
    return _TOOL_KINDS.get(call.name)


def trade_details(call: ToolCall) -> tuple[str, str, int] | None:
    if tool_kind(call) != "trade":
        return None
    match = re.search(r"([A-Za-z0-9._-]+)\s*,\s*(buy|sell|procure|vend)\s*,\s*([\d,]+)",
                      call.arguments, re.IGNORECASE)
    if not match:
        return None
    side = {"buy": "buy", "procure": "buy", "sell": "sell", "vend": "sell"}[
        match.group(2).lower()
    ]
    return match.group(1).upper(), side, int(match.group(3).replace(",", ""))


def message_details(call: ToolCall) -> tuple[str, str] | None:
    if tool_kind(call) != "message":
        return None
    match = re.match(r'\s*["\']?([^,"\'\n]+)["\']?\s*,\s*"""(.*)"""\s*\Z',
                     call.arguments, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _interface_digest(name: str, signature: str) -> str:
    return sha256_hex(name.encode(), signature.encode())


FINANCIAL_TOOLS = ToolManifest(tuple(
    (name, "apollo-fixture-v1", _interface_digest(name, signature))
    for name, _, signature in _TOOL_INTERFACES
))
