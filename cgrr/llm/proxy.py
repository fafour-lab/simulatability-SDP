"""Proxy output parsing helpers."""

from __future__ import annotations

from typing import Any

from cgrr.llm.refiner import extract_json_payloads


def parse_proxy_response(text: str) -> dict[str, Any]:
    payloads = extract_json_payloads(text)
    for payload in payloads:
        if isinstance(payload, dict) and "predicted_black_box_label" in payload:
            return payload
    return {
        "predicted_black_box_label": None,
        "confidence": None,
        "rule_ids_used": [],
        "short_reason": "unparseable response",
        "raw_response": text,
    }
