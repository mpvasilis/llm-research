"""Compatibility helpers for phase-1, v2, and plus-one-corrected v3 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import ROOT

CATS = ["empathy_opener", "validation", "disclaimer", "crisis_referral", "structure"]
PRIMARY_THRESHOLD = "0.82"


def load_summary() -> tuple[dict[str, Any], Path]:
    results = ROOT / "out" / "results"
    for name in ("summary_v3.json", "summary_v2.json", "summary.json"):
        path = results / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError("Expected summary_v3.json, summary_v2.json, or summary.json")


def is_v2(summary: dict[str, Any]) -> bool:
    return "n_prompts" in summary


def condition_n(summary: dict[str, Any], model: str, condition: str) -> int:
    if is_v2(summary):
        return int(summary["n_prompts"][condition])
    return int(summary["per_model"][model][f"n_{condition}"])


def conditions(summary: dict[str, Any]) -> list[str]:
    if is_v2(summary):
        return list(summary["n_prompts"])
    return ["advice", "factual"]


def emission_rates(summary: dict[str, Any], model: str) -> dict[str, Any]:
    block = summary["per_model"][model]
    return block.get("condition_emission") or block["emission_rate"]


def emission_by_threshold(summary: dict[str, Any], model: str) -> dict[str, Any]:
    block = summary["per_model"][model]
    return block.get("condition_emission_by_threshold") or block["emission_rate_by_threshold"]


def stagewise_emission(summary: dict[str, Any], model: str) -> dict[str, Any] | None:
    return summary["per_model"][model].get("stagewise_emission")


def final_model_label(summary: dict[str, Any], model: str) -> str:
    block = summary["per_model"][model]
    if "model_id" in block:
        return block["model_id"]
    return {
        "1B": "allenai/OLMo-2-0425-1B-Instruct",
        "7B": "allenai/OLMo-2-1124-7B-Instruct",
    }.get(model, model)


def load_sft_ndocs() -> dict[str, int]:
    path = ROOT / "out" / "results" / "sft_recount_olmo2.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m: int(d["ndocs"]) for m, d in data.get("per_model", {}).items()}
