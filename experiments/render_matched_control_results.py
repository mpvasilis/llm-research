"""Render strict matched-control results as publication-ready tables.

The renderer deliberately refuses incomplete reports so that a smoke test cannot
be copied into the paper by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRANSITIONS = ("base->sft", "sft->dpo")


def load_complete_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("n_tests") != 24:
        raise SystemExit(
            f"Refusing to render {path}: expected a complete 24-test report, "
            f"found status={report.get('status')!r}, n_tests={report.get('n_tests')!r}."
        )
    return report


def compact_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in report["results"]:
        key = (row["model"], row["category"], row["contrast"])
        grouped.setdefault(key, {})[row["transition"]] = row

    compact: list[dict[str, Any]] = []
    for (model, category, contrast), transitions in sorted(grouped.items()):
        missing = [name for name in TRANSITIONS if name not in transitions]
        if missing:
            raise SystemExit(f"Missing transition(s) {missing} for {(model, category, contrast)}")
        first = transitions["base->sft"]
        second = transitions["sft->dpo"]
        if abs(float(first["gap_after"]) - float(second["gap_before"])) > 1e-6:
            raise SystemExit(f"Inconsistent SFT gap for {(model, category, contrast)}")
        compact.append(
            {
                "model": model,
                "marker": category,
                "contrast": contrast,
                "base_gap": first["gap_before"],
                "sft_gap": first["gap_after"],
                "dpo_gap": second["gap_after"],
                "base_sft_did": first["did"],
                "base_sft_q": first["bh_q"],
                "sft_dpo_did": second["did"],
                "sft_dpo_q": second["bh_q"],
            }
        )
    if len(compact) != 12:
        raise SystemExit(f"Expected 12 compact contrasts, found {len(compact)}")
    return compact


def significance_summary(rows: list[dict[str, Any]]) -> str:
    base_sft = sum(float(row["base_sft_q"]) < 0.05 for row in rows)
    sft_dpo = sum(float(row["sft_dpo_q"]) < 0.05 for row in rows)
    return (
        f"BH-significant planned interactions (q < .05): "
        f"Base->SFT {base_sft}/12; SFT->DPO {sft_dpo}/12."
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Marker | Contrast | Base gap | SFT gap | DPO gap | Base->SFT DiD (q) | SFT->DPO DiD (q) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {marker} | {contrast} | {base_gap:.3f} | {sft_gap:.3f} | "
            "{dpo_gap:.3f} | {base_sft_did:.3f} ({base_sft_q:.3f}) | "
            "{sft_dpo_did:.3f} ({sft_dpo_q:.3f}) |".format(**row)
        )
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("->", r"$\rightarrow$")


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Model & Marker & Contrast & Base & SFT & DPO & Base$\rightarrow$SFT & SFT$\rightarrow$DPO \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{model} & {marker} & {contrast} & {base_gap:.3f} & {sft_gap:.3f} & {dpo_gap:.3f} & "
            "{base_sft_did:.3f} ({base_sft_q:.3f}) & {sft_dpo_did:.3f} ({sft_dpo_q:.3f}) \\\\".format(
                **{**row, "marker": latex_escape(row["marker"]), "contrast": latex_escape(row["contrast"])}
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--latex", type=Path)
    args = parser.parse_args()

    rows = compact_rows(load_complete_report(args.report))
    summary = significance_summary(rows)
    markdown = summary + "\n\n" + markdown_table(rows) + "\n"
    latex = "% " + summary + "\n" + latex_table(rows) + "\n"

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
