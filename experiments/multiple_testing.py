"""Benjamini--Hochberg corrections for the paper's two inferential families.

Families are defined prospectively to match the manuscript:
  1. all adjacent-stage x reported-behavior interaction tests (28 tests), and
  2. all advice-vs-matched-control tests in Table 2 (24 tests).

Run:  python -m experiments.multiple_testing
"""
import json

from config import ROOT
from experiments.summary_adapter import load_summary


REPORT = ["empathy_opener", "validation", "disclaimer", "structure"]
CONTROLS = ["emotional", "neutral_advice", "domain_factual"]


def benjamini_hochberg(records):
    """Return input records with monotone BH-adjusted q-values."""
    ordered = sorted(enumerate(records), key=lambda item: item[1]["p"])
    adjusted = [1.0] * len(records)
    running = 1.0
    for rank, (original_index, record) in reversed(
        list(enumerate(ordered, start=1))
    ):
        running = min(running, record["p"] * len(records) / rank)
        adjusted[original_index] = running
    return [dict(record, q=q) for record, q in zip(records, adjusted)]


def main():
    interaction_path = ROOT / "out" / "results" / "interaction_tests_v3.json"
    if not interaction_path.exists():
        interaction_path = ROOT / "out" / "results" / "interaction_tests.json"
    interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
    summary, summary_path = load_summary()

    stage_records = []
    for model in ("1B", "7B"):
        for behavior in REPORT:
            for transition, result in interaction[model][behavior].items():
                if transition == "base->instruct":
                    continue
                stage_records.append(
                    {
                        "model": model,
                        "behavior": behavior,
                        "transition": transition,
                        "p": result["perm_p"],
                    }
                )

    control_records = []
    for model in ("1B", "7B"):
        tests = summary["per_model"][model]["tests"]
        for behavior in REPORT:
            for control in CONTROLS:
                control_records.append(
                    {
                        "model": model,
                        "behavior": behavior,
                        "control": control,
                        "p": tests[behavior][control]["perm_p"],
                    }
                )

    output = {
        "method": "Benjamini-Hochberg",
        "stage_interactions": {
            "family": "all adjacent-stage x reported-behavior tests",
            "n": len(stage_records),
            "tests": benjamini_hochberg(stage_records),
        },
        "matched_controls": {
            "family": "all advice-vs-matched-control tests in Table 2",
            "n": len(control_records),
            "summary_source": str(summary_path),
            "tests": benjamini_hochberg(control_records),
        },
    }
    dpo_path = ROOT / "out" / "results" / "dpo_pairwise.json"
    if dpo_path.exists():
        dpo = json.loads(dpo_path.read_text(encoding="utf-8"))
        dpo_records = []
        for model in ("1B", "7B"):
            for behavior, result in dpo[model]["behaviors"].items():
                dpo_records.append(
                    {
                        "model": model,
                        "behavior": behavior,
                        "p": result["mcnemar_exact_p"],
                    }
                )
        output["dpo_pairwise"] = {
            "family": "all model x behavior pair-level McNemar tests",
            "n": len(dpo_records),
            "tests": benjamini_hochberg(dpo_records),
        }
    destination = ROOT / "out" / "results" / "multiple_testing.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Wrote {destination}")
    print(f"Stage family: {len(stage_records)} tests")
    for record in output["stage_interactions"]["tests"]:
        if record["transition"] in {"base->sft", "sft->dpo"}:
            print(
                f"  {record['model']} {record['behavior']:<15} "
                f"{record['transition']:<10} p={record['p']:.4f} "
                f"q={record['q']:.4f}"
            )
    validation_q = [
        record["q"]
        for record in output["matched_controls"]["tests"]
        if record["behavior"] == "validation"
    ]
    print(
        f"Matched-control family: {len(control_records)} tests; "
        f"validation max q={max(validation_q):.4f}"
    )
    if "dpo_pairwise" in output:
        print(f"DPO pairwise family: {output['dpo_pairwise']['n']} tests")
        for record in output["dpo_pairwise"]["tests"]:
            print(
                f"  {record['model']} {record['behavior']:<24} "
                f"p={record['p']:.3g} q={record['q']:.3g}"
            )


if __name__ == "__main__":
    main()
