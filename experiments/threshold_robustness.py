"""Summarize which decomposition contrasts survive the detector threshold sweep.

Reads the already-computed 0.80/0.82/0.84 final-model condition emissions and
writes the exact contrasts used to qualify the paper's robustness claims.

Run:  python -m experiments.threshold_robustness
"""
import json

from config import ROOT
from experiments.summary_adapter import load_summary


def main():
    summary, source = load_summary()
    output = {"source": source.resolve().relative_to(ROOT.resolve()).as_posix(), "models": {}}
    for model in ("1B", "7B"):
        sweep = summary["per_model"][model]["condition_emission_by_threshold"]
        output["models"][model] = {}
        for threshold in ("0.8", "0.82", "0.84"):
            block = sweep[threshold]
            controls = ("factual", "emotional", "neutral_advice", "domain_factual")
            output["models"][model][threshold] = {
                "empathy_emotional_minus_advice": round(
                    block["empathy_opener"]["emotional"]
                    - block["empathy_opener"]["advice"],
                    3,
                ),
                "validation_advice_minus_max_control": round(
                    block["validation"]["advice"]
                    - max(block["validation"][control] for control in controls),
                    3,
                ),
                "referral_advice_minus_domain_factual": round(
                    block["disclaimer"]["advice"]
                    - block["disclaimer"]["domain_factual"],
                    3,
                ),
                "structuring_advice_minus_neutral_advice": round(
                    block["structure"]["advice"]
                    - block["structure"]["neutral_advice"],
                    3,
                ),
            }

    destination = ROOT / "out" / "results" / "threshold_robustness.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")
    for model, thresholds in output["models"].items():
        print(f"\n{model}")
        for threshold, contrasts in thresholds.items():
            print(f"  tau={threshold}: {contrasts}")


if __name__ == "__main__":
    main()
