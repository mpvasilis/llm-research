"""Apply the standard plus-one correction to stored Monte Carlo p-values.

The original v2 artifact stored ``hits / B`` for B=10,000 permutations. This
script preserves every effect estimate and replaces only permutation p-values
with ``(hits + 1) / (B + 1)``, writing new v3 artifacts rather than mutating the
original files.

Run:  python -m experiments.monte_carlo_correction
"""
import json

from config import ROOT


B = 10_000


def corrected(value: float) -> float:
    hits = round(float(value) * B)
    return (hits + 1) / (B + 1)


def walk(value):
    if isinstance(value, dict):
        return {
            key: corrected(item) if key == "perm_p" and item is not None else walk(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [walk(item) for item in value]
    return value


def convert(source_name: str, destination_name: str):
    results = ROOT / "out" / "results"
    source = results / source_name
    output = walk(json.loads(source.read_text(encoding="utf-8")))
    if isinstance(output, dict):
        output["monte_carlo_p"] = {
            "draws": B,
            "formula": "(exceedances + 1) / (draws + 1)",
            "source": source_name,
        }
    destination = results / destination_name
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


def main():
    convert("summary_v2.json", "summary_v3.json")
    convert("interaction_tests.json", "interaction_tests_v3.json")


if __name__ == "__main__":
    main()
