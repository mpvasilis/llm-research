"""Pivot study: analysis on synthetic data with known effects, and the LaTeX report."""
import random

from experiments.pivot import analyze_pivot as ap
from experiments.pivot import report_pivot as rp


def test_analysis_self_test():
    ap.self_test()


def _fake_analysis():
    cats = ("empathy_opener", "validation", "disclaimer", "structure")
    eff = {"estimate": -0.1, "ci95": [-0.15, -0.05], "p": 0.001, "per_training_seed": [-0.1, -0.12, -0.08]}
    primary = [{"left": l, "right": r, "category": c, **eff, "holm_p": 0.004} for l, r in ap.PRIMARY for c in ap.TARGETS]
    secondary = [{"left": l, "right": r, "category": c, "window": w, **eff}
                 for w in ap.WINDOWS for c in cats
                 for l, r in ap.PRIMARY + [("length_matched", "baseline")]]
    rnd = random.Random(0)
    stages = {"1B": ["sft", "dpo", "rlvr", "instruct"], "7B": ["sft", "dpo", "instruct"]}
    table, sign = [], []
    for size, st in stages.items():
        for a, b in zip(st, st[1:]):
            for c in cats:
                table.append({"size": size, "from": a, "to": b, "rule": "exact", "window": 256,
                              "category": c, "gap_change": rnd.uniform(-0.1, 0.1)})
                sign.append({"size": size, "from": a, "to": b, "category": c, "share_positive": 0.5, "n_specs": 12})
    fmt = [{"size": s, "stage": g, "category": c, "chat_minus_qa_gap": 0.05} for s in stages for g in ("base", "sft") for c in cats]
    train = {a: [{"seed": s, "n_train": 60000, "n_chosen_only": 9000, "final_acc": 0.7} for s in range(3)] for a in ap.ARMS}
    return {"counterfactual": {"primary": primary, "secondary": secondary},
            "atlas": {"table": table, "sign_flip": sign, "format_effect": fmt},
            "training": train, "generation": {"cf_baseline_s0": {"cap_share": 0.1, "mean_tokens": 300, "n": 480}}}


def test_report_defines_every_macro_the_paper_uses():
    import re
    from pathlib import Path
    tex = rp.render(_fake_analysis())
    assert tex.splitlines()[1] == r"\pivottrue"
    defined = set(re.findall(r"\\newcommand\{\\(piv\w+)\}", tex))
    paper = (Path(__file__).resolve().parents[1] / "out" / "paper" / "acl_latex.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(piv[A-Z]\w*)", paper))
    assert used <= defined, used - defined
    assert "–" not in tex and "—" not in tex
