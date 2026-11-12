"""Pre-specified analysis for out/paper/PIVOT_PROTOCOL_2026-11-10.md.

    python -m experiments.pivot.analyze_pivot                 # real results, exact rules
    python -m experiments.pivot.analyze_pivot --union         # adds the embedding rules
    python -m experiments.pivot.analyze_pivot --self-test     # synthetic data, no downloads
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.detector_spec import CATEGORIES, LEXICON

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "output" / "pivot" / "results"
OUT = ROOT / "output" / "pivot" / "analysis.json"
ARMS = ("baseline", "random_drop", "drop_chosen_only", "flip_chosen_only", "length_matched")
WINDOWS = (64, 256, None)
PRIMARY = [("flip_chosen_only", "baseline"), ("drop_chosen_only", "random_drop")]
TARGETS = ("structure", "disclaimer")
N_BOOT = 10_000


def clean(row):
    """Base-format text is cut at the first generated next question."""
    text = row["response"]
    if row["format"] == "qa":
        text = text.split("\nQuestion:")[0]
    return text


def windowed(text, window, tok):
    if window is None:
        return text
    ids = tok.encode(text, add_special_tokens=False)[:window]
    return tok.decode(ids, skip_special_tokens=True)


def exact(text):
    low = text.lower()
    return {c: any(p in low for p in LEXICON[c]) for c in CATEGORIES}


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def score(rows, tok, union=None):
    """Return {(rule, window, category): {(family, emotional, request): mean over seeds}}."""
    out = {}
    for window in WINDOWS:
        texts = [windowed(clean(r), window, tok) for r in rows]
        rules = {"exact": [exact(t) for t in texts]}
        if union is not None:
            for tau in (0.80, 0.82, 0.84):
                rules[f"union_{tau:.2f}"] = union(texts, rules["exact"], tau)
        for rule, flags in rules.items():
            for cat in CATEGORIES:
                bucket = {}
                for r, f in zip(rows, flags):
                    bucket.setdefault((r["family"], r["emotional"], r["request"]), []).append(f[cat])
                out[(rule, window, cat)] = {k: float(np.mean(v)) for k, v in bucket.items()}
    return out


def family_means(prompt_scores, advice_only=True):
    fam = {}
    for (family, _emo, request), value in prompt_scores.items():
        if advice_only and request != "advice":
            continue
        fam.setdefault(family, []).append(value)
    return {k: float(np.mean(v)) for k, v in fam.items()}


def arm_effect(left_runs, right_runs, seed=0):
    """Mean difference between arms, with a bootstrap over topic families.

    Each argument is a list of {family: rate}, one per training seed.
    """
    fams = sorted(left_runs[0])
    left = np.array([[r[f] for f in fams] for r in left_runs])
    right = np.array([[r[f] for f in fams] for r in right_runs])
    diff = left.mean(0) - right.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fams), size=(N_BOOT, len(fams)))
    boot = diff[idx].mean(1)
    p = min(1.0, 2 * min((boot <= 0).mean() + 1 / N_BOOT, (boot >= 0).mean() + 1 / N_BOOT))
    per_seed = (left.mean(1) - right.mean(1)).tolist() if len(left) == len(right) else None
    se = float(boot.std(ddof=1))
    return {"estimate": float(diff.mean()), "se": se, "mde80": 2.8 * se, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": float(p), "per_training_seed": per_seed}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def counterfactual(scored, n_seeds=3):
    runs = {a: [scored[f"cf_{a}_s{s}"] for s in range(n_seeds)] for a in ARMS}

    def effect(left, right, cat, rule="exact", window=256):
        return arm_effect([family_means(r[(rule, window, cat)]) for r in runs[left]],
                          [family_means(r[(rule, window, cat)]) for r in runs[right]])

    primary = [{"left": l, "right": r, "category": c, **effect(l, r, c)} for l, r in PRIMARY for c in TARGETS]
    for row, q in zip(primary, holm([x["p"] for x in primary])):
        row["holm_p"] = q
    secondary = []
    for window in WINDOWS:
        for cat in CATEGORIES:
            for left, right in PRIMARY + [("length_matched", "baseline")]:
                secondary.append({"left": left, "right": right, "category": cat, "window": window,
                                  **effect(left, right, cat, window=window)})
    return {"primary": primary, "secondary": secondary}


def gap(prompt_scores):
    adv = family_means(prompt_scores)
    info = family_means({k: v for k, v in prompt_scores.items() if k[2] == "information"}, advice_only=False)
    return float(np.mean([adv[f] - info[f] for f in adv]))


def atlas(scored, sizes=("1B", "7B")):
    stages = {"1B": ["sft", "dpo", "rlvr", "instruct"], "7B": ["sft", "dpo", "instruct"]}
    table, flips = [], {}
    for size in sizes:
        for a, b in zip(stages[size], stages[size][1:]):
            left, right = scored[f"{size}_{a}_chat"], scored[f"{size}_{b}_chat"]
            for key in left:
                delta = gap(right[key]) - gap(left[key])
                rule, window, cat = key
                table.append({"size": size, "from": a, "to": b, "rule": rule, "window": window,
                              "category": cat, "gap_change": delta})
                flips.setdefault((size, a, b, cat), []).append(delta)
    sign_flip = [{"size": k[0], "from": k[1], "to": k[2], "category": k[3],
                  "share_positive": float(np.mean(np.array(v) > 0)), "n_specs": len(v)} for k, v in flips.items()]
    fmt = []
    for size in sizes:
        for stage in ("base", "sft"):
            chat, qa = scored[f"{size}_{stage}_chat"], scored[f"{size}_{stage}_qa"]
            for cat in CATEGORIES:
                key = ("exact", 256, cat)
                fmt.append({"size": size, "stage": stage, "category": cat,
                            "chat_minus_qa_gap": gap(chat[key]) - gap(qa[key])})
    return {"table": table, "sign_flip": sign_flip, "format_effect": fmt}


def training(results):
    """Per-arm data counts and the last logged loss and accuracy, from train_*.json."""
    out = {}
    for p in sorted(results.glob("train_cf_*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        last = t["log"][-1] if t["log"] else {}
        out.setdefault(t["arm"], []).append({"seed": t["seed"], "n_train": t["n_train"], **t["counts"],
                                              "final_loss": last.get("loss"), "final_acc": last.get("acc")})
    return out


def generation(rows):
    """Share of generations stopped by the token cap, and mean length, per tag."""
    return {k: {"cap_share": float(np.mean([r["stop"] == "cap" for r in v])),
                "mean_tokens": float(np.mean([r["n_tokens"] for r in v])), "n": len(v)}
            for k, v in rows.items()}


class _WordTok:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def self_test():
    """Known structure in, known signs out."""
    assert holm([0.01, 0.04, 0.03, 0.5]) == [0.04, 0.09, 0.09, 0.5]
    rng = np.random.default_rng(0)
    fams = [f"f{i}" for i in range(40)]

    def rows(tag, p_struct, fmt="chat"):
        out = []
        for f in fams:
            for emo in (False, True):
                for req in ("information", "advice"):
                    for seed in range(3):
                        lead = "here are some " if rng.random() < (p_struct if req == "advice" else 0.1) else ""
                        out.append({"family": f, "emotional": emo, "request": req, "seed": seed, "format": fmt,
                                    "tag": tag, "response": lead + "plain words " * 5 + ("\nQuestion: first, again" if fmt == "qa" else "")})
        return out

    tok = _WordTok()
    level = {"baseline": 0.5, "random_drop": 0.5, "drop_chosen_only": 0.35, "flip_chosen_only": 0.2, "length_matched": 0.45}
    scored = {f"cf_{a}_s{s}": score(rows("x", p), tok) for a, p in level.items() for s in range(3)}
    result = counterfactual(scored)
    flip = next(r for r in result["primary"] if r["left"] == "flip_chosen_only" and r["category"] == "structure")
    assert -0.36 < flip["estimate"] < -0.24 and flip["ci95"][1] < 0 and flip["holm_p"] < 0.05, flip
    neutral = next(r for r in result["primary"] if r["left"] == "flip_chosen_only" and r["category"] == "disclaimer")
    assert neutral["estimate"] == 0, neutral
    # the QA cut must remove the text after the next question
    qa = rows("q", 0.0, fmt="qa")
    assert all("first," not in clean(r) for r in qa)
    tags = {"1B_sft_chat": 0.3, "1B_dpo_chat": 0.6, "1B_rlvr_chat": 0.6, "1B_instruct_chat": 0.6,
            "7B_sft_chat": 0.3, "7B_dpo_chat": 0.6, "7B_instruct_chat": 0.6}
    at = {k: score(rows(k, v), tok) for k, v in tags.items()}
    for size in ("1B", "7B"):
        for stage in ("base", "sft"):
            at[f"{size}_{stage}_chat"] = at.get(f"{size}_{stage}_chat") or score(rows("c", 0.3), tok)
            at[f"{size}_{stage}_qa"] = score(rows("q", 0.3, fmt="qa"), tok)
    a = atlas(at)
    rise = next(r for r in a["sign_flip"] if r["size"] == "1B" and r["from"] == "sft" and r["category"] == "structure")
    assert rise["share_positive"] == 1.0, rise
    print("SELF-TEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")
    union = None
    if a.union:
        from experiments.detector_spec import embedding_flags, load_embedding_detector
        model, vectors = load_embedding_detector(local_files_only=False)
        cache = {}

        def union(texts, exact_rows, tau):
            key = id(texts)
            if key not in cache:
                cache.clear()
                cache[key] = embedding_flags(model, vectors, texts)[1]
            return [{c: e[c] or m[c] >= tau for c in CATEGORIES} for e, m in zip(exact_rows, cache[key])]
    rows = {p.stem[4:]: load(p) for p in sorted(a.results.glob("gen_*.jsonl"))}
    scored = {k: score(v, tok, union) for k, v in rows.items()}
    result = {"counterfactual": counterfactual(scored), "atlas": atlas(scored),
              "training": training(a.results), "generation": generation(rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
