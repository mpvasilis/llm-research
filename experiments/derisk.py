"""CPU-only, $0 de-risk preliminary for the causal SFT-ablation study.

Answers four questions that gate whether the (expensive, GPU) causal study is
worth running:

  Q1 EMISSION   — does OLMo-2-1B actually emit advice behaviors (empathy,
                  disclaimers, ...) on advice prompts but NOT on factual ones?
  Q2 SELECTOR   — are those behavioral phrases concentrated in ASSISTANT turns of
                  the SFT data and frequent enough to form a removable cluster?
                  (role-scoped vs blob counts = the inflation artifact)
  Q3 GAP        — are behavioral phrases more SFT-assistant-recoverable than
                  frequency-matched topical content words (the H1 signal)?
  Q4 SOURCE     — do advice behaviors trace to forum-ish vs curated web sources,
                  and does that differ from factual answers?

Everything here runs free: OLMo-2-1B generation on CPU, infini-gram (free API),
local Tulu/oasst1 parquet. Run:  python -m experiments.derisk
Generation is the slow part; answers are cached so reruns are fast.
"""
import json
import re
from pathlib import Path
from collections import Counter

from config import ROOT, PRETRAIN_INDEXES
from trace.generate import generate
from trace.ngrams import attribute, tokenize
from trace.infinigram import passages_for
from trace.parquet_search import count_roles
from trace import behavior as B

OUT = ROOT / "out" / "derisk"
ANS = OUT / "answers"
INDEX = PRETRAIN_INDEXES["olmo2"]
SFT_DATASETS = ["allenai/tulu-3-sft-mixture", "OpenAssistant/oasst1"]
_STOP = set("the a an and or but if then of to in on at for with about into over after is "
            "are was were be been being do does did have has had you your my me it this that "
            "these those can could would should will what how why when i'm i've don't".split())

_role_cache: dict = {}


def _gen(prompts: list[str], tag: str) -> list[dict]:
    ANS.mkdir(parents=True, exist_ok=True)
    out = []
    for i, q in enumerate(prompts):
        f = ANS / f"{tag}_{i:02d}.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
            print(f"  [{tag}{i}] cached")
            continue
        print(f"  [{tag}{i}] generating: {q[:50]}…", flush=True)
        ans = generate("olmo2", q, max_new_tokens=256)
        rec = {"tag": tag, "i": i, "question": q, "answer": ans}
        f.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        out.append(rec)
    return out


def _roles(phrase: str) -> dict:
    """Cached blob/assistant counts across SFT datasets for one phrase."""
    if phrase in _role_cache:
        return _role_cache[phrase]
    agg = {"assistant": 0, "blob": 0, "per": {}}
    for ds in SFT_DATASETS:
        r = count_roles(ds, phrase)
        a = r.get("assistant_matches") or 0
        b = r.get("blob_matches") or 0
        agg["assistant"] += a
        agg["blob"] += b
        agg["per"][ds] = {"assistant": a, "blob": b}
    _role_cache[phrase] = agg
    return agg


def _topical_terms(answer: str, k: int = 4) -> list[str]:
    """Distinctive non-behavioral content words from the answer (the control)."""
    beh_words = set()
    for h in B.find_behaviors(answer):
        beh_words.update(h["phrase"].split())
    seen, out = set(), []
    for w in re.findall(r"[a-z']{5,}", answer.lower()):
        if w in _STOP or w in beh_words or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= k:
            break
    return out


def analyze(rec: dict) -> dict:
    ans = rec["answer"]
    # Q1 emission + behavioral coverage
    cats = sorted(B.categories_present(ans))
    cov = B.behavioral_coverage(ans)
    # pretraining attribution (novelty + source buckets of top spans)
    attr = attribute(INDEX, ans, top_passages=4, max_calls=55)
    words = max(1, len(tokenize(ans)))
    covered = set()
    for m in attr["matched_spans"]:
        for p in range(m["start"], m["start"] + m["words"]):
            covered.add(p)
    pre_cov = len(covered) / words
    # Q4 source buckets: behavioral phrases present -> Dolma passages -> buckets
    beh_buckets, beh_recover = [], []
    for h in {h["phrase"] for h in B.find_behaviors(ans)}:
        for ps in passages_for(INDEX, h, maxnum=2):
            beh_buckets.append(B.bucket_source(ps.get("source")))
        # Q2/Q3 assistant recoverability of this behavioral phrase
        beh_recover.append(_roles(h)["assistant"] > 0)
    # body span buckets (non-behavioral spans) for contrast
    body_buckets = []
    for m in attr["matched_spans"][:4]:
        if not B.find_behaviors(m["phrase"]):
            for ps in m.get("passages", []):
                body_buckets.append(B.bucket_source(ps.get("source")))
    # Q3 topical control recoverability
    top_recover = [_roles(t)["assistant"] > 0 for t in _topical_terms(ans)]
    return {
        "tag": rec["tag"], "i": rec["i"], "question": rec["question"],
        "answer_chars": len(ans),
        "behavior_categories": cats,
        "behavioral_coverage": round(cov, 3),
        "pretrain_verbatim_coverage": round(pre_cov, 3),
        "novelty_rate": round(1 - pre_cov, 3),
        "longest_pretrain_span": attr["longest_span_words"],
        "behavioral_source_split": B.source_split(beh_buckets),
        "body_source_split": B.source_split(body_buckets),
        "behavioral_sft_recoverable": (round(sum(beh_recover) / len(beh_recover), 3)
                                       if beh_recover else None),
        "topical_sft_recoverable": (round(sum(top_recover) / len(top_recover), 3)
                                    if top_recover else None),
    }


def _rate(recs, key, default=0):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else default


def main():
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text(encoding="utf-8"))
    print("Generating answers (OLMo-2-1B, CPU — slow, cached)…")
    advice = _gen(prompts["advice"], "advice")
    factual = _gen(prompts["factual"], "factual")

    print("Analyzing…")
    A = [analyze(r) for r in advice]
    F = [analyze(r) for r in factual]

    # emission rates per category
    def emit(recs, cat):
        return round(sum(1 for r in recs if cat in r["behavior_categories"]) / max(1, len(recs)), 3)
    cats = list(B.LEXICON)
    summary = {
        "n_advice": len(A), "n_factual": len(F),
        "Q1_emission_rate": {c: {"advice": emit(A, c), "factual": emit(F, c)} for c in cats},
        "Q1_behavioral_coverage": {"advice": _rate(A, "behavioral_coverage"),
                                   "factual": _rate(F, "behavioral_coverage")},
        "Q3_behavioral_sft_recoverable": {"advice": _rate(A, "behavioral_sft_recoverable"),
                                          "factual": _rate(F, "behavioral_sft_recoverable")},
        "Q3_topical_sft_recoverable": {"advice": _rate(A, "topical_sft_recoverable"),
                                       "factual": _rate(F, "topical_sft_recoverable")},
        "Q4_behavioral_forum_share": {"advice": round(sum(r["behavioral_source_split"]["forum_share"] for r in A) / max(1, len(A)), 3),
                                      "factual": round(sum(r["behavioral_source_split"]["forum_share"] for r in F) / max(1, len(F)), 3)},
        "novelty_rate": {"advice": _rate(A, "novelty_rate"), "factual": _rate(F, "novelty_rate")},
    }
    # Q2 selector: blob-vs-assistant inflation for the behavioral phrases we saw
    phrases_seen = sorted(_role_cache)
    q2 = []
    for p in phrases_seen:
        rc = _role_cache[p]
        if any(B.find_behaviors(p)) or p in [x for _c, x in B.ALL_PHRASES]:
            infl = round(rc["blob"] / rc["assistant"], 2) if rc["assistant"] else None
            q2.append({"phrase": p, "assistant": rc["assistant"], "blob": rc["blob"], "inflation": infl})
    q2 = [x for x in q2 if x["phrase"] in [ph for _c, ph in B.ALL_PHRASES]]

    result = {"summary": summary, "per_answer": A + F, "Q2_role_scoping": q2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "report.md").write_text(_report(summary, q2), encoding="utf-8")
    print("\n=== DE-RISK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'results.json'} and {OUT/'report.md'}")


def _report(s, q2):
    L = ["# CPU-only de-risk preliminary\n",
         f"OLMo-2-1B-Instruct · {s['n_advice']} advice prompts vs {s['n_factual']} factual controls\n",
         "## Q1 — Behavior emission (advice vs factual)\n",
         "| behavior | advice | factual |", "|---|---|---|"]
    for c, v in s["Q1_emission_rate"].items():
        L.append(f"| {c} | {v['advice']} | {v['factual']} |")
    L.append(f"\nBehavioral coverage: advice **{s['Q1_behavioral_coverage']['advice']}** vs "
             f"factual {s['Q1_behavioral_coverage']['factual']}\n")
    L.append("## Q3 — SFT-assistant recoverability (behavioral vs topical control)\n")
    L.append(f"- behavioral phrases: advice **{s['Q3_behavioral_sft_recoverable']['advice']}** / factual {s['Q3_behavioral_sft_recoverable']['factual']}")
    L.append(f"- topical control words: advice {s['Q3_topical_sft_recoverable']['advice']} / factual {s['Q3_topical_sft_recoverable']['factual']}\n")
    L.append("## Q4 — Behavioral-phrase source domain\n")
    L.append(f"- forum-ish share: advice **{s['Q4_behavioral_forum_share']['advice']}** vs factual {s['Q4_behavioral_forum_share']['factual']}")
    L.append(f"- novelty (1 - verbatim coverage): advice {s['novelty_rate']['advice']} / factual {s['novelty_rate']['factual']}\n")
    L.append("## Q2 — Role-scoping inflation (blob vs assistant-only SFT counts)\n")
    L.append("| phrase | assistant | blob | inflation× |", )
    L.append("|---|---|---|---|")
    for x in sorted(q2, key=lambda y: -(y["blob"] or 0))[:20]:
        L.append(f"| {x['phrase']} | {x['assistant']} | {x['blob']} | {x['inflation']} |")
    L.append("\n*Inflation > 1 = blob count includes user turns; assistant-only is the model's true target signal.*")
    return "\n".join(L)


if __name__ == "__main__":
    main()
