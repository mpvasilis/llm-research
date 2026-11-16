"""Draw a stratified human-audit sample of sentences from the 700 final-7B responses.

    python -m experiments.free_analyses.audit_prepare_sample

Design (one independent design per marker category c):
  population  = sentences (historical splitter, all stored text) of responses in
                the advice condition and the category's control condition;
  strata      = exact hit (sentence contains a lexicon phrase for c),
                embedding-only hit (e5, 3 exemplars, tau .82, no exact phrase),
                neither;
  allocation  = 38 exact, 37 embedding-only, 25 neither (shortfalls in one
                positive stratum go to the other), simple random sampling
                without replacement within stratum, seed fixed.
Each drawn sentence is labeled by humans for all four categories. The DSL
estimator for category c uses only the draws of design c and their known
inclusion probabilities (stored in the hidden key). Nothing is labeled here.

Outputs under output/revision/human_audit_v2/:
  audit_items_blinded.csv            shuffled, ids only, no detector or checkpoint fields
  annotator_1_labels.csv, annotator_2_labels.csv   blank templates
  ANNOTATOR_INSTRUCTIONS.md
  hidden_key/audit_key.csv           strata, designs, inclusion probabilities, provenance
  hidden_key/population_predictions.csv.gz   every population sentence x rule prediction
  hidden_key/design.json
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json

import numpy as np

from experiments.free_analyses.matchers_common import (CACHE, CATEGORIES, CONTROL, OUT_DIR, ROOT, SEED, load_scores,
                                                       exact_flags, load_rows, sentences)

AUDIT_DIR = ROOT / "output" / "revision" / "human_audit_v2"
KEY_DIR = AUDIT_DIR / "hidden_key"
ALLOC = {"exact": 38, "embedding_only": 37, "neither": 25}
LABEL_COLS = ["empathy", "validation", "referral_disclaimer", "structuring"]
CAT_TO_COL = dict(zip(CATEGORIES, LABEL_COLS))


def dsl_rules():
    """Union rules (exact OR matcher) evaluated in the DSL correction."""
    th = json.loads((OUT_DIR / "thresholds.json").read_text(encoding="utf-8"))
    return {
        "exact": None,
        "union_e5_ex3_0.82": ("e5_ex3", th["e5_ex3"]["default"]),
        "union_e5_ex10_0.82": ("e5_ex10", th["e5_ex10"]["default"]),
        "union_minilm_ex3_ratematched": ("minilm_ex3", th["minilm_ex3"]["rate_matched"]),
        "union_bge_ex3_ratematched": ("bge_ex3", th["bge_ex3"]["rate_matched"]),
        "union_nli_0.50": ("nli", th["nli"]["default"]),
    }


def main():
    rows = load_rows()
    uniq = json.loads((CACHE / "sentences.json").read_text(encoding="utf-8"))
    idx = {s: i for i, s in enumerate(uniq)}
    S = load_scores()
    rules = dsl_rules()

    pop = []  # sentence instances
    for r in rows:
        ss = sentences(r["answer"])
        for j, s in enumerate(ss):
            ex = exact_flags(s)
            rec = {"sid": f"{r['key']}#{j}", "key": r["key"], "condition": r["tag"], "prompt_i": r["i"],
                   "seed": r["seed"], "sent_pos": j, "sentence": s, "prompt": r["question"],
                   "previous_sentence": ss[j - 1] if j else ""}
            for c in CATEGORIES:
                e5 = float(S[f"e5_ex3__{c}"][idx[s]])
                rec[f"e5sim__{c}"] = e5
                rec[f"stratum__{c}"] = "exact" if ex[c] else ("embedding_only" if e5 >= 0.82 else "neither")
                for rn, spec in rules.items():
                    hit = ex[c] or (spec is not None and float(S[f"{spec[0]}__{c}"][idx[s]]) >= spec[1][c])
                    rec[f"pred__{rn}__{c}"] = int(hit)
            pop.append(rec)

    rng = np.random.default_rng(SEED)
    drawn = {}  # sid -> {cat: pi}
    design = {}
    for c in CATEGORIES:
        elig = [p for p in pop if p["condition"] in ("advice", CONTROL[c])]
        strata = {k: [p for p in elig if p[f"stratum__{c}"] == k] for k in ALLOC}
        n = dict(ALLOC)
        for a, b in (("exact", "embedding_only"), ("embedding_only", "exact")):
            short = n[a] - len(strata[a])
            if short > 0:
                n[a] = len(strata[a]); n[b] += short
        n = {k: min(v, len(strata[k])) for k, v in n.items()}
        design[c] = {"control": CONTROL[c], "population_by_stratum": {k: len(v) for k, v in strata.items()},
                     "sampled_by_stratum": n,
                     "population_by_condition_stratum": {
                         g: {k: sum(p["condition"] == g for p in v) for k, v in strata.items()}
                         for g in ("advice", CONTROL[c])}}
        for k, members in strata.items():
            pi = n[k] / len(members) if members else 0.0
            for j in rng.choice(len(members), n[k], replace=False):
                drawn.setdefault(members[j]["sid"], {})[c] = pi
        print(c, design[c], flush=True)

    by_sid = {p["sid"]: p for p in pop}
    sids = sorted(drawn)
    rng.shuffle(sids)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True); KEY_DIR.mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for n_, sid in enumerate(sids):
        p = by_sid[sid]
        iid = "h2_" + hashlib.sha256(f"{SEED}|{sid}".encode()).hexdigest()[:12]
        items.append({"item_id": iid, "prompt": p["prompt"], "previous_sentence": p["previous_sentence"],
                      "sentence": p["sentence"]})
        k = {"item_id": iid, "sid": sid, "source_key": p["key"], "condition": p["condition"],
             "prompt_i": p["prompt_i"], "seed": p["seed"], "sent_pos": p["sent_pos"]}
        for c in CATEGORIES:
            k[f"in_design__{c}"] = int(c in drawn[sid])
            k[f"pi__{c}"] = drawn[sid].get(c, "")
            k[f"stratum__{c}"] = p[f"stratum__{c}"]
        key.append(k)

    def write(path, recs, cols):
        with path.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, cols); w.writeheader(); w.writerows(recs)

    write(AUDIT_DIR / "audit_items_blinded.csv", items, list(items[0]))
    blank = [{"item_id": it["item_id"], **{c: "" for c in LABEL_COLS}, "notes": ""} for it in items]
    for a in (1, 2):
        write(AUDIT_DIR / f"annotator_{a}_labels.csv", blank, ["item_id", *LABEL_COLS, "notes"])
    write(KEY_DIR / "audit_key.csv", key, list(key[0]))
    pcols = ["sid", "key", "condition", "prompt_i", "sent_pos"] + [
        f"{x}__{c}" for c in CATEGORIES for x in ("stratum", "e5sim")] + [
        f"pred__{rn}__{c}" for rn in rules for c in CATEGORIES]
    with gzip.open(KEY_DIR / "population_predictions.csv.gz", "wt", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, pcols, extrasaction="ignore"); w.writeheader(); w.writerows(pop)
    blinded_hash = hashlib.sha256((AUDIT_DIR / "audit_items_blinded.csv").read_bytes()).hexdigest()
    (KEY_DIR / "design.json").write_text(json.dumps({
        "seed": SEED, "allocation_target": ALLOC, "n_items": len(items), "designs": design,
        "rules": {k: (None if v is None else {"scorer": v[0], "thresholds": v[1]}) for k, v in rules.items()},
        "label_columns": CAT_TO_COL, "blinded_items_sha256": blinded_hash,
        "population_unit": "sentence instance (historical splitter, all stored text)"}, indent=1),
        encoding="utf-8")
    print("items", len(items))


if __name__ == "__main__":
    main()
