"""Builds colab_analysis_v2.ipynb — a GPU-FREE analysis notebook that runs on the
CACHED v2 Drive checkpoints (no model loading, no generation). It produces the two
artifacts still missing from the paper:

  * results/interaction_tests.json  — formal stage x condition interaction
    (difference-in-differences) with prompt-clustered permutation p-values.
  * results/detector_validation_v2.json — per-condition detector precision/recall
    (incl. the measured factual false-positive rate) + Cohen's kappa, from the
    stratified validation_sheet_v2.csv.

It also ships an in-Colab labeler (google.colab.output HTML UI) so the sheet can be
labeled without leaving the notebook. Runs in ~1-2 minutes on a CPU runtime.

Run:  python experiments/build_analysis_notebook.py
"""
import json
from pathlib import Path

CELLS = []


def md(src): CELLS.append(("markdown", src))
def code(src): CELLS.append(("code", src))


md(r"""# OLMo-2 Advice-Provenance — v2 ANALYSIS (no GPU)

Reads the **cached checkpoints** from your v2 run and computes the two remaining
results for the paper. **No generation, no model loading** — a plain CPU runtime
is fine (~1-2 min).

1. **Stage F — stage x condition interaction (DiD + permutation p):** does the
   advice-minus-factual *gap* change significantly between adjacent training
   stages? Answers the reviewer's "is the rise advice-directed or generic drift?"
2. **Detector validation — stratified, per condition:** label the 120-sentence
   sheet (in-notebook UI below, or edit the CSV in Drive), then score per-category
   P/R **per condition** — giving the measured factual false-positive rate the
   paper currently marks "asserted, not measured" — plus Cohen's kappa.

Point the `PROJECT` path at your existing v2 project (the one with `behavior/` and
`results/validation_sheet_v2.csv`).""")

code(r"""#@title 1 · Mount Drive + locate the v2 project
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/llm-research-experiments-v2')  #@param {type:"string"}
assert (PROJECT/'behavior').exists(), f"No behavior/ under {PROJECT}; set PROJECT to your v2 run."
(PROJECT/'results').mkdir(exist_ok=True)
import json, glob
BEH = {Path(p).stem: p for p in glob.glob(str(PROJECT/'behavior'/'*.json'))}
print(f"Found {len(BEH)} cached behavior checkpoints under {PROJECT}")""")

code(r"""#@title 2 · Discover run structure from the checkpoints (no hardcoding)
import json
EMB_THRESHOLD = 0.82  #@param {type:"number"}
STAGE_ORDER = ["base","sft","dpo","rlvr","instruct"]
CATS = ["empathy_opener","validation","disclaimer","crisis_referral","structure"]

def parse_key(k):
    m, st, rest, seed = k.split('__'); t, i = rest.rsplit('_', 1)
    return m, st, t, int(i), int(seed[1:])

models, stages, tags, nprompt, seeds = set(), {}, set(), {}, set()
for k in BEH:
    m, st, t, i, s = parse_key(k)
    models.add(m); stages.setdefault(m, set()).add(st); tags.add(t); seeds.add(s)
    nprompt[t] = max(nprompt.get(t, 0), i+1)
MODELS = sorted(models)
STAGES = {m: [s for s in STAGE_ORDER if s in stages[m]] for m in MODELS}
SEEDS = max(seeds)+1
_beh_cache = {}
def load_beh(k):
    if k not in _beh_cache: _beh_cache[k] = json.loads(Path(BEH[k]).read_text(encoding='utf-8'))
    return _beh_cache[k]
def emits(b, c, thr): return (c in b["lexicon_categories"]) or (b.get("embedding_max_sims",{}).get(c,0) >= thr)
print("models:", MODELS, "| stages:", STAGES)
print("conditions:", {t: nprompt[t] for t in sorted(tags)}, "| seeds:", SEEDS)""")

code(r'''#@title 3 · STAGE F — stage x condition interaction (DiD, prompt-clustered permutation)
import numpy as np, random as _rnd
def prompt_traj(model, tag, cat, thr):
    out = {}
    for i in range(nprompt[tag]):
        rates = {}
        for st in STAGES[model]:
            vals = [1.0 if emits(load_beh(k), cat, thr) else 0.0
                    for s in range(SEEDS)
                    for k in [f"{model}__{st}__{tag}_{i:02d}__s{s}"] if k in BEH]
            if vals: rates[st] = sum(vals)/len(vals)
        if rates: out[i] = rates
    return out
def did(adv, fac, s0, s1):
    return ((np.mean([t[s1] for t in adv]) - np.mean([t[s1] for t in fac]))
            - (np.mean([t[s0] for t in adv]) - np.mean([t[s0] for t in fac])))
def interaction(model, cat, s0, s1, thr=EMB_THRESHOLD, n=10000, seed=0):
    A = prompt_traj(model, "advice", cat, thr); F = prompt_traj(model, "factual", cat, thr)
    adv = [t for t in A.values() if s0 in t and s1 in t]
    fac = [t for t in F.values() if s0 in t and s1 in t]
    if not adv or not fac: return None
    obs = did(adv, fac, s0, s1); pool = adv+fac; na = len(adv); _rnd.seed(seed); hits = 0
    for _ in range(n):
        _rnd.shuffle(pool)
        if abs(did(pool[:na], pool[na:], s0, s1)) >= abs(obs)-1e-12: hits += 1
    return {"did": round(float(obs),4), "perm_p": round((hits+1)/(n+1),5), "n_advice": na, "n_factual": len(fac)}

inter = {}
for m in MODELS:
    inter[m] = {}
    pairs = list(zip(STAGES[m][:-1], STAGES[m][1:])) + [(STAGES[m][0], STAGES[m][-1])]
    print(f"\n=== {m} ({' -> '.join(STAGES[m])}) ===")
    for c in CATS:
        inter[m][c] = {}
        for s0, s1 in pairs:
            r = interaction(m, c, s0, s1)
            if r: inter[m][c][f"{s0}->{s1}"] = r
        print(f"  {c:<16} " + " | ".join(f"{k}: {v['did']:+.3f} (p={v['perm_p']})" for k,v in inter[m][c].items()))
(PROJECT/'results'/'interaction_tests.json').write_text(json.dumps(inter, indent=2))
print("\nWrote results/interaction_tests.json  (send this file back)")''')

md(r"""## Detector validation — label the 120-sentence sheet, then score

Two ways to label:
- **In-notebook (cell 4 below):** click categories per sentence; autosaves to the
  Drive CSV. The detector's guess and the condition are hidden to avoid anchoring.
- **Or** open `results/validation_sheet_v2.csv` from Drive in Google Sheets and fill
  the `human_label_1` column (comma-separated categories, or `none`).

For inter-annotator agreement, have a second person fill `human_label_2` (rerun
cell 4 with `ANNOTATOR = 2`). Then run cell 5 to score.""")

code(r'''#@title 4 · (Optional) In-notebook labeler — click to label, autosaves to Drive
ANNOTATOR = 1  #@param [1, 2] {type:"raw"}
import csv, json, html as _html
from google.colab import output as _out
import IPython
SHEET = PROJECT/'results'/'validation_sheet_v2.csv'
col = f"human_label_{ANNOTATOR}"
with open(SHEET, encoding='utf-8', newline='') as f:
    _reader = csv.DictReader(f); FIELDS = _reader.fieldnames; ROWS = list(_reader)
if col not in FIELDS:
    FIELDS = FIELDS + [col]
    for r in ROWS: r.setdefault(col, "")
def _write():
    with open(SHEET, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(ROWS)
def _save_label(i, label):
    ROWS[int(i)][col] = label; _write()
    return IPython.display.JSON({"ok": True, "done": sum(1 for r in ROWS if r.get(col,"").strip())})
_out.register_callback('nb.save_label', _save_label)
CATS_JS = json.dumps(CATS)
payload = json.dumps([{"s": r["sentence"], "l": r.get(col,"") or ""} for r in ROWS])
HTML = r"""
<div id=prog style="position:sticky;top:0;background:#111;color:#9cf;padding:6px;font-weight:600"></div>
<div id=cards style="font-family:system-ui;max-width:820px"></div>
<style>.c{background:#1c1c1c;color:#ddd;border:1px solid #333;border-radius:6px;padding:8px 12px;margin:6px 0}
.c.d{border-color:#2a5}.s{white-space:pre-wrap;margin-bottom:6px}label{margin-right:12px;font-size:13px;cursor:pointer}
.n{color:#e90}input{accent-color:#4a8}</style>
<script>
const CATS=__CATS__, ROWS=__ROWS__;
function prog(d){document.getElementById('prog').textContent=(d==null?ROWS.filter(r=>r.l.trim()).length:d)+' / '+ROWS.length+' labeled (annotator __ANN__)';}
function save(i){google.colab.kernel.invokeFunction('nb.save_label',[i,ROWS[i].l],{}).then(r=>{
  try{prog(r.data['application/json'].done)}catch(e){prog()}});}
function tog(i,cat,card){let s=new Set(ROWS[i].l.split(',').map(x=>x.trim()).filter(x=>x&&x!='none'));
  if(cat=='none'){s.clear();ROWS[i].l='none';}else{s.has(cat)?s.delete(cat):s.add(cat);ROWS[i].l=[...s].sort().join(',')||'';}
  card.className='c'+(ROWS[i].l.trim()?' d':'');card.querySelectorAll('input').forEach(cb=>cb.checked=new Set(ROWS[i].l.split(',').map(x=>x.trim())).has(cb.dataset.c));save(i);}
const wrap=document.getElementById('cards');
ROWS.forEach((r,i)=>{const c=document.createElement('div');c.className='c'+(r.l.trim()?' d':'');
  const s=document.createElement('div');s.className='s';s.textContent='#'+(i+1)+'  '+r.s;c.appendChild(s);
  CATS.concat(['none']).forEach(cat=>{const l=document.createElement('label');if(cat=='none')l.className='n';
    const cb=document.createElement('input');cb.type='checkbox';cb.dataset.c=cat;
    cb.checked=new Set(r.l.split(',').map(x=>x.trim())).has(cat);cb.onclick=()=>tog(i,cat,c);
    l.appendChild(cb);l.appendChild(document.createTextNode(cat));c.appendChild(l);});
  wrap.appendChild(c);});
prog();
</script>"""
HTML = HTML.replace("__CATS__", CATS_JS).replace("__ROWS__", payload).replace("__ANN__", str(ANNOTATOR))
IPython.display.display(IPython.display.HTML(HTML))
print("Labeling annotator", ANNOTATOR, "-> column", col, "(autosaves to Drive on every click)")''')

code(r'''#@title 5 · Score the labeled sheet -> per-condition P/R + factual FP rate + kappa
import csv, json
SHEET = PROJECT/'results'/'validation_sheet_v2.csv'
rows = [r for r in csv.DictReader(open(SHEET, encoding='utf-8')) if r.get("human_label_1","").strip()]
total = sum(1 for _ in csv.DictReader(open(SHEET, encoding='utf-8')))
if not rows:
    print(f"No labels yet in {SHEET} — label via cell 4 (or Google Sheets), then re-run.")
else:
    def parse(x): return {v.strip() for v in x.replace(';',',').split(',') if v.strip() and v.strip()!='none'}
    def score(sub, gold="human_label_1"):
        res={}
        for c in CATS:
            tp=sum(1 for r in sub if c in parse(r["predicted"]) and c in parse(r[gold]))
            fp=sum(1 for r in sub if c in parse(r["predicted"]) and c not in parse(r[gold]))
            fn=sum(1 for r in sub if c not in parse(r["predicted"]) and c in parse(r[gold]))
            res[c]={"P":round(tp/(tp+fp),3) if tp+fp else None,"R":round(tp/(tp+fn),3) if tp+fn else None,
                    "tp":tp,"fp":fp,"fn":fn}
        return res
    out={"n_labeled":len(rows),"overall":score(rows),"by_condition":{}}
    print(f"Scored {len(rows)}/{total} labeled sentences\n{'cat':<16}{'P':>7}{'R':>7}  tp/fp/fn")
    for c,v in out["overall"].items(): print(f"{c:<16}{str(v['P']):>7}{str(v['R']):>7}  {v['tp']}/{v['fp']}/{v['fn']}")
    for t in sorted({r["condition"] for r in rows}):
        out["by_condition"][t]={"n":sum(1 for r in rows if r["condition"]==t),**score([r for r in rows if r["condition"]==t])}
    print("\nPer-condition detector false-positives (the 'measured factual FP rate'):")
    for t,v in out["by_condition"].items():
        print(f"  {t:<16} n={v['n']:>3}  fp={ {c:v[c]['fp'] for c in CATS if v[c]['fp']} }")
    both=[r for r in rows if r.get("human_label_2","").strip()]
    if both:
        from sklearn.metrics import cohen_kappa_score
        ks={}
        for c in CATS:
            y1=[int(c in parse(r['human_label_1'])) for r in both]; y2=[int(c in parse(r['human_label_2'])) for r in both]
            ks[c]=round(cohen_kappa_score(y1,y2),3) if (len(set(y1))>1 or len(set(y2))>1) else None
        out["cohen_kappa"]={"n_double":len(both),**ks}; print("\nCohen's kappa:",ks)
    (PROJECT/'results'/'detector_validation_v2.json').write_text(json.dumps(out,indent=2))
    print("\nWrote results/detector_validation_v2.json  (send this file back)")''')

md(r"""## Send back
Download from Drive and share: `results/interaction_tests.json` and (once labeled)
`results/detector_validation_v2.json`. These slot directly into the paper's
Stagewise and Detector-validation sections.""")

# ---------------------------------------------------------------- emit
nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
         if t == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": s.splitlines(keepends=True)})
        for t, s in CELLS
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).parent / "colab_analysis_v2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} with {len(CELLS)} cells")
