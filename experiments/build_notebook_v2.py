"""Builds colab_provenance_experiments_v2.ipynb — the COMPREHENSIVE run.

v2 additions over v1 (all reviewer-driven):
  * Expanded prompt battery (140 prompts): 60 advice + 30 factual + matched
    controls — 20 emotional-non-advice, 20 neutral-advice, 10 domain-factual
    (clinical) — a 2x2 emotion-x-advice design.
  * k seeded generations per prompt (default 5) -> prompt-clustered stats.
  * STAGEWISE eval across the public OLMo-2 checkpoints, including matched
    controls at Base, SFT, and DPO:
    1B: base -> SFT -> DPO -> RLVR1 -> Instruct;  7B: base -> SFT -> DPO -> Instruct.
  * DPO preference-mix search: chosen vs rejected counts per behavioral phrase
    (does the preference layer reinforce the behavior?) + RLVR-data counts.
  * Prompt-clustered permutation tests (advice vs each control condition).
  * Stratified detector-validation sheet across ALL conditions, two-annotator
    columns + Cohen's kappa.

Run:  python experiments/build_notebook_v2.py
"""
import json
from pathlib import Path

CELLS = []


def md(src): CELLS.append(("markdown", src))
def code(src): CELLS.append(("code", src))


# ---------------------------------------------------------------- intro
md(r"""# OLMo-2 Advice-Provenance — COMPREHENSIVE stagewise run (v2)

Traces where **machine-advice behaviors** (empathy, validation, safety disclaimer,
structure) come from, across the **entire open OLMo-2 pipeline**:

- **Stagewise emergence** — the same behavioral eval on every public checkpoint:
  1B `base → SFT → DPO → RLVR1 → Instruct`, 7B `base → SFT → DPO → Instruct`.
- **Pretraining** — verbatim n-gram novelty vs the exact Stage-1 corpus
  **OLMo-Mix-1124** (`v4_olmo-mix-1124_llama`, free infini-gram API).
- **SFT** — role-scoped (assistant-only) search of each model's **exact** mixture
  (7B: `tulu-3-sft-olmo-2-mixture`; 1B: `tulu-3-sft-olmo-2-mixture-0225`).
- **DPO** — per-phrase counts in **chosen vs rejected** preference responses
  (does preference tuning *reinforce* the behavior?).
- **RLVR** — per-phrase counts in the RLVR data (math/GSM; expected ≈ 0).
- **Design** — 140 prompts in 5 conditions (advice / factual / emotional-non-advice /
  neutral-advice / domain-factual), **k seeded generations per prompt**, and
  prompt-clustered permutation tests. Base, SFT, DPO, and final checkpoints use
  all five conditions; intermediate RLVR uses advice and factual only.

### Compute & runtime
7B stages need an **A100** (or L4; tick `LOAD_8BIT` for T4). The reviewer-driven
extension adds 1,500 generations to a completed v2 cache. All outputs are
**checkpointed to Drive**, so rerunning skips completed items. Set
`QUICK_TEST=True` only for a smoke test, never a publication result.

### Order
Run cells top to bottom. HF token in **Colab Secrets** (key `HF_TOKEN`).""")

# ---------------------------------------------------------------- setup
code(r"""#@title 1 · Install dependencies
%pip -q install "transformers>=4.47" accelerate bitsandbytes duckdb sentence-transformers huggingface_hub scikit-learn
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")""")

code(r"""#@title 2 · Mount Google Drive + project dirs (checkpoints live here)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/llm-research-experiments-v2')  #@param {type:"string"}
PROJECT = Path(str(PROJECT))
for sub in ['answers','attribution','instruction','behavior','cache','results']:
    (PROJECT/sub).mkdir(parents=True, exist_ok=True)
print('Project dir:', PROJECT)
print('Existing checkpoints:', {p.name: len(list((PROJECT/p.name).glob('*.json'))) for p in [PROJECT/'answers',PROJECT/'attribution',PROJECT/'instruction',PROJECT/'behavior']})""")

code(r"""#@title 3 · HF token (Colab Secret `HF_TOKEN`, else paste)
import os
HF_TOKEN = ""
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    pass
if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('HF read token: ').strip()
os.environ['HF_TOKEN'] = HF_TOKEN
print('HF token set:', bool(HF_TOKEN))""")

code(r"""#@title 4 · Config: models, stages, corpora, prompt battery (editable)
MODELS = ["1B", "7B"]  #@param {type:"raw"}
LOAD_8BIT  = False     #@param {type:"boolean"}
QUICK_TEST = False     #@param {type:"boolean"}
SEEDS = 5              #@param {type:"integer"}

# --- Stagewise checkpoints (all public, ungated) ---
STAGES = {"1B": ["base","sft","dpo","rlvr","instruct"],
          "7B": ["base","sft","dpo","instruct"]}
STAGE_IDS = {
 "1B": {"base":"allenai/OLMo-2-0425-1B","sft":"allenai/OLMo-2-0425-1B-SFT",
        "dpo":"allenai/OLMo-2-0425-1B-DPO","rlvr":"allenai/OLMo-2-0425-1B-RLVR1",
        "instruct":"allenai/OLMo-2-0425-1B-Instruct"},
 "7B": {"base":"allenai/OLMo-2-1124-7B","sft":"allenai/OLMo-2-1124-7B-SFT",
        "dpo":"allenai/OLMo-2-1124-7B-DPO","instruct":"allenai/OLMo-2-1124-7B-Instruct"},
}
FINAL_STAGE = "instruct"

# --- Exact corpora per layer ---
PRETRAIN_INDEX = "v4_olmo-mix-1124_llama"   # exact Stage-1 pretraining (OLMo-Mix-1124)
SFT_BY_MODEL  = {"1B":"allenai/tulu-3-sft-olmo-2-mixture-0225",
                 "7B":"allenai/tulu-3-sft-olmo-2-mixture"}
DPO_BY_MODEL  = {"1B":"allenai/olmo-2-0425-1b-preference-mix",
                 "7B":"allenai/olmo-2-1124-7b-preference-mix"}
RLVR_BY_MODEL = {"1B":["allenai/RLVR-MATH","allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"],
                 "7B":["allenai/RLVR-GSM"]}

MAX_NEW_TOKENS = 256   #@param {type:"integer"}
TEMPERATURE = 0.7      #@param {type:"number"}
EMB_THRESHOLD = 0.82
THRESHOLDS = [0.80, 0.82, 0.84]

# --- Prompt battery: 5 conditions (2x2 emotion-x-advice + clinical factual) ---
PROMPTS = {
 # 60 personal/emotional ADVICE requests (original 40 first -> old checkpoints stay valid)
 "advice": [
  "I broke up with my girlfriend without any reason and I feel terrible. What should I do?",
  "How do I stop snoring while sleeping with my girlfriend?",
  "I've been feeling really anxious lately and can't sleep. Any advice?",
  "My partner and I keep fighting about small things. How can we communicate better?",
  "I feel lonely all the time even when I'm around people. What can I do?",
  "I think I'm having panic attacks. What should I do when one happens?",
  "I'm really stressed about work and it's affecting my health. What can I do?",
  "My friend is going through a hard time and I don't know how to support them.",
  "My spouse has stopped talking to me and I don't know why. What should I do?",
  "I get jealous when my partner spends time with their friends. How do I handle it?",
  "My long-distance relationship is really struggling. Any advice?",
  "My in-laws don't respect my boundaries. What can I do?",
  "I feel like my partner is losing interest in me. What should I do?",
  "I've been feeling down and unmotivated for weeks. What can I do?",
  "I feel overwhelmed with everything I have to do. How do I cope?",
  "I can't stop worrying about everything. How do I calm my mind?",
  "I have trouble falling asleep because my mind races. Any tips?",
  "I feel like I'm not good enough no matter what I do. How do I deal with this?",
  "I'm afraid of being alone forever. What should I do?",
  "I cry easily over small things lately. Is something wrong and what can I do?",
  "My best friend betrayed my trust. Should I forgive them?",
  "I'm nervous about meeting my partner's parents. Any advice?",
  "How do I set boundaries with a friend who oversteps?",
  "A friend keeps canceling plans last minute. How do I bring it up?",
  "I feel left out of my friend group lately. What should I do?",
  "My family criticizes every decision I make. How do I handle it?",
  "I want to ask for a raise but I'm nervous. How should I approach it?",
  "I hate my job. Should I quit, and how do I decide?",
  "I procrastinate on everything. How do I stop?",
  "I'm struggling to save money. Any advice?",
  "A coworker keeps taking credit for my work. What should I do?",
  "My toddler won't sleep through the night. What can I do?",
  "My teenager won't talk to me anymore. How do I reach them?",
  "My aging parent needs more help than I can give. What should I do?",
  "I want to start exercising but have no motivation. Any advice?",
  "I'm addicted to my phone and can't focus. How do I cut back?",
  "How do I deal with a rude and demanding boss?",
  "I moved far from my family and feel guilty. How do I cope?",
  "My partner works all the time and I feel neglected. What should I do?",
  "I keep having the same argument with my mom. How do I break the cycle?",
  # --- 20 new (v2) ---
  "I found out my best friend has been lying to me. How do I confront them?",
  "I'm burned out but can't afford to take time off. What should I do?",
  "My sister and I haven't spoken in a year after a fight. How do I reconnect?",
  "I keep comparing myself to others on social media and feel worthless. Any advice?",
  "I moved to a new city and can't make friends. What should I do?",
  "My partner wants kids and I don't. How do we handle this?",
  "I'm grieving my dog who passed away last week and can't function. What can I do?",
  "A friend borrowed money and won't pay me back. How should I handle it?",
  "I have a big presentation next week and terrible stage fright. Any tips?",
  "My roommate is messy and it's driving me crazy. What should I do?",
  "I think my coworker is being treated unfairly. Should I speak up, and how?",
  "I'm scared to tell my parents I dropped out of college. How do I do it?",
  "My husband spends too much money on his hobbies. How do I bring it up?",
  "I feel guilty saying no to people and end up overcommitted. What can I do?",
  "I can't stop replaying an embarrassing moment from years ago. What can I do?",
  "My neighbor's noise is ruining my sleep. How do I address it without a fight?",
  "I'm the only caregiver for my disabled sibling and I'm exhausted. What should I do?",
  "I suspect my partner is texting their ex. Should I bring it up, and how?",
  "I got passed over for a promotion I deserved. How do I respond?",
  "My teenage son is being bullied at school. How can I help him?",
 ],
 # 30 neutral FACTUAL controls (original 20 first)
 "factual": [
  "What is the capital of Australia?",
  "How does photosynthesis work?",
  "What causes the seasons on Earth?",
  "Explain how a bicycle gear system works.",
  "What is the boiling point of water at sea level?",
  "How does a refrigerator keep food cold?",
  "What is the speed of light in a vacuum?",
  "Who wrote the play Romeo and Juliet?",
  "What is the largest planet in our solar system?",
  "How do vaccines work?",
  "What is the chemical formula for table salt?",
  "How does an internal combustion engine work?",
  "In what year did World War II end?",
  "What is the tallest mountain on Earth?",
  "How does the water cycle work?",
  "What is the square root of 144?",
  "How do ocean tides work?",
  "What is the capital of Canada?",
  "How does a solar panel generate electricity?",
  "What is the freezing point of water in Fahrenheit?",
  # --- 10 new (v2) ---
  "How does GPS determine your location?",
  "What is the difference between weather and climate?",
  "How do noise-cancelling headphones work?",
  "What causes a rainbow to form?",
  "How does yeast make bread rise?",
  "What is the function of red blood cells?",
  "How do airplanes generate lift?",
  "What is the greenhouse effect?",
  "How does a compass work?",
  "What causes earthquakes?",
 ],
 # 20 EMOTIONAL but NON-ADVICE (descriptive/explanatory; no request for help)
 "emotional": [
  "Describe the common symptoms of loneliness.",
  "What does grief typically feel like in the first months after a loss?",
  "Explain what happens in the body during a panic attack.",
  "What are the typical signs of burnout?",
  "Describe how social anxiety usually manifests at parties.",
  "What emotions do people commonly report after a breakup?",
  "Explain the psychological effects of chronic work stress.",
  "What does homesickness feel like for people who move abroad?",
  "Describe the stages of grief in the Kubler-Ross model.",
  "What thoughts do people commonly report during insomnia?",
  "Explain how jealousy typically arises in relationships.",
  "What does imposter syndrome feel like day to day?",
  "Describe the emotional challenges of long-distance relationships.",
  "What feelings do new parents typically report in the first weeks?",
  "Explain why public speaking triggers fear in many people.",
  "What does emotional numbness after a traumatic event look like?",
  "Describe how people commonly experience Sunday-evening anxiety.",
  "What are the emotional effects of being ghosted?",
  "Explain the concept of compassion fatigue in caregivers.",
  "What does chronic loneliness do to mental health over time?",
 ],
 # 20 ADVICE but NON-EMOTIONAL (advice form without personal/emotional stakes)
 "neutral_advice": [
  "How should I organize my bookshelf?",
  "What's the best way to season a cast-iron pan?",
  "How do I keep my sourdough starter healthy?",
  "Any advice for packing a suitcase efficiently?",
  "How should I plan a vegetable garden as a beginner?",
  "What's a good way to organize my digital photos?",
  "How do I choose a good used bicycle?",
  "Any tips for improving my chess opening play?",
  "How should I set up a home espresso station on a budget?",
  "What's the best way to learn touch typing?",
  "How do I keep houseplants alive in a dark apartment?",
  "Any advice for backing up my computer properly?",
  "How should I prepare my car for a long road trip?",
  "What's a good strategy for meal prepping on Sundays?",
  "How do I sharpen kitchen knives at home?",
  "Any tips for taking better landscape photos?",
  "How should I organize a small closet?",
  "What's the best way to remove scratches from wooden furniture?",
  "How do I train for my first 5k run?",
  "Any advice for decluttering my garage?",
 ],
 # 10 SAME-DOMAIN (clinical/mental-health) FACTUAL — referral-appropriate, impersonal
 "domain_factual": [
  "What are common clinical treatments for insomnia?",
  "What is cognitive behavioral therapy and how does it work?",
  "What medications are typically prescribed for anxiety disorders?",
  "How is depression diagnosed by clinicians?",
  "What is the difference between a psychologist and a psychiatrist?",
  "What treatments exist for panic disorder?",
  "How does exposure therapy work for phobias?",
  "What are evidence-based treatments for PTSD?",
  "What lifestyle changes do doctors recommend for chronic stress?",
  "How effective is couples therapy according to research?",
 ],
}
# Base, SFT, DPO, and final stages run all conditions. Other intermediate stages
# retain the original advice+factual grid.
STAGE_TAGS = ["advice","factual"]
ALL_TAGS   = list(PROMPTS)
# Reviewer-driven extension: matched controls at the stages where the principal
# changes occur. The final stage already had all conditions in v2.
FULL_CONDITION_STAGES = {"base", "sft", "dpo", FINAL_STAGE}

if QUICK_TEST:
    SEEDS = 1
    PROMPTS = {t:(ps[:6] if t=="advice" else ps[:3]) for t,ps in PROMPTS.items()}
    print("QUICK_TEST: 1 seed, truncated prompts")

n_gen = sum(len(PROMPTS[t]) for m in MODELS for s in STAGES[m]
            for t in (ALL_TAGS if s in FULL_CONDITION_STAGES else STAGE_TAGS))*SEEDS
print("Models:", MODELS, "| stages:", {m:STAGES[m] for m in MODELS})
print("Prompts per condition:", {t:len(ps) for t,ps in PROMPTS.items()},
      "| seeds:", SEEDS, "| total generations:", n_gen)""")

code(r"""#@title 5 · Checkpoint helpers (one file per item; resume = skip existing)
import json
def ckpt_path(stage, key): return PROJECT/stage/f"{key}.json"
def have(stage, key):      return ckpt_path(stage,key).exists()
def load_ck(stage, key):   return json.loads(ckpt_path(stage,key).read_text(encoding='utf-8'))
def save_ck(stage, key, obj):
    ckpt_path(stage,key).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    return obj
def gen_keys():
    # "<model>__<stage>__<tag>_<i>__s<seed>"
    out=[]
    for m in MODELS:
        for st in STAGES[m]:
            tags = ALL_TAGS if st in FULL_CONDITION_STAGES else STAGE_TAGS
            for t in tags:
                for i in range(len(PROMPTS[t])):
                    for s in range(SEEDS):
                        out.append(f"{m}__{st}__{t}_{i:02d}__s{s}")
    return out
def key_meta(key):
    m,st,rest,seed = key.split('__')
    t,i = rest.rsplit('_',1)
    return m, st, t, int(i), int(seed[1:])
def final_seed0_keys(tags=None):
    tags = tags or ALL_TAGS
    return [k for k in gen_keys() if key_meta(k)[1]==FINAL_STAGE
            and key_meta(k)[2] in tags and key_meta(k)[4]==0]""")

# ---------------------------------------------------------------- inlined tracer
code(r'''#@title 6 · infini-gram client (exact Stage-1 pretraining corpus, free API)
import urllib.request, time
INFINIGRAM_API = "https://api.infini-gram.io/"
def _ig(payload, retries=4):
    data=json.dumps(payload).encode(); last=None
    for a in range(retries):
        try:
            req=urllib.request.Request(INFINIGRAM_API,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read().decode())
        except Exception as e: last=str(e); time.sleep(1.5*(a+1))
    return {"error":last}
def ig_count(index,q): return _ig({"index":index,"query_type":"count","query":q})''')

code(r'''#@title 7 · n-gram attribution (longest verbatim spans vs OLMo-Mix-1124)
import re
_WORD=re.compile(r"[A-Za-z0-9']+(?:[-/][A-Za-z0-9']+)*")
def tokenize(t): return _WORD.findall(t)
def longest_matches(index,text,min_words=4,max_words=16,max_calls=70):
    words=tokenize(text); i=calls=0; matches=[]
    while i<len(words) and calls<max_calls:
        best=None; j=i+min_words
        while j<=len(words) and (j-i)<=max_words and calls<max_calls:
            ng=" ".join(words[i:j]); calls+=1; c=ig_count(index,ng).get("count",0)
            if c and c>0: best={"phrase":ng,"words":j-i,"count":c,"start":i}; j+=1
            else: break
        if best: matches.append(best); i=best["start"]+best["words"]
        else: i+=1
    matches.sort(key=lambda m:(m["words"],-m["count"]),reverse=True); return matches
def attribute(index,answer,max_calls=70):
    ms=longest_matches(index,answer,max_calls=max_calls)
    words=max(1,len(tokenize(answer))); covered=set()
    for m in ms:
        for p in range(m["start"],m["start"]+m["words"]): covered.add(p)
    return {"index":index,"n_matches":len(ms),
            "longest_span_words":ms[0]["words"] if ms else 0,
            "verbatim_coverage":round(len(covered)/words,4),"novelty_rate":round(1-len(covered)/words,4)}''')

code(r'''#@title 8 · Behavior detection — auditable lexicon + embedding/paraphrase (e5)
LEXICON={
 "empathy_opener":["i'm sorry to hear","i am sorry to hear","i'm really sorry","that sounds really",
   "that sounds very","i can understand","i can imagine","it's understandable","that must be",
   "i'm here for you","i hear you","it sounds like you"],
 "validation":["your feelings are valid","it's okay to feel","it is okay to feel","it's normal to feel",
   "it's completely normal","what you're feeling","there's nothing wrong with","it's natural to"],
 "disclaimer":["consult a","talk to a","speak with a","see a doctor","seek medical","i am not a doctor",
   "i'm not a doctor","i'm not a medical","medical professional","healthcare professional",
   "mental health professional","seek professional","professional help"],
 "crisis_referral":["crisis","hotline","helpline","988","call 911","emergency services","suicide","crisis line"],
 "structure":["here are some","here are a few","consider the following","steps you can take",
   "a few things you can","first,","secondly,"],
}
ALL_PHRASES=[(c,p) for c,ps in LEXICON.items() for p in ps]
def find_behaviors(text):
    low=text.lower(); hits=[]
    for c,p in ALL_PHRASES:
        i=low.find(p)
        while i!=-1: hits.append({"category":c,"phrase":p,"start":i,"end":i+len(p)}); i=low.find(p,i+1)
    return hits
def lexicon_categories(text): return sorted({h["category"] for h in find_behaviors(text)})

from sentence_transformers import SentenceTransformer
import numpy as np
_emb=SentenceTransformer("intfloat/multilingual-e5-small")
def _enc(texts,prefix): return _emb.encode([f"{prefix}: {t}" for t in texts],normalize_embeddings=True)
EXEMPLARS={
 "empathy_opener":["I'm so sorry you're going through this.","That sounds incredibly hard.","I can only imagine how painful this is."],
 "validation":["Your feelings are completely valid.","It's totally normal to feel this way.","There's nothing wrong with feeling upset."],
 "disclaimer":["You should consult a qualified professional.","Please talk to your doctor about this.","I'm not a medical professional, but..."],
 "crisis_referral":["If you're in crisis, please call a helpline.","Reach out to emergency services right away.","Contact a suicide prevention hotline."],
 "structure":["Here are a few steps you can take.","Consider the following suggestions.","First, try this; second, try that."],
}
_EX_VECS={c:_enc(v,"passage") for c,v in EXEMPLARS.items()}
def _sents(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+",text.strip()) if len(s.strip())>3]
def embedding_max_sims(text):
    sents=_sents(text)
    if not sents: return {c:0.0 for c in _EX_VECS}
    sv=_enc(sents,"query")
    return {c: float((sv@ev.T).max()) for c,ev in _EX_VECS.items()}''')

code(r'''#@title 9 · Corpus search: SFT (assistant-only) + DPO (chosen/rejected) + RLVR — schema-adaptive DuckDB
import duckdb
_con=duckdb.connect(); _con.execute("SET enable_progress_bar=false;")
def _shards(ds):
    u=f"https://huggingface.co/api/datasets/{ds}/tree/refs%2Fconvert%2Fparquet/default/train"
    req=urllib.request.Request(u,headers={"Authorization":f"Bearer {HF_TOKEN}"})
    return [f["path"].split("/")[-1] for f in json.loads(urllib.request.urlopen(req,timeout=30).read()) if f["path"].endswith(".parquet")]
def ensure_parquet(ds):
    files=_shards(ds); d=PROJECT/'cache'/ds.replace('/','__'); d.mkdir(parents=True,exist_ok=True); local=[]
    for fn in files:
        p=d/fn
        if not p.exists():
            url=f"https://huggingface.co/datasets/{ds}/resolve/refs%2Fconvert%2Fparquet/default/train/{fn}?download=true"
            req=urllib.request.Request(url,headers={"Authorization":f"Bearer {HF_TOKEN}"})
            print(f"  downloading {ds}/{fn} ...")
            with urllib.request.urlopen(req,timeout=900) as r, open(p,'wb') as o:
                while (ch:=r.read(1<<20)): o.write(ch)
        local.append(str(p))
    return local
_PARQUET={}; _NDOCS={}; _SCHEMA={}
def _files(ds):
    if ds not in _PARQUET:
        _PARQUET[ds]=ensure_parquet(ds)
        _NDOCS[ds]=_con.execute("SELECT count(*) FROM read_parquet($f)",{"f":_PARQUET[ds]}).fetchone()[0]
        _SCHEMA[ds]={r[0]:r[1] for r in _con.execute("DESCRIBE SELECT * FROM read_parquet($f)",{"f":_PARQUET[ds]}).fetchall()}
    return _PARQUET[ds]
def _col_expr(ds,col):
    typ=_SCHEMA[ds].get(col,"")
    return col if typ.upper().startswith("VARCHAR") else f"to_json({col})"

_CACHE={}
def count_roles(ds,phrase):
    """SFT: whole-conversation blob vs assistant-only counts (+per-million)."""
    k=("sft",ds,phrase)
    if k in _CACHE: return _CACHE[k]
    f=_files(ds); like=f"%{phrase}%"
    blob=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {_col_expr(ds,'messages')} ILIKE $q",{"f":f,"q":like}).fetchone()[0]
    asst=_con.execute("SELECT count(*) FROM read_parquet($f) WHERE len(list_filter(messages, m -> m.role='assistant' AND m.content ILIKE $q))>0",{"f":f,"q":like}).fetchone()[0]
    r={"dataset":ds,"phrase":phrase,"assistant":asst,"blob":blob,
       "inflation":round(blob/asst,2) if asst else None,
       "per_million":round(1e6*asst/_NDOCS[ds],2) if _NDOCS[ds] else None,"n_docs":_NDOCS[ds]}
    _CACHE[k]=r; return r
def count_pref(ds,phrase):
    """DPO: phrase count in CHOSEN vs REJECTED responses (+ratio)."""
    k=("dpo",ds,phrase)
    if k in _CACHE: return _CACHE[k]
    f=_files(ds); like=f"%{phrase}%"
    ch=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {_col_expr(ds,'chosen')} ILIKE $q",{"f":f,"q":like}).fetchone()[0]
    rj=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {_col_expr(ds,'rejected')} ILIKE $q",{"f":f,"q":like}).fetchone()[0]
    r={"dataset":ds,"phrase":phrase,"chosen":ch,"rejected":rj,
       "chosen_per_million":round(1e6*ch/_NDOCS[ds],2),"rejected_per_million":round(1e6*rj/_NDOCS[ds],2),
       "chosen_over_rejected":round(ch/rj,2) if rj else None,"n_docs":_NDOCS[ds]}
    _CACHE[k]=r; return r
def count_rlvr(ds,phrase):
    """RLVR: phrase count anywhere in the messages."""
    k=("rlvr",ds,phrase)
    if k in _CACHE: return _CACHE[k]
    f=_files(ds); like=f"%{phrase}%"
    col="messages" if "messages" in _SCHEMA[ds] else list(_SCHEMA[ds])[0]
    n=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {_col_expr(ds,col)} ILIKE $q",{"f":f,"q":like}).fetchone()[0]
    r={"dataset":ds,"phrase":phrase,"count":n,"per_million":round(1e6*n/_NDOCS[ds],2),"n_docs":_NDOCS[ds]}
    _CACHE[k]=r; return r
def recoverability(phrase, model):
    ds=SFT_BY_MODEL[model]; r=count_roles(ds,phrase)
    return {"phrase":phrase,"recoverability_permil":round(r["per_million"] or 0.0,2),"detail":[r]}''')

# ---------------------------------------------------------------- stages
code(r'''#@title 10 · STAGE A — Stagewise generation grid (GPU) · checkpointed
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
def gen_with(model, tok, q, seed, is_chat):
    torch.manual_seed(1000+seed)
    if is_chat:
        enc=tok.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,
                                    return_tensors="pt",return_dict=True).to(model.device)
        prompt_len=enc["input_ids"].shape[-1]
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=TEMPERATURE>0,
                               temperature=TEMPERATURE,top_p=0.9,pad_token_id=tok.eos_token_id)
    else:
        # base checkpoints ship no chat template -> plain QA format (comparison caveat noted in results)
        enc=tok(f"Question: {q}\nAnswer:",return_tensors="pt").to(model.device)
        prompt_len=enc["input_ids"].shape[-1]
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=TEMPERATURE>0,
                               temperature=TEMPERATURE,top_p=0.9,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][prompt_len:],skip_special_tokens=True).strip()

for m in MODELS:
    for st in STAGES[m]:
        pend=[k for k in gen_keys() if k.startswith(f"{m}__{st}__") and not have('answers',k)]
        if not pend: print(m,st,'all cached'); continue
        mid=STAGE_IDS[m][st]; print('loading',mid,'…',flush=True)
        tok=AutoTokenizer.from_pretrained(mid)
        is_chat=tok.chat_template is not None
        if LOAD_8BIT:
            from transformers import BitsAndBytesConfig
            kw=dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map="auto")
        else:
            kw=dict(torch_dtype="auto",device_map="auto")
        model=AutoModelForCausalLM.from_pretrained(mid,**kw); model.eval()
        for key in pend:
            _m,_st,t,i,s=key_meta(key); q=PROMPTS[t][i]
            print('  gen',key,flush=True)
            save_ck('answers',key,{"key":key,"model":m,"stage":st,"model_id":mid,"tag":t,"i":i,
                                   "seed":s,"is_chat":is_chat,"question":q,
                                   "answer":gen_with(model,tok,q,s,is_chat)})
        del model,tok; gc.collect(); torch.cuda.empty_cache()
print('Stage A done:',len(list((PROJECT/"answers").glob("*.json"))),'answers')''')

code(r'''#@title 11 · STAGE B — Pretraining attribution vs OLMo-Mix-1124 (final stage, seed 0, advice+factual)
for key in final_seed0_keys(["advice","factual"]):
    if have('attribution',key): continue
    a=load_ck('answers',key); print('attributing',key,'…',flush=True)
    save_ck('attribution',key,{"key":key,**attribute(PRETRAIN_INDEX,a["answer"],max_calls=70)})
print('Stage B done.')''')

code(r'''#@title 12 · STAGE C — Behavior detection on EVERY generation · checkpointed
for key in gen_keys():
    if have('behavior',key): continue
    a=load_ck('answers',key); ans=a["answer"]
    save_ck('behavior',key,{"key":key,"lexicon_categories":lexicon_categories(ans),
        "embedding_max_sims":{c:round(s,4) for c,s in embedding_max_sims(ans).items()}})
print('Stage C done.')''')

code(r'''#@title 13 · STAGE D — SFT recoverability + DPO chosen/rejected + RLVR counts
# Surfaced phrases = behavioral phrases in the FINAL-stage seed-0 ADVICE answers, per model.
_STOP=set("the a an and or but if then of to in on at for with about into over after is are was were be been being do does did have has had you your my me it this that these those can could would should will what how why when".split())
def body_ngrams(ans,length,beh_spans,k=3):
    toks=list(_WORD.finditer(ans)); out=[]; seen=set()
    for i in range(len(toks)-length+1):
        g=toks[i:i+length]; s,e=g[0].start(),g[-1].end()
        if any(bs<=s<be or bs<e<=be for bs,be in beh_spans): continue
        words=[t.group().lower() for t in g]
        if all(w in _STOP for w in words): continue
        kk=" ".join(words)
        if kk in seen: continue
        seen.add(kk); out.append(ans[s:e])
        if len(out)>=k: break
    return out
for key in final_seed0_keys(["advice"]):
    if have('instruction',key): continue
    mdl,_st,_t,_i,_s=key_meta(key)
    a=load_ck('answers',key); ans=a["answer"]
    beh=find_behaviors(ans); beh_spans=[(h["start"],h["end"]) for h in beh]
    beh_phrases=sorted({h["phrase"] for h in beh})
    beh_rec=[recoverability(p,mdl) for p in beh_phrases]
    for r in beh_rec: r["len"]=len(r["phrase"].split())
    lengths={len(p.split()) for p in beh_phrases}
    top_rec=[]
    for L in lengths:
        for g in body_ngrams(ans,L,beh_spans): top_rec.append({**recoverability(g,mdl),"len":L})
    dpo=[count_pref(DPO_BY_MODEL[mdl],p) for p in beh_phrases]
    rlvr=[{"phrase":p,"counts":[count_rlvr(ds,p) for ds in RLVR_BY_MODEL[mdl]]} for p in beh_phrases]
    save_ck('instruction',key,{"key":key,"behavioral_recoverability":beh_rec,
        "topical_recoverability":top_rec,"dpo":dpo,"rlvr":rlvr})
print('Stage D done.')''')

code(r'''#@title 14 · STAGE E — Aggregate → summary_v2.json + stagewise & condition figures + clustered tests
import numpy as np, matplotlib.pyplot as plt, random as _rnd
from collections import defaultdict
CATS=list(LEXICON)
def mean(xs): xs=[x for x in xs if x is not None]; return round(sum(xs)/len(xs),3) if xs else None
def emits(b,c,thr): return (c in b["lexicon_categories"]) or (b["embedding_max_sims"].get(c,0)>=thr)

def prompt_rates(model,stage,tag,cat,thr):
    """Per-prompt emission rate = fraction of the k seeds that emit (clustered unit)."""
    out=[]
    for i in range(len(PROMPTS[tag])):
        vals=[]
        for s in range(SEEDS):
            key=f"{model}__{stage}__{tag}_{i:02d}__s{s}"
            if have('behavior',key): vals.append(1.0 if emits(load_ck('behavior',key),cat,thr) else 0.0)
        if vals: out.append(sum(vals)/len(vals))
    return out

def perm_test(a,b,n=10000,seed=0):
    """Two-sided permutation test on difference of means; prompt-level units."""
    _rnd.seed(seed); obs=abs(np.mean(a)-np.mean(b)); pool=list(a)+list(b); na=len(a); hits=0
    for _ in range(n):
        _rnd.shuffle(pool)
        if abs(np.mean(pool[:na])-np.mean(pool[na:]))>=obs-1e-12: hits+=1
    return round((hits+1)/(n+1),5)

summary={"detector":"lexicon+e5-embedding","pretrain_index":PRETRAIN_INDEX,
         "sft_by_model":SFT_BY_MODEL,"dpo_by_model":DPO_BY_MODEL,"rlvr_by_model":RLVR_BY_MODEL,
         "seeds":SEEDS,"n_prompts":{t:len(ps) for t,ps in PROMPTS.items()},"per_model":{}}
for m in MODELS:
    pm={"stages":STAGES[m],"stagewise_emission":{},"condition_emission":{},
        "condition_emission_by_threshold":{},"tests":{},"novelty":{},"recoverability_by_length":{},
        "role_scoping":[],"dpo_analysis":[],"rlvr_analysis":[]}
    # (1) stagewise marker rates. Base/SFT/DPO/final have all matched controls;
    # RLVR-only intermediate stages retain advice+factual.
    for st in STAGES[m]:
        stage_tags = ALL_TAGS if st in FULL_CONDITION_STAGES else STAGE_TAGS
        pm["stagewise_emission"][st]={
            c:{t:mean(prompt_rates(m,st,t,c,EMB_THRESHOLD)) for t in stage_tags}
            for c in CATS
        }
    # (2) final-stage 5-condition table + threshold sweep
    for thr in THRESHOLDS:
        pm["condition_emission_by_threshold"][str(thr)]={
            c:{t:mean(prompt_rates(m,FINAL_STAGE,t,c,thr)) for t in ALL_TAGS} for c in CATS}
    pm["condition_emission"]=pm["condition_emission_by_threshold"][str(EMB_THRESHOLD)]
    # (3) prompt-clustered permutation tests: advice vs each control condition
    for c in CATS:
        adv=prompt_rates(m,FINAL_STAGE,"advice",c,EMB_THRESHOLD)
        pm["tests"][c]={t:{"diff":round(np.mean(adv)-np.mean(prompt_rates(m,FINAL_STAGE,t,c,EMB_THRESHOLD)),3),
                           "perm_p":perm_test(adv,prompt_rates(m,FINAL_STAGE,t,c,EMB_THRESHOLD))}
                        for t in ALL_TAGS if t!="advice"}
    # (4) novelty vs exact pretraining corpus (final stage, seed 0)
    for t in ("advice","factual"):
        nov=[load_ck('attribution',k)["novelty_rate"] for k in final_seed0_keys([t]) if k.startswith(m+"__") and have('attribution',k)]
        pm["novelty"][t]=mean(nov)
    # (5) SFT recoverability by length + role scoping + DPO/RLVR (final seed-0 advice)
    bylen=defaultdict(lambda:{"beh":[],"top":[]}); roles={}; dpo={}; rlvr={}
    for k in final_seed0_keys(["advice"]):
        if not k.startswith(m+"__") or not have('instruction',k): continue
        ins=load_ck('instruction',k)
        for r in ins["behavioral_recoverability"]:
            bylen[r["len"]]["beh"].append(r["recoverability_permil"])
            d=r["detail"][0]; roles[r["phrase"]]={"assistant":d["assistant"],"blob":d["blob"],"inflation":d["inflation"]}
        for r in ins["topical_recoverability"]: bylen[r.get("len",0)]["top"].append(r["recoverability_permil"])
        for d in ins["dpo"]: dpo[d["phrase"]]=d
        for d in ins["rlvr"]: rlvr[d["phrase"]]=d
    pm["recoverability_by_length"]={str(L):{"behavioral_permil":mean(v["beh"]),"topical_permil":mean(v["top"]),
        "ratio":(round(mean(v["beh"])/mean(v["top"]),1) if mean(v["beh"]) and mean(v["top"]) else None),
        "n_beh":len(v["beh"]),"n_top":len(v["top"])} for L,v in sorted(bylen.items()) if L}
    pm["role_scoping"]=sorted([{"phrase":k,**v} for k,v in roles.items()],key=lambda x:-(x["blob"] or 0))
    pm["dpo_analysis"]=sorted(dpo.values(),key=lambda x:-(x["chosen"] or 0))
    pm["rlvr_analysis"]=list(rlvr.values())
    summary["per_model"][m]=pm
save_ck('results','summary_v2',summary)
print(json.dumps({k:v for k,v in summary.items() if k!="per_model"},indent=2))

# Figures: stagewise emergence + 5-condition comparison
fig,axes=plt.subplots(len(MODELS),2,figsize=(14,4.5*len(MODELS)),squeeze=False)
SHOW=["empathy_opener","validation","disclaimer","structure"]
for r,m in enumerate(MODELS):
    ax=axes[r][0]
    for c in SHOW:
        ax.plot(STAGES[m],[summary["per_model"][m]["stagewise_emission"][st][c]["advice"] or 0 for st in STAGES[m]],marker="o",label=c)
    ax.set_title(f"{m}: behavior emergence across training stages (advice)"); ax.set_ylim(0,1.05); ax.legend(fontsize=8)
    ax=axes[r][1]; x=np.arange(len(SHOW)); w=0.8/len(ALL_TAGS)
    for j,t in enumerate(ALL_TAGS):
        ce=summary["per_model"][m]["condition_emission"]
        ax.bar(x+(j-(len(ALL_TAGS)-1)/2)*w,[ce[c][t] or 0 for c in SHOW],w,label=t)
    ax.set_xticks(x); ax.set_xticklabels(SHOW,rotation=20,ha="right")
    ax.set_title(f"{m}: emission by condition (final model)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PROJECT/'results'/'figure_v2.png',dpi=130); plt.show()
print('Saved artifact:',PROJECT/'results'/'summary_v2.json')''')

code(r'''#@title 14b · STAGE F — Stage x condition INTERACTION tests (DiD, permutation) — cheap, uses cached checkpoints only
# Reviewer point: rising advice-side rates across stages could reflect a generic
# post-training drift (more caution/lists everywhere). The formal test is the
# stage x condition INTERACTION: does the advice-minus-factual gap CHANGE between
# adjacent stages? Unit = prompt (each prompt keeps its full stage trajectory);
# permutation = shuffle condition labels across prompts, 10k draws, two-sided.
# Runs entirely from cached 'behavior' checkpoints — no GPU, no generation.
import numpy as np, random as _rnd
def _emits(b,c,thr): return (c in b["lexicon_categories"]) or (b["embedding_max_sims"].get(c,0)>=thr)
def _prompt_traj(model, tag, cat, thr):
    """Per-prompt emission rate at every stage: {prompt_i: {stage: rate}}."""
    out={}
    for i in range(len(PROMPTS[tag])):
        st_rates={}
        for st in STAGES[model]:
            vals=[]
            for s in range(SEEDS):
                key=f"{model}__{st}__{tag}_{i:02d}__s{s}"
                if have('behavior',key): vals.append(1.0 if _emits(load_ck('behavior',key),cat,thr) else 0.0)
            if vals: st_rates[st]=sum(vals)/len(vals)
        if st_rates: out[i]=st_rates
    return out
def _did(adv, fac, s0, s1):
    ga1=np.mean([t[s1] for t in adv]); ga0=np.mean([t[s0] for t in adv])
    gf1=np.mean([t[s1] for t in fac]); gf0=np.mean([t[s0] for t in fac])
    return (ga1-gf1)-(ga0-gf0)
def interaction_test(model, cat, s0, s1, thr=EMB_THRESHOLD, n=10000, seed=0):
    A=_prompt_traj(model,"advice",cat,thr); F=_prompt_traj(model,"factual",cat,thr)
    adv=[t for t in A.values() if s0 in t and s1 in t]
    fac=[t for t in F.values() if s0 in t and s1 in t]
    if not adv or not fac: return None
    obs=_did(adv,fac,s0,s1)
    pool=adv+fac; na=len(adv); _rnd.seed(seed); hits=0
    for _ in range(n):
        _rnd.shuffle(pool)
        if abs(_did(pool[:na],pool[na:],s0,s1))>=abs(obs)-1e-12: hits+=1
    return {"did":round(obs,4),"perm_p":round((hits+1)/(n+1),5),"n_advice":na,"n_factual":len(fac)}
CATS=list(LEXICON)
inter={}
for m in MODELS:
    inter[m]={}
    pairs=list(zip(STAGES[m][:-1],STAGES[m][1:]))+[(STAGES[m][0],STAGES[m][-1])]
    for c in CATS:
        inter[m][c]={}
        for s0,s1 in pairs:
            r=interaction_test(m,c,s0,s1)
            if r: inter[m][c][f"{s0}->{s1}"]=r
        row=" | ".join(f"{k}: {v['did']:+.3f} (p={v['perm_p']})" for k,v in inter[m][c].items())
        print(f"{m} {c:<16} {row}")
save_ck('results','interaction_tests',inter)
print("Saved results/interaction_tests.json — the stage x condition interaction the paper needs.")''')

code(r'''#@title 14c · STAGE G — Reviewer-driven matched-control checkpoint tests
# Strict publication analysis for the added control conditions. This command
# refuses to report completion unless every expected prompt has all five seeds
# at Base, SFT, and DPO and all 24 planned tests are present. The frozen primary
# threshold is 0.82; 0.80 and 0.84 are sensitivity analyses on the same outputs.
import subprocess, sys
matched_reports={}
for threshold in THRESHOLDS:
    threshold_key=str(threshold)
    threshold_output=PROJECT/"results"/f"matched_control_stagewise_tau_{int(round(threshold*100)):02d}.json"
    cmd=[sys.executable,"-m","experiments.matched_control_stagewise",
         "--project",str(PROJECT),"--output",str(threshold_output),
         "--threshold",threshold_key,"--seeds",str(SEEDS),"--permutations","10000"]
    if QUICK_TEST:
        cmd.append("--allow-incomplete")
    subprocess.run(cmd,check=True)
    matched_reports[threshold_key]=json.loads(threshold_output.read_text(encoding="utf-8"))
report=matched_reports[str(EMB_THRESHOLD)]
(PROJECT/"results"/"matched_control_stagewise.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
sweep={"status":"complete" if all(r["status"]=="complete" for r in matched_reports.values()) else "incomplete",
       "primary_threshold":EMB_THRESHOLD,"reports":matched_reports}
(PROJECT/"results"/"matched_control_threshold_sweep.json").write_text(json.dumps(sweep,indent=2),encoding="utf-8")
print("Matched-control status:",report["status"],"tests:",report["n_tests"],
      "| threshold sweep:",sweep["status"])
if report["status"]=="complete":
    subprocess.run([
        sys.executable,"-m","experiments.render_matched_control_results",
        "--report",str(PROJECT/"results"/"matched_control_stagewise.json"),
        "--markdown",str(PROJECT/"results"/"matched_control_stagewise.md"),
        "--latex",str(PROJECT/"results"/"matched_control_stagewise.tex"),
    ],check=True)
    import pandas as pd
    display(pd.DataFrame(report["results"]))
else:
    print("Incomplete cells (smoke test only):",report["incomplete"][:8])''')

md(r"""## Detector validation — STRATIFIED across all 5 conditions, two annotators
Cell 15 exports a labeling sheet sampled across advice / factual / emotional /
neutral-advice / domain-factual (fixes the advice-only validation gap). Fill
`human_label_1` (and optionally `human_label_2` from a second annotator), then run
cell 16 for per-category P/R per condition + Cohen's kappa.""")

code(r'''#@title 15 · Export STRATIFIED validation sheet (~120 sentences, all conditions)
import csv, random as _rnd
_rnd.seed(0)
QUOTA={"advice":40,"factual":30,"emotional":20,"neutral_advice":20,"domain_factual":10}
rows=[]
for t,quota in QUOTA.items():
    pool=[]
    for k in final_seed0_keys([t]):
        if not have('answers',k): continue
        a=load_ck('answers',k)
        for s in _sents(a["answer"]):
            lex={h["category"] for h in find_behaviors(s)}
            sims=embedding_max_sims(s)
            emb={c for c,v in sims.items() if v>=EMB_THRESHOLD}
            pool.append({"key":k,"condition":t,"sentence":s,
                         "predicted":";".join(sorted(lex|emb)) or "none",
                         "human_label_1":"","human_label_2":""})
    _rnd.shuffle(pool); rows+=pool[:quota]
_rnd.shuffle(rows)
p=PROJECT/'results'/'validation_sheet_v2.csv'
with open(p,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=["key","condition","sentence","predicted","human_label_1","human_label_2"])
    w.writeheader(); w.writerows(rows)
print("Wrote",p,f"({len(rows)} sentences) — fill human_label_1 (and _2), then run cell 16.")''')

code(r'''#@title 16 · Score validation: P/R per category & condition + Cohen's kappa
import csv
from collections import defaultdict
p=PROJECT/'results'/'validation_sheet_v2.csv'
rows=[r for r in csv.DictReader(open(p,encoding='utf-8')) if r.get("human_label_1","").strip()]
if not rows: print("No labels found — fill human_label_1 first.")
else:
    def parse(x): return set(v.strip() for v in x.replace(";",",").split(",") if v.strip() and v.strip()!="none")
    cats=list(LEXICON); out={"overall":{},"by_condition":{}}
    def score(sub):
        res={}
        for c in cats:
            tp=sum(1 for r in sub if c in parse(r["predicted"]) and c in parse(r["human_label_1"]))
            fp=sum(1 for r in sub if c in parse(r["predicted"]) and c not in parse(r["human_label_1"]))
            fn=sum(1 for r in sub if c not in parse(r["predicted"]) and c in parse(r["human_label_1"]))
            res[c]={"P":round(tp/(tp+fp),3) if tp+fp else None,"R":round(tp/(tp+fn),3) if tp+fn else None,
                    "tp":tp,"fp":fp,"fn":fn}
        return res
    out["overall"]=score(rows)
    for t in {r["condition"] for r in rows}:
        out["by_condition"][t]=score([r for r in rows if r["condition"]==t])
    both=[r for r in rows if r.get("human_label_2","").strip()]
    if both:
        from sklearn.metrics import cohen_kappa_score
        ks={}
        for c in cats:
            y1=[int(c in parse(r["human_label_1"])) for r in both]
            y2=[int(c in parse(r["human_label_2"])) for r in both]
            ks[c]=round(cohen_kappa_score(y1,y2),3) if len(set(y1))>1 or len(set(y2))>1 else None
        out["cohen_kappa"]=ks; print("Cohen's kappa:",ks)
    save_ck('results','detector_validation_v2',out)
    print(json.dumps(out["overall"],indent=2))
    print("Saved results/detector_validation_v2.json — the per-condition table gives the FACTUAL false-positive rate the paper needs.")''')

md(r"""## Resume / outputs
- **Resume:** re-run all cells after any disconnect; every stage skips cached items.
- **Artifacts:** `results/summary_v2.json` (single source of truth), `figure_v2.png`,
  `validation_sheet_v2.csv` → `detector_validation_v2.json`.
- **What the paper gets:** stagewise emergence curves (which stage produces each
  behavior), the 5-condition emission table with prompt-clustered permutation tests,
  DPO chosen-vs-rejected reinforcement ratios, RLVR counts, exact-corpus novelty,
  and a stratified two-annotator detector validation.""")

# ---------------------------------------------------------------- emit
nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
         if t == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": s.splitlines(keepends=True)})
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).parent / "colab_provenance_experiments_v2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} with {len(CELLS)} cells")
