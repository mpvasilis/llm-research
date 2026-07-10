"""Builds colab_provenance_experiments.ipynb — a self-contained Colab notebook
for the answer-provenance experiments with Google-Drive checkpointing.

Run:  python experiments/build_notebook.py
"""
import json
from pathlib import Path

CELLS = []


def md(src): CELLS.append(("markdown", src))
def code(src): CELLS.append(("code", src))


# ---------------------------------------------------------------- intro
md(r"""# LLM Answer-Provenance Experiments — Colab runner

Traces how an LLM's free-form **advice** answers are shaped by its training data,
in two layers, and runs the **de-risk preliminary** for the causal SFT-ablation study:

- **Pretraining** layer → verbatim n-gram spans vs the exact OLMo-2 Stage-1 corpus **OLMo-Mix-1124** (`v4_olmo-mix-1124_llama`) / **The Pile** (Pythia) via the free **infini-gram** API. (Stage-2 Dolmino-Mix-1124 has no standalone infini-gram index.)
- **Instruction** layer → role-scoped (assistant-only) search of each model's **exact** OLMo-2 SFT mixture (7B: `tulu-3-sft-olmo-2-mixture`; 1B: `tulu-3-sft-olmo-2-mixture-0225`) via local parquet + DuckDB, with a **length-matched, per-million** recoverability metric. oasst1 is NOT searched separately (it is already an internal subset of each mixture).
- **Behavior detection** → both an auditable **lexicon** and an **embedding/paraphrase** detector (e5-small).

### Checkpointing
Everything is written to **Google Drive** as one file per item. If Colab disconnects,
just **re-run all cells** — completed answers/attributions/searches are loaded from Drive
and skipped, so you resume exactly where you left off. Delete a stage folder to recompute it.

### Order
Run cells top to bottom. Set your HF token in **Colab Secrets** (key `HF_TOKEN`) or paste when prompted.
Runs **OLMo-2 1B + 7B** by default (`MODELS` in cell 4) and produces a per-model comparison.
7B needs an **L4/A100** runtime (or tick `LOAD_8BIT` to fit a T4). Drop to `["1B"]` for a free-T4 smoke test.""")

# ---------------------------------------------------------------- setup
code(r"""#@title 1 · Install dependencies
%pip -q install "transformers>=4.47" accelerate bitsandbytes duckdb sentence-transformers huggingface_hub scikit-learn
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")""")

code(r"""#@title 2 · Mount Google Drive + project dirs (checkpoints live here)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/llm-research-experiments')  #@param {type:"string"}
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

code(r"""#@title 4 · Config: models, corpora, prompts (editable)
# Open-data models only (so provenance tracing works). Runs ALL listed sizes and
# produces a per-model comparison. 7B needs A100/L4 (or LOAD_8BIT on a T4).
MODELS = ["1B", "7B"]  #@param {type:"raw"}
LOAD_8BIT  = False #@param {type:"boolean"}
MODEL_IDS = {"1B":"allenai/OLMo-2-0425-1B-Instruct","7B":"allenai/OLMo-2-1124-7B-Instruct",
             "13B":"allenai/OLMo-2-1124-13B-Instruct"}
PRETRAIN_INDEX = "v4_olmo-mix-1124_llama"   # OLMo-2 Stage-1 pretraining (OLMo-Mix-1124).
# Stage-2 mid-training (Dolmino-Mix-1124) has no standalone infini-gram index.
# SFT is MODEL-SPECIFIC: the 0425-1B line trained on tulu-3-sft-olmo-2-mixture-0225,
# the 1124-7B/13B lines on tulu-3-sft-olmo-2-mixture. oasst1 is NOT searched
# separately (it is already an internal subset of each mixture -> double-count).
SFT_BY_MODEL = {"1B":"allenai/tulu-3-sft-olmo-2-mixture-0225",
                "7B":"allenai/tulu-3-sft-olmo-2-mixture",
                "13B":"allenai/tulu-3-sft-olmo-2-mixture"}
SFT_DATASETS = sorted({SFT_BY_MODEL[m] for m in MODELS})  # per-model mixtures actually used
TULU_SHARD_CAP = None   # None = ALL shards. The per-model mixtures are the search targets.
MAX_NEW_TOKENS = 256 #@param {type:"integer"}
TEMPERATURE = 0.7    #@param {type:"number"}
EMB_THRESHOLD = 0.82                 # primary e5 paraphrase threshold
THRESHOLDS = [0.80, 0.82, 0.84]      # sweep reported as a robustness check

PROMPTS = {
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
  "I recently lost a parent and I'm struggling with grief. Any advice?",
  "I think I'm burned out at work. What should I do?",
  "I have trouble falling asleep every night. What can I do?",
  "I get lower back pain from sitting at my desk all day. Any advice?",
  "I'm always tired and have no energy. What can I do?",
  "I get headaches several times a week. What should I do?",
  "I want to lose weight but don't know where to start. Any advice?",
  "I'm drinking too much coffee and feel jittery. How do I cut back?",
  "I get really nervous about public speaking. How can I get better?",
  "How do I make new friends as an adult?",
  "My roommate never cleans up after themselves. How do I deal with it?",
  "My neighbor plays loud music late at night. What should I do?",
  "I'm shy at parties and don't know how to act. Any advice?",
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
 ],
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
 ],
}
print("Models:", [MODEL_IDS[m] for m in MODELS], "| 8-bit:", LOAD_8BIT,
      "| prompts:", len(PROMPTS["advice"]), "advice +", len(PROMPTS["factual"]), "factual",
      "| Tulu shards:", TULU_SHARD_CAP)""")

code(r"""#@title 5 · Checkpoint helpers (one file per item; resume = skip existing)
import json
def ckpt_path(stage, key): return PROJECT/stage/f"{key}.json"
def have(stage, key):      return ckpt_path(stage,key).exists()
def load_ck(stage, key):   return json.loads(ckpt_path(stage,key).read_text(encoding='utf-8'))
def save_ck(stage, key, obj):
    ckpt_path(stage,key).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    return obj
def all_keys():
    # keys namespaced by model: "<model>__<tag>_<i>"
    return [f"{m}__{t}_{i:02d}" for m in MODELS for t,ps in PROMPTS.items() for i in range(len(ps))]
def key_meta(key):
    m,rest = key.split('__',1); t,i = rest.rsplit('_',1); return m, t, int(i)""")

# ---------------------------------------------------------------- inlined tracer
code(r'''#@title 6 · infini-gram client (pretraining corpus search, free API)
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
def ig_count(index,q): return _ig({"index":index,"query_type":"count","query":q})
def ig_passages(index,q,maxnum=2):
    res=_ig({"index":index,"query_type":"search_docs","query":q,"maxnum":maxnum,"max_disp_len":250})
    out=[]
    for d in (res.get("documents") or res.get("docs") or []):
        spans=d.get("spans");
        text="".join(s[0] if isinstance(s,(list,tuple)) else str(s) for s in spans) if spans else d.get("text","")
        raw=d.get("metadata") or d.get("doc_meta") or {}
        if isinstance(raw,str):
            try: raw=json.loads(raw)
            except: raw={"raw":raw}
        out.append({"text":text.strip()[:400],"source":raw})
    return out''')

code(r'''#@title 7 · n-gram attribution (longest verbatim spans vs pretraining)
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
def attribute(index,answer,top_passages=4,max_calls=70):
    ms=longest_matches(index,answer,max_calls=max_calls)
    for m in ms[:top_passages]: m["passages"]=ig_passages(index,m["phrase"],maxnum=2)
    words=max(1,len(tokenize(answer))); covered=set()
    for m in ms:
        for p in range(m["start"],m["start"]+m["words"]): covered.add(p)
    return {"index":index,"matched_spans":ms,"n_matches":len(ms),
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
def behavioral_coverage(text):
    ws=list(_WORD.finditer(text))
    if not ws: return 0.0
    sp=[(h["start"],h["end"]) for h in find_behaviors(text)]
    return sum(1 for w in ws if any(s<=w.start()<e for s,e in sp))/len(ws)

# embedding/paraphrase detector (fix for lexicon under-counting)
from sentence_transformers import SentenceTransformer
import numpy as np
_emb=SentenceTransformer("intfloat/multilingual-e5-small")
def _enc(texts,prefix): return _emb.encode([f"{prefix}: {t}" for t in texts],normalize_embeddings=True)
# seed exemplars per behavioral category (paraphrase anchors)
EXEMPLARS={
 "empathy_opener":["I'm so sorry you're going through this.","That sounds incredibly hard.","I can only imagine how painful this is."],
 "validation":["Your feelings are completely valid.","It's totally normal to feel this way.","There's nothing wrong with feeling upset."],
 "disclaimer":["You should consult a qualified professional.","Please talk to your doctor about this.","I'm not a medical professional, but..."],
 "crisis_referral":["If you're in crisis, please call a helpline.","Reach out to emergency services right away.","Contact a suicide prevention hotline."],
 "structure":["Here are a few steps you can take.","Consider the following suggestions.","First, try this; second, try that."],
}
_EX_VECS={c:_enc(v,"passage") for c,v in EXEMPLARS.items()}
def _sents(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+",text.strip()) if len(s.strip())>3]
def embedding_categories(text,thr=0.82):
    sents=_sents(text)
    if not sents: return {}
    sv=_enc(sents,"query"); out={}
    for c,ev in _EX_VECS.items():
        sim=(sv@ev.T).max(axis=1)   # best exemplar match per sentence
        hit=int((sim>=thr).sum())
        if hit: out[c]=hit
    return out
def embedding_max_sims(text):
    # per-category max cosine sim over sentences; threshold later (robustness sweep)
    sents=_sents(text)
    if not sents: return {c:0.0 for c in _EX_VECS}
    sv=_enc(sents,"query")
    return {c: float((sv@ev.T).max()) for c,ev in _EX_VECS.items()}

# source bucketing (Dolma path -> forum vs curated)
FORUM={"reddit","falcon","cc_tail"}; CURATED={"cc_head","c4","wiki","books","academic"}
def bucket_source(src):
    p=(src.get("path") or src.get("url") or (src.get("metadata",{}) or {}).get("url") or "") if isinstance(src,dict) else str(src or "")
    p=p.lower()
    for key,b in [("reddit","reddit"),("cc_en_head","cc_head"),("cc_en_middle","cc_middle"),
                  ("cc_en_tail","cc_tail"),("falcon","falcon"),("c4","c4"),("wiki","wiki"),
                  ("gutenberg","books"),("pes2o","academic"),("s2orc","academic"),
                  ("stack","code"),("starcoder","code"),("math","math")]:
        if key in p: return b
    return "other"
def source_split(buckets):
    n=len(buckets) or 1
    return {"n":len(buckets),"forum_share":round(sum(b in FORUM for b in buckets)/n,3),
            "curated_share":round(sum(b in CURATED for b in buckets)/n,3),
            "by_bucket":{b:buckets.count(b) for b in sorted(set(buckets))}}''')

code(r'''#@title 9 · Instruction-layer: parquet download + role-scoped, GRADED recoverability
import duckdb
_con=duckdb.connect(); _con.execute("SET enable_progress_bar=false;")
_ASST="len(list_filter(messages, m -> m.role='assistant' AND m.content ILIKE $q))>0"
TEXT_EXPR={"OpenAssistant/oasst1":"text",
           "allenai/tulu-3-sft-mixture":"to_json(messages)",
           "allenai/tulu-3-sft-olmo-2-mixture":"to_json(messages)",
           "allenai/tulu-3-sft-olmo-2-mixture-0225":"to_json(messages)"}
ASST_PRED={"allenai/tulu-3-sft-mixture":_ASST,
           "allenai/tulu-3-sft-olmo-2-mixture":_ASST,
           "allenai/tulu-3-sft-olmo-2-mixture-0225":_ASST,
           "OpenAssistant/oasst1":"role='assistant' AND text ILIKE $q"}
SHARD_CAP={}  # download all shards of each per-model mixture
def _shards(ds):
    u=f"https://huggingface.co/api/datasets/{ds}/tree/refs%2Fconvert%2Fparquet/default/train"
    req=urllib.request.Request(u,headers={"Authorization":f"Bearer {HF_TOKEN}"})
    return [f["path"].split("/")[-1] for f in json.loads(urllib.request.urlopen(req,timeout=30).read()) if f["path"].endswith(".parquet")]
def ensure_parquet(ds):
    files=_shards(ds); cap=SHARD_CAP.get(ds); files=files[:cap] if cap else files
    d=PROJECT/'cache'/ds.replace('/','__'); d.mkdir(parents=True,exist_ok=True); local=[]
    for fn in files:
        p=d/fn
        if not p.exists():
            url=f"https://huggingface.co/datasets/{ds}/resolve/refs%2Fconvert%2Fparquet/default/train/{fn}?download=true"
            req=urllib.request.Request(url,headers={"Authorization":f"Bearer {HF_TOKEN}"})
            print(f"  downloading {ds}/{fn} ...");
            with urllib.request.urlopen(req,timeout=900) as r, open(p,'wb') as o:
                while (ch:=r.read(1<<20)): o.write(ch)
        local.append(str(p))
    return local
_PARQUET={}; _NDOCS={}
def _files(ds):
    if ds not in _PARQUET:
        _PARQUET[ds]=ensure_parquet(ds)
        _NDOCS[ds]=_con.execute("SELECT count(*) FROM read_parquet($f)",{"f":_PARQUET[ds]}).fetchone()[0]
    return _PARQUET[ds]
_ROLE_CACHE={}
def count_roles(ds,phrase):
    k=(ds,phrase)
    if k in _ROLE_CACHE: return _ROLE_CACHE[k]
    f=_files(ds); like=f"%{phrase}%"; be=TEXT_EXPR.get(ds,"text"); ap=ASST_PRED.get(ds)
    blob=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {be} ILIKE $q",{"f":f,"q":like}).fetchone()[0]
    asst=_con.execute(f"SELECT count(*) FROM read_parquet($f) WHERE {ap}",{"f":f,"q":like}).fetchone()[0] if ap else None
    # GRADED metric: assistant matches per-million assistant-containing docs (size-fair)
    permil=round(1e6*asst/_NDOCS[ds],2) if asst is not None and _NDOCS[ds] else None
    r={"dataset":ds,"phrase":phrase,"assistant":asst,"blob":blob,
       "inflation":round(blob/asst,2) if asst else None,"per_million":permil,"n_docs":_NDOCS[ds]}
    _ROLE_CACHE[k]=r; return r
def recoverability(phrase, model):
    """Assistant per-million against the EXACT mixture this model was trained on
    (per-model, not summed across mixtures)."""
    ds=SFT_BY_MODEL[model]; r=count_roles(ds,phrase)
    return {"phrase":phrase,"recoverability_permil":round(r["per_million"] or 0.0,2),"detail":[r]}''')

# ---------------------------------------------------------------- stages
code(r'''#@title 10 · STAGE A — Generate answers per model (GPU) · checkpointed
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
def gen_with(model, tok, q):
    enc=tok.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,
                                return_tensors="pt",return_dict=True).to(model.device)
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=TEMPERATURE>0,
                           temperature=TEMPERATURE,top_p=0.9,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
for m in MODELS:                       # load each model once
    pending=[k for k in all_keys() if k.startswith(m+"__") and not have('answers',k)]
    if not pending: print(m,'all cached'); continue
    mid=MODEL_IDS[m]; print('loading',mid,'…',flush=True)
    tok=AutoTokenizer.from_pretrained(mid)
    if LOAD_8BIT:   # recent transformers: 8-bit goes via BitsAndBytesConfig, not load_in_8bit=
        from transformers import BitsAndBytesConfig
        kw=dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map="auto")
    else:
        kw=dict(torch_dtype="auto",device_map="auto")
    model=AutoModelForCausalLM.from_pretrained(mid,**kw); model.eval()
    for key in pending:
        _m,t,i=key_meta(key); q=PROMPTS[t][i]; print('  gen',key,flush=True)
        save_ck('answers',key,{"key":key,"model":m,"model_id":mid,"tag":t,"i":i,"question":q,
                               "answer":gen_with(model,tok,q)})
    del model,tok; gc.collect(); torch.cuda.empty_cache()
print('Stage A done:',len(list((PROJECT/"answers").glob("*.json"))),'answers')''')

code(r'''#@title 11 · STAGE B — Pretraining attribution (infini-gram) · checkpointed
for key in all_keys():
    if have('attribution',key): print(key,'cached'); continue
    a=load_ck('answers',key); print('attributing',key,'…',flush=True)
    save_ck('attribution',key,{"key":key,**attribute(PRETRAIN_INDEX,a["answer"],max_calls=70)})
print('Stage B done.')''')

code(r'''#@title 12 · STAGE C — Behavior detection (lexicon + embedding) · checkpointed
# Store per-category MAX embedding similarity (not a fixed-threshold count) so Stage E
# can sweep thresholds (0.80/0.82/0.84) without re-encoding -> a robustness check.
for key in all_keys():
    if have('behavior',key): print(key,'cached'); continue
    a=load_ck('answers',key); ans=a["answer"]
    save_ck('behavior',key,{"key":key,"lexicon_categories":lexicon_categories(ans),
        "embedding_categories":embedding_categories(ans,EMB_THRESHOLD),
        "embedding_max_sims":{c:round(s,4) for c,s in embedding_max_sims(ans).items()},
        "behavioral_coverage":round(behavioral_coverage(ans),4)})
print('Stage C done.')''')

code(r'''#@title 13 · STAGE D — Instruction-layer recoverability (LENGTH-MATCHED) + source split
# Fix for the phrase-length confound: compare each behavioral phrase to topical
# n-grams of the SAME word-length (not single words). Comparing multiword phrases
# to single words spuriously favors topical (short strings match more docs).
_STOP=set("the a an and or but if then of to in on at for with about into over after is are was were be been being do does did have has had you your my me it this that these those can could would should will what how why when".split())
def body_ngrams(ans,length,beh_spans,k=3):
    toks=list(_WORD.finditer(ans)); out=[]; seen=set()
    for i in range(len(toks)-length+1):
        g=toks[i:i+length]; s,e=g[0].start(),g[-1].end()
        if any(bs<=s<be or bs<e<=be for bs,be in beh_spans): continue
        words=[t.group().lower() for t in g]
        if all(w in _STOP for w in words): continue
        key=" ".join(words)
        if key in seen: continue
        seen.add(key); out.append(ans[s:e])
        if len(out)>=k: break
    return out
for key in all_keys():
    if have('instruction',key): print(key,'cached'); continue
    mdl,_t,_i=key_meta(key)   # search each model's answer against ITS OWN SFT mixture
    a=load_ck('answers',key); attr=load_ck('attribution',key); ans=a["answer"]
    beh=find_behaviors(ans); beh_spans=[(h["start"],h["end"]) for h in beh]
    beh_phrases=sorted({h["phrase"] for h in beh})
    beh_rec=[recoverability(p,mdl) for p in beh_phrases]
    # length-matched topical controls: for each behavioral phrase length, body n-grams of that length
    lengths={len(p.split()) for p in beh_phrases}
    top_rec=[]
    for L in lengths:
        for g in body_ngrams(ans,L,beh_spans): top_rec.append({**recoverability(g,mdl),"len":L})
    for r in beh_rec: r["len"]=len(r["phrase"].split())
    beh_buckets=[]
    for p in beh_phrases:
        for ps in ig_passages(PRETRAIN_INDEX,p,maxnum=2): beh_buckets.append(bucket_source(ps["source"]))
    body_buckets=[]
    for m in attr["matched_spans"][:4]:
        if not find_behaviors(m["phrase"]):
            for ps in m.get("passages",[]): body_buckets.append(bucket_source(ps["source"]))
    save_ck('instruction',key,{"key":key,
        "behavioral_recoverability":beh_rec,"topical_recoverability":top_rec,
        "behavioral_recov_mean":round(sum(r["recoverability_permil"] for r in beh_rec)/len(beh_rec),2) if beh_rec else None,
        "topical_recov_mean":round(sum(r["recoverability_permil"] for r in top_rec)/len(top_rec),2) if top_rec else None,
        "behavioral_source":source_split(beh_buckets),"body_source":source_split(body_buckets)})
print('Stage D done (length-matched).')''')

code(r'''#@title 14 · STAGE E — Per-model aggregate -> ONE reproducible summary.json + plot
import numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
def mean(xs): xs=[x for x in xs if x is not None]; return round(sum(xs)/len(xs),3) if xs else None
CATS=list(LEXICON)
def cats_at(lex,sims,t): return set(lex)|{c for c,s in sims.items() if s>=t}
def summarize(model):
    rows=[]; bylen=defaultdict(lambda:{"beh":[],"top":[]}); roles={}
    for key in [k for k in all_keys() if k.startswith(model+"__")]:
        _m,t,i=key_meta(key); b=load_ck('behavior',key); attr=load_ck('attribution',key); ins=load_ck('instruction',key)
        rows.append({"tag":t,"lex":set(b["lexicon_categories"]),"sims":b.get("embedding_max_sims",{}),
                     "cov":b["behavioral_coverage"],"novelty":attr["novelty_rate"],
                     "forum":ins["behavioral_source"]["forum_share"]})
        if t=="advice":
            for r in ins["behavioral_recoverability"]:
                bylen[r.get("len",len(r["phrase"].split()))]["beh"].append(r["recoverability_permil"])
                a=sum(d.get("assistant") or 0 for d in r.get("detail",[]))
                bl=sum(d.get("blob") or 0 for d in r.get("detail",[]))
                roles[r["phrase"]]={"assistant":a,"blob":bl,"inflation":round(bl/a,2) if a else None}
            for r in ins["topical_recoverability"]:
                bylen[r.get("len",0)]["top"].append(r["recoverability_permil"])
    adv=[r for r in rows if r["tag"]=="advice"]; fac=[r for r in rows if r["tag"]=="factual"]
    def emis(rs,c,thr): return round(sum(c in cats_at(r["lex"],r["sims"],thr) for r in rs)/max(1,len(rs)),3)
    per_length={str(L):{"behavioral_permil":mean(v["beh"]),"topical_permil":mean(v["top"]),
       "ratio":(round(mean(v["beh"])/mean(v["top"]),1) if mean(v["beh"]) and mean(v["top"]) else None),
       "n_beh":len(v["beh"]),"n_top":len(v["top"])} for L,v in sorted(bylen.items()) if L}
    return {"model_id":MODEL_IDS[model],"n_advice":len(adv),"n_factual":len(fac),
     "emb_threshold":EMB_THRESHOLD,
     "emission_rate":{c:{"advice":emis(adv,c,EMB_THRESHOLD),"factual":emis(fac,c,EMB_THRESHOLD)} for c in CATS},
     "emission_rate_by_threshold":{str(thr):{c:{"advice":emis(adv,c,thr),"factual":emis(fac,c,thr)} for c in CATS} for thr in THRESHOLDS},
     "behavioral_coverage":{"advice":mean([r["cov"] for r in adv]),"factual":mean([r["cov"] for r in fac])},
     "recoverability_by_length":per_length,
     "role_scoping":sorted([{"phrase":k,**v} for k,v in roles.items()],key=lambda x:-(x["blob"] or 0)),
     "behavioral_forum_share":{"advice":mean([r["forum"] for r in adv]),"factual":mean([r["forum"] for r in fac])},
     "novelty":{"advice":mean([r["novelty"] for r in adv]),"factual":mean([r["novelty"] for r in fac])}}
summary={"detector":"lexicon+e5-embedding","pretrain_index":PRETRAIN_INDEX,
         "sft_datasets":SFT_DATASETS,"per_model":{m:summarize(m) for m in MODELS}}
save_ck('results','summary',summary)
print(json.dumps(summary,indent=2))

# Plot: emission per model (advice) + recoverability-by-length per model
fig,ax=plt.subplots(1,2,figsize=(13,4)); w=0.8/len(MODELS); x=np.arange(len(CATS))
for j,m in enumerate(MODELS):
    er=summary["per_model"][m]["emission_rate"]
    ax[0].bar(x+(j-(len(MODELS)-1)/2)*w,[er[c]["advice"] for c in CATS],w,label=m)
ax[0].set_xticks(x); ax[0].set_xticklabels(CATS,rotation=30,ha="right")
ax[0].set_title("Advice-behavior emission by model"); ax[0].legend()
for m in MODELS:
    pl=summary["per_model"][m]["recoverability_by_length"]
    Ls=sorted(pl,key=int)
    ax[1].plot([f"{L}w" for L in Ls],[pl[L]["ratio"] or 0 for L in Ls],marker="o",label=m)
ax[1].set_title("Behavioral/topical recoverability ratio by length"); ax[1].set_yscale("log"); ax[1].legend()
plt.tight_layout(); plt.savefig(PROJECT/'results'/'figure1.png',dpi=130); plt.show()
print('Saved the single-source artifact to',PROJECT/'results'/'summary.json')''')

md(r"""## (Optional) Detector validation — precision/recall of the behavior detector
Run cell 15 to export a labeling sheet (sampled sentences + the detector's prediction). Open the CSV
in Drive, fill the `human_label` column (the category you judge the sentence to express, or `none`),
then run cell 16 to compute per-category precision/recall. This addresses the "uncalibrated detector"
limitation. Skip both cells if you don't want to hand-label.""")

code(r'''#@title 15 · (Optional) Export detector-validation labeling sheet
import csv, random as _rnd
_rnd.seed(0)
rows=[]
for key in all_keys():
    a=load_ck('answers',key)
    for s in _sents(a["answer"]):
        lex={h["category"] for h in find_behaviors(s)}
        sims=embedding_max_sims(s)
        emb={c for c,v in sims.items() if v>=EMB_THRESHOLD}
        pred=sorted(lex|emb)
        rows.append({"key":key,"sentence":s,"predicted":";".join(pred) or "none","human_label":""})
_rnd.shuffle(rows); rows=rows[:80]   # sample 80 sentences to label
p=PROJECT/'results'/'validation_sheet.csv'
with open(p,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=["key","sentence","predicted","human_label"]); w.writeheader(); w.writerows(rows)
print("Wrote",p,"-- fill the human_label column (one category, comma-separated, or 'none'), then run cell 16.")''')

code(r'''#@title 16 · (Optional) Compute detector precision/recall from filled sheet
import csv
p=PROJECT/'results'/'validation_sheet.csv'
rows=list(csv.DictReader(open(p,encoding='utf-8')))
labeled=[r for r in rows if r.get("human_label","").strip()!=""]
if not labeled:
    print("No labels found in",p,"- fill the human_label column first.")
else:
    cats=list(LEXICON); tp={c:0 for c in cats}; fp={c:0 for c in cats}; fn={c:0 for c in cats}
    for r in labeled:
        pred=set(x for x in r["predicted"].split(";") if x and x!="none")
        gold=set(x.strip() for x in r["human_label"].replace(";",",").split(",") if x.strip() and x.strip()!="none")
        for c in cats:
            if c in pred and c in gold: tp[c]+=1
            elif c in pred and c not in gold: fp[c]+=1
            elif c not in pred and c in gold: fn[c]+=1
    print(f"{'category':<16} {'P':>6} {'R':>6}  (n_labeled={len(labeled)})")
    out={}
    for c in cats:
        P=tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else None
        R=tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else None
        out[c]={"precision":P,"recall":R,"tp":tp[c],"fp":fp[c],"fn":fn[c]}
        print(f"{c:<16} {('%.2f'%P) if P is not None else '  - ':>6} {('%.2f'%R) if R is not None else '  - ':>6}")
    save_ck('results','detector_validation',out); print("Saved to results/detector_validation.json")''')

md(r"""## Resume / scale / next steps
- **Resume after a disconnect:** just re-run all cells. Each stage skips items already in Drive. Nothing recomputes.
- **Recompute a stage:** delete its folder under the project dir (e.g. `answers/`) and re-run.
- **Statistical power:** ships with 40 advice + 20 factual prompts, the full per-model OLMo-2 SFT mixtures (all shards, ~1.3–1.4 GB each, downloaded to Drive once), and a 0.80/0.82/0.84 threshold sweep. For a third scale point add `"13B"` to `MODELS` (needs A100-40GB; it also uses `tulu-3-sft-olmo-2-mixture`).
- **Provenance scope:** the pretraining index is the exact OLMo-2 Stage-1 corpus (`v4_olmo-mix-1124_llama`); the SFT layer is each model's exact mixture. Only Stage-2 Dolmino-Mix-1124 is not covered (no infini-gram index).
- **Toward the causal study:** this notebook produces the *observational* + *de-risk* numbers. The causal leave-cluster-out re-SFT (GPU, ~72 H100-hrs/run) is the separate next phase; these results decide whether to fund it.""")

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
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).parent / "colab_provenance_experiments.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} with {len(CELLS)} cells")
