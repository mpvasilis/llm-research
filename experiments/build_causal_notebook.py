"""Builds colab_causal_ablation.ipynb — phase-2 causal leave-cluster-out SFT.

Trains OLMo-2-1B *base* on Tulu-3 with vs. without a provenance-selected cluster,
and measures whether a target advice behavior drops MORE than for size-matched
coherent-distractor and random controls (the causal claim). Drive-checkpointed
per condition so a Colab disconnect resumes.

Run:  python experiments/build_causal_notebook.py
"""
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md(r"""# Causal SFT-Ablation — leave-cluster-out re-training (phase 2)

**Question.** Does removing a *human-auditable, provenance-selected* cluster of SFT conversations
(empathy / disclaimer / structure examples) and re-running SFT make the corresponding **advice
behavior drop more** than removing a **size-matched coherent-distractor** cluster or a
**size-matched random** set? If yes → causal, auditable evidence for the behavior's SFT origin.

**Conditions** (each = one SFT run from the OLMo-2-1B *base* checkpoint):
| id | training data |
|---|---|
| `base_no_sft` | none (eval base model — floor) |
| `full_sft` | full subset (baseline) |
| `minus_behavioral` | subset − auditable behavioral cluster |
| `minus_distractor` | subset − coherent distractor cluster (size-matched) |
| `minus_random` | subset − random set (size-matched) |

**Causal test:** `emission(full_sft) − emission(minus_behavioral)` should exceed the drops from
`minus_distractor` and `minus_random`, and be **specific** (removing disclaimers drops disclaimers, not empathy).

**⚠ Compute honesty.** Defaults are a *tractable first-signal* config (N≈20k convos, 1 seed, full
removal) that fits a single Colab **A100/L4** session (~2–6 h total, checkpointed). The
**publishable** run needs the official recipe, full Tülu, **multiple seeds**, and dose-response
(≈72 H100-hrs/run × ~15–24 runs ≈ the $2–6k budget) — not free Colab. This notebook produces the
pilot causal signal that decides whether to fund that.

**Checkpointing.** Each condition's eval is one JSON in Drive; re-run all cells to resume (finished
conditions skipped). Trained weights stay on local Colab disk (not Drive) to save space; set
`SAVE_MODELS_TO_DRIVE=True` to persist them.""")

code(r"""#@title 1 · Install
%pip -q install "transformers>=4.47" "trl>=0.12" peft accelerate datasets bitsandbytes duckdb sentence-transformers scikit-learn huggingface_hub
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — set Runtime>GPU")""")

code(r"""#@title 2 · Mount Drive + dirs
from google.colab import drive; drive.mount('/content/drive')
from pathlib import Path
PROJECT = Path('/content/drive/MyDrive/llm-research-experiments')  #@param {type:"string"}
PROJECT = Path(str(PROJECT))
for s in ['clusters','eval','results','models']: (PROJECT/s).mkdir(parents=True, exist_ok=True)
print('Project:', PROJECT)""")

code(r"""#@title 3 · HF token
import os
HF_TOKEN=""
try:
    from google.colab import userdata; HF_TOKEN=userdata.get('HF_TOKEN')
except Exception: pass
if not HF_TOKEN:
    from getpass import getpass; HF_TOKEN=getpass('HF token: ').strip()
os.environ['HF_TOKEN']=HF_TOKEN; print('token set:', bool(HF_TOKEN))""")

code(r"""#@title 4 · Config
BASE_MODEL   = "allenai/OLMo-2-0425-1B"   # the PRE-SFT base checkpoint
SFT_SOURCE   = "allenai/tulu-3-sft-olmo-2-mixture-0225"  # exact 1B OLMo-2 mixture
SUBSET_N     = 20000   #@param {type:"integer"}   # convos sampled from Tulu (tractable)
METHOD       = "full"  #@param ["full","lora"]      # full-FT = cleaner causal claim; lora = fits T4
EPOCHS       = 2       #@param {type:"number"}
LR           = 2e-5    #@param {type:"number"}
MAXLEN       = 2048    #@param {type:"integer"}
BATCH        = 2       #@param {type:"integer"}
GRAD_ACCUM   = 8       #@param {type:"integer"}
TARGET_BEHAVIOR = "disclaimer"  #@param ["disclaimer","empathy_opener","structure","validation"]
SEED         = 0       #@param {type:"integer"}
SAVE_MODELS_TO_DRIVE = False  #@param {type:"boolean"}
CONDITIONS = ["base_no_sft","full_sft","minus_behavioral","minus_distractor","minus_random"]
EVAL_PROMPTS = {
 "advice":[
  "I broke up with my girlfriend without any reason and I feel terrible. What should I do?",
  "How do I stop snoring while sleeping with my girlfriend?",
  "I've been feeling really anxious lately and can't sleep. Any advice?",
  "My partner and I keep fighting about small things. How can we communicate better?",
  "I feel lonely all the time even when I'm around people. What can I do?",
  "I think I'm having panic attacks. What should I do when one happens?",
  "I'm really stressed about work and it's affecting my health. What can I do?",
  "My friend is going through a hard time and I don't know how to support them."],
 "factual":[
  "What is the capital of Australia?","How does photosynthesis work?",
  "What causes the seasons on Earth?","Explain how a bicycle gear system works."],
}
print(f"{BASE_MODEL} | N={SUBSET_N} | {METHOD} | target={TARGET_BEHAVIOR} | seed={SEED}")""")

code(r"""#@title 5 · Checkpoint helpers + behavior detectors (lexicon + embedding)
import json, re, numpy as np
def ck(stage,key): return PROJECT/stage/f"{key}.json"
def have(stage,key): return ck(stage,key).exists()
def load_ck(stage,key): return json.loads(ck(stage,key).read_text(encoding='utf-8'))
def save_ck(stage,key,o): ck(stage,key).write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8'); return o

LEXICON={
 "empathy_opener":["i'm sorry to hear","i am sorry to hear","i'm really sorry","that sounds really","that sounds very","i can understand","i can imagine","it's understandable","that must be","i'm here for you","i hear you","it sounds like you"],
 "validation":["your feelings are valid","it's okay to feel","it's normal to feel","it's completely normal","what you're feeling","there's nothing wrong with","it's natural to"],
 "disclaimer":["consult a","talk to a","speak with a","see a doctor","seek medical","i am not a doctor","i'm not a doctor","medical professional","healthcare professional","mental health professional","seek professional","professional help"],
 "crisis_referral":["crisis","hotline","helpline","988","call 911","emergency services","suicide"],
 "structure":["here are some","here are a few","consider the following","steps you can take","a few things you can","first,","secondly,"],
}
ALL_PHRASES=[(c,p) for c,ps in LEXICON.items() for p in ps]
def find_behaviors(t):
    low=t.lower(); out=[]
    for c,p in ALL_PHRASES:
        i=low.find(p)
        while i!=-1: out.append((c,p,i)); i=low.find(p,i+1)
    return out
def lex_categories(t): return sorted({c for c,_,_ in find_behaviors(t)})

from sentence_transformers import SentenceTransformer
_emb=SentenceTransformer("intfloat/multilingual-e5-small")
def enc(ts,pfx): return _emb.encode([f"{pfx}: {x}" for x in ts],normalize_embeddings=True,batch_size=256,show_progress_bar=False)
EXEMPLARS={
 "empathy_opener":["I'm so sorry you're going through this.","That sounds incredibly hard."],
 "validation":["Your feelings are completely valid.","It's totally normal to feel this way."],
 "disclaimer":["You should consult a qualified professional.","Please talk to your doctor about this."],
 "crisis_referral":["If you're in crisis, please call a helpline.","Contact a suicide prevention hotline."],
 "structure":["Here are a few steps you can take.","Consider the following suggestions."],
}
EXV={c:enc(v,"passage") for c,v in EXEMPLARS.items()}
def _sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+",t.strip()) if len(s.strip())>3]
def emb_categories(t,thr=0.82):
    s=_sents(t)
    if not s: return {}
    sv=enc(s,"query"); out={}
    for c,ev in EXV.items():
        h=int(((sv@ev.T).max(axis=1)>=thr).sum())
        if h: out[c]=h
    return out
def categories(t): return set(lex_categories(t))|set(emb_categories(t).keys())""")

code(r"""#@title 6 · STAGE 0 — Load Tulu subset + build selector clusters (checkpointed)
import random
from datasets import load_dataset
random.seed(SEED)
if have('clusters','manifest'):
    man=load_ck('clusters','manifest'); print('clusters cached:',{k:len(v) for k,v in man.items() if isinstance(v,list)})
else:
    print('streaming Tulu subset…')
    ds=load_dataset(SFT_SOURCE,split='train',streaming=True,token=HF_TOKEN)
    convos=[]
    for ex in ds:
        convos.append(ex['messages']);
        if len(convos)>=SUBSET_N: break
    def asst_text(m): return " ".join(t.get('content','') for t in m if t.get('role')=='assistant')
    asst=[asst_text(m) for m in convos]
    # auditable behavioral selector: assistant turns matching TARGET_BEHAVIOR phrases
    beh_ids=[i for i,a in enumerate(asst) if any(p in a.lower() for p in LEXICON[TARGET_BEHAVIOR])]
    k=len(beh_ids); print('behavioral cluster size:',k)
    # coherent distractor: KMeans on assistant embeddings, pick a non-behavioral cluster, size-matched
    from sklearn.cluster import KMeans
    sample_idx=list(range(len(asst)))
    V=enc([asst[i][:500] for i in sample_idx],"passage")
    km=KMeans(n_clusters=20,n_init=4,random_state=SEED).fit(V)
    labels=km.labels_; beh_set=set(beh_ids)
    from collections import Counter
    # cluster with fewest behavioral members and >=k items = coherent distractor
    cl_counts={c:[i for i in sample_idx if labels[i]==c] for c in range(20)}
    cand=sorted(cl_counts.values(),key=lambda ids:(sum(i in beh_set for i in ids), -len(ids)))
    # primarily the cleanest coherent cluster; pad from next-cleanest if short, to size-match k
    distractor=[]; ci=0
    while len(distractor)<k and ci<len(cand):
        distractor+=[i for i in cand[ci] if i not in beh_set and i not in distractor]; ci+=1
    distractor=distractor[:k]
    pool=[i for i in range(len(asst)) if i not in beh_set]
    random.shuffle(pool); rand_ids=pool[:k]
    if k<20: print(f"WARNING: behavioral cluster is small (k={k}); raise SUBSET_N or pick a more common TARGET_BEHAVIOR for a detectable effect.")
    man={"n":len(convos),"target":TARGET_BEHAVIOR,
         "behavioral":beh_ids,"distractor":distractor,"random":rand_ids}
    save_ck('clusters','manifest',man)
    # persist the convos to local disk for training (not Drive)
    import pickle
    with open('/content/convos.pkl','wb') as f: pickle.dump(convos,f)
    print('clusters:',{k2:len(v) for k2,v in man.items() if isinstance(v,list)})""")

code(r"""#@title 7 · Training + eval functions
import torch, pickle, gc, dataclasses, inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
tok=AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.chat_template is None:   # base OLMo-2 has NO chat template; Instruct sibling shares the vocab
    print("base tokenizer has no chat_template; loading", BASE_MODEL+"-Instruct", "template")
    tok=AutoTokenizer.from_pretrained(BASE_MODEL+"-Instruct")
if tok.pad_token is None: tok.pad_token=tok.eos_token
# bf16 only on Ampere+ (A100/L4); T4 is fp16 — auto-detect so it runs anywhere.
BF16OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE  = torch.bfloat16 if BF16OK else torch.float16
print("precision:", "bf16" if BF16OK else "fp16")

def _load_convos():
    try:
        with open('/content/convos.pkl','rb') as f: return pickle.load(f)
    except FileNotFoundError:
        from datasets import load_dataset
        ds=load_dataset(SFT_SOURCE,split='train',streaming=True,token=HF_TOKEN); c=[]
        for ex in ds:
            c.append(ex['messages'])
            if len(c)>=SUBSET_N: break
        with open('/content/convos.pkl','wb') as f: pickle.dump(c,f)
        return c

def build_text_dataset(remove_ids):
    convos=_load_convos(); rm=set(remove_ids)
    from datasets import Dataset
    texts=[tok.apply_chat_template(m,tokenize=False) for i,m in enumerate(convos) if i not in rm]
    return Dataset.from_dict({"text":texts})

def train_condition(cond, remove_ids):
    # No device_map on a trainable model (fragile with Trainer); let Trainer place it on GPU.
    model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=DTYPE)
    if METHOD=="lora":
        from peft import LoraConfig, get_peft_model
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
              target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
    dset=build_text_dataset(remove_ids)
    out=f"/content/sft_{cond}"
    # TRL renamed max_seq_length -> max_length (>=0.20). Build kwargs by introspection so
    # the SAME notebook works across TRL versions.
    fields={f.name for f in dataclasses.fields(SFTConfig)}
    kw=dict(output_dir=out,num_train_epochs=EPOCHS,per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,learning_rate=LR,
            bf16=BF16OK,fp16=not BF16OK,logging_steps=25,save_strategy="no",report_to=[],seed=SEED)
    if "dataset_text_field" in fields: kw["dataset_text_field"]="text"
    kw["max_length" if "max_length" in fields else "max_seq_length"]=MAXLEN
    cfg=SFTConfig(**kw)
    # SFTTrainer renamed tokenizer -> processing_class; pass whichever it accepts.
    sig=inspect.signature(SFTTrainer.__init__).parameters
    tkw={"processing_class":tok} if "processing_class" in sig else ({"tokenizer":tok} if "tokenizer" in sig else {})
    tr=SFTTrainer(model=model,args=cfg,train_dataset=dset,**tkw)
    tr.train()
    if SAVE_MODELS_TO_DRIVE: tr.save_model(str(PROJECT/'models'/cond))
    return model

@torch.no_grad()
def gen(model,q):
    enc_=tok.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,
            return_tensors="pt",return_dict=True).to(model.device)
    o=model.generate(**enc_,max_new_tokens=256,do_sample=True,temperature=0.7,top_p=0.9,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc_["input_ids"].shape[-1]:],skip_special_tokens=True).strip()

def eval_model(model):
    rows=[]
    for tag,ps in EVAL_PROMPTS.items():
        for i,q in enumerate(ps):
            a=gen(model,q); rows.append({"tag":tag,"q":q,"answer":a,"cats":sorted(categories(a))})
    cats=list(LEXICON)
    adv=[r for r in rows if r["tag"]=="advice"]
    emis={c:round(sum(c in r["cats"] for r in adv)/max(1,len(adv)),3) for c in cats}
    return {"emission_advice":emis,"answers":rows}

def free(model):
    del model; gc.collect(); torch.cuda.empty_cache()""")

code(r"""#@title 8 · STAGE 1 — Run all conditions (checkpointed; resumes on rerun)
man=load_ck('clusters','manifest')
REMOVE={"full_sft":[],"minus_behavioral":man["behavioral"],
        "minus_distractor":man["distractor"],"minus_random":man["random"]}
for cond in CONDITIONS:
    if have('eval',cond): print(cond,'cached'); continue
    print('=== condition:',cond,'===',flush=True)
    if cond=="base_no_sft":
        m=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=DTYPE,device_map="auto")
    else:
        m=train_condition(cond,REMOVE[cond])
    ev=eval_model(m); ev["condition"]=cond; ev["n_removed"]=len(REMOVE.get(cond,[]))
    save_ck('eval',cond,ev); free(m); print(cond,'emission:',ev["emission_advice"])
print('Stage 1 done.')""")

code(r"""#@title 9 · STAGE 2 — Causal table + plot
import matplotlib.pyplot as plt, numpy as np
evals={c:load_ck('eval',c) for c in CONDITIONS if have('eval',c)}
tb=TARGET_BEHAVIOR
base=evals.get("full_sft",{}).get("emission_advice",{}).get(tb,0)
def drop(c): return round(base-evals[c]["emission_advice"].get(tb,0),3) if c in evals else None
result={"target":tb,"full_sft_emission":base,
        "drop_minus_behavioral":drop("minus_behavioral"),
        "drop_minus_distractor":drop("minus_distractor"),
        "drop_minus_random":drop("minus_random"),
        "base_no_sft_emission":evals.get("base_no_sft",{}).get("emission_advice",{}).get(tb)}
causal = (result["drop_minus_behavioral"] or 0) > max(result["drop_minus_distractor"] or 0, result["drop_minus_random"] or 0)
result["causal_signal"]=bool(causal)
save_ck('results','causal_summary',result)
print(json.dumps(result,indent=2))
print('\\nCAUSAL SIGNAL (targeted drop > control drops):', causal)

cats=list(LEXICON); conds=[c for c in CONDITIONS if c in evals]
M=np.array([[evals[c]["emission_advice"][k] for k in cats] for c in conds])
plt.figure(figsize=(10,4)); im=plt.imshow(M,aspect='auto',cmap='Greens',vmin=0,vmax=1)
plt.xticks(range(len(cats)),cats,rotation=30,ha='right'); plt.yticks(range(len(conds)),conds)
plt.colorbar(im,label='advice emission rate'); plt.title(f'Behavior emission by condition (target={tb})')
for i in range(len(conds)):
    for j in range(len(cats)): plt.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.savefig(PROJECT/'results'/'causal_heatmap.png',dpi=130); plt.show()""")

md(r"""## Reading the result & next steps
- **Causal signal = True** when the targeted-cluster removal drops the target behavior **more** than both controls. That's the pilot evidence that the auditable selector picks a *causally* responsible cluster.
- **Specificity check:** in the heatmap, `minus_behavioral` should darken the **target** column far more than other behavior columns.
- **This is a pilot, not the paper.** For a publishable claim add: multiple **seeds** (rerun with `SEED=1,2,…`; each is a fresh checkpoint set), **dose-response** (remove 25/50/100% — subsample the cluster id list), **capability-regression** (eval MMLU/GSM8K so you show you didn't just damage the model), and **selection-set overlap** vs a probe/LESS selector. Then scale N and use the official open-instruct recipe.
- **Resume:** re-run all cells; finished conditions load from Drive. Change `SEED` or `TARGET_BEHAVIOR` to start a new, separately-checkpointed run (namespace the project dir per run for cleanliness).
- **Out of GPU/time:** set `METHOD="lora"` and lower `SUBSET_N`; or persist models with `SAVE_MODELS_TO_DRIVE=True` and continue later.""")

nb={"cells":[({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)} if t=="markdown"
              else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)})
             for t,s in CELLS],
    "metadata":{"colab":{"provenance":[],"toc_visible":True},"kernelspec":{"name":"python3","display_name":"Python 3"},
                "language_info":{"name":"python"},"accelerator":"GPU"},
    "nbformat":4,"nbformat_minor":0}
out=Path(__file__).parent/"colab_causal_ablation.ipynb"
out.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {out} with {len(CELLS)} cells")
