"""Validate the causal notebook's NON-GPU logic against the installed TRL/peft/datasets
(esp. the SFTConfig/SFTTrainer API-drift fixes), without GPU training.

Run:  python -m experiments._validate_causal
"""
import dataclasses
import inspect
import json

import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

ROOT_TOK = "allenai/OLMo-2-0425-1B"
HF_TOKEN = ""
import pathlib
for line in (pathlib.Path(__file__).parent.parent / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("HF_TOKEN="):
        HF_TOKEN = line.split("=", 1)[1].strip()

print("=== 1. SFTConfig API-drift fix ===")
fields = {f.name for f in dataclasses.fields(SFTConfig)}
print("has max_length:", "max_length" in fields, "| has max_seq_length:", "max_seq_length" in fields,
      "| has dataset_text_field:", "dataset_text_field" in fields)
kw = dict(output_dir="/tmp/sft_test", num_train_epochs=1, per_device_train_batch_size=2,
          gradient_accumulation_steps=8, learning_rate=2e-5, bf16=False, fp16=False,
          logging_steps=25, save_strategy="no", report_to=[], seed=0)
if "dataset_text_field" in fields: kw["dataset_text_field"] = "text"
kw["max_length" if "max_length" in fields else "max_seq_length"] = 256
cfg = SFTConfig(**kw)
print("SFTConfig constructed OK; max field used:",
      "max_length" if "max_length" in fields else "max_seq_length")

print("\n=== 2. SFTTrainer signature ===")
sig = inspect.signature(SFTTrainer.__init__).parameters
print("processing_class:", "processing_class" in sig, "| tokenizer:", "tokenizer" in sig)

print("\n=== 3. Stage-0 data path (stream 120 Tulu convos) ===")
tok = AutoTokenizer.from_pretrained(ROOT_TOK)
if tok.chat_template is None:
    print("base has no chat_template; using", ROOT_TOK + "-Instruct")
    tok = AutoTokenizer.from_pretrained(ROOT_TOK + "-Instruct")
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train", streaming=True, token=HF_TOKEN)
convos = []
for ex in ds:
    convos.append(ex["messages"])
    if len(convos) >= 120: break
print("streamed convos:", len(convos))
DISC = ["consult a", "talk to a", "speak with a", "see a doctor", "seek medical", "medical professional",
        "healthcare professional", "mental health professional", "seek professional", "professional help"]
def asst_text(m): return " ".join(t.get("content", "") for t in m if t.get("role") == "assistant")
asst = [asst_text(m) for m in convos]
beh = [i for i, a in enumerate(asst) if any(p in a.lower() for p in DISC)]
print("behavioral (disclaimer) cluster size in 120:", len(beh))
# render full conversations (the SFT training text) — the chat-template path
texts = [tok.apply_chat_template(m, tokenize=False) for i, m in enumerate(convos) if i not in set(beh[:5])]
dset = Dataset.from_dict({"text": texts})
print("rendered dataset rows:", len(dset), "| sample chars:", len(dset[0]["text"]))

print("\n=== 4. End-to-end TRL wiring (tiny model, 1 step, CPU) ===")
try:
    from transformers import AutoModelForCausalLM
    tiny = "sshleifer/tiny-gpt2"
    tmodel = AutoModelForCausalLM.from_pretrained(tiny)
    ttok = AutoTokenizer.from_pretrained(tiny)
    if ttok.pad_token is None: ttok.pad_token = ttok.eos_token
    kw2 = dict(kw); kw2["output_dir"] = "/tmp/sft_tiny"; kw2["max_steps"] = 1
    kw2["per_device_train_batch_size"] = 1; kw2["gradient_accumulation_steps"] = 1
    cfg2 = SFTConfig(**{k: v for k, v in kw2.items()})
    tkw = {"processing_class": ttok} if "processing_class" in sig else {"tokenizer": ttok}
    small = Dataset.from_dict({"text": ["Hello world. This is a test.", "Another short example here."]})
    tr = SFTTrainer(model=tmodel, args=cfg2, train_dataset=small, **tkw)
    tr.train()
    print("TRL end-to-end train(1 step) OK")
except Exception as e:
    print("TRL end-to-end FAILED:", repr(e)[:300])

print("\nALL CAUSAL-NOTEBOOK NON-GPU CHECKS DONE")
