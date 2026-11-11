"""GPU runner for the pivot study (out/paper/PIVOT_PROTOCOL_2026-11-10.md).

    python pivot_runner.py atlas  --out results
    python pivot_runner.py train  --out results --arm baseline --seed 0
    python pivot_runner.py all    --out results
    python pivot_runner.py smoke  --out /tmp/x      # tiny random model, CPU
    python pivot_runner.py dryrun --out /tmp/y      # whole schedule, tiny models, CPU
    python pivot_runner.py bench  --out results     # GPU timing and projected hours

Every finished unit is written atomically, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE / "prompts_revision_v1.json"
PREF_DATASET = "allenai/olmo-2-0425-1b-preference-mix"
PREF_REVISION = "c6454f3d364622718a8ecd3d307e524febbba569"
PREF_SHARDS = [f"data/train-{i:05d}-of-00005.parquet" for i in range(5)]
SFT_1B = "allenai/OLMo-2-0425-1B-SFT"
SFT_7B = "allenai/OLMo-2-1124-7B-SFT"

# tag -> (model id, format, tokenizer used for the chat template)
ATLAS = {
    "1B_base_qa": ("allenai/OLMo-2-0425-1B", "qa"),
    "1B_base_chat": ("allenai/OLMo-2-0425-1B", "chat"),
    "1B_sft_chat": (SFT_1B, "chat"),
    "1B_sft_qa": (SFT_1B, "qa"),
    "1B_dpo_chat": ("allenai/OLMo-2-0425-1B-DPO", "chat"),
    "1B_rlvr_chat": ("allenai/OLMo-2-0425-1B-RLVR1", "chat"),
    "1B_instruct_chat": ("allenai/OLMo-2-0425-1B-Instruct", "chat"),
    "7B_base_qa": ("allenai/OLMo-2-1124-7B", "qa"),
    "7B_base_chat": ("allenai/OLMo-2-1124-7B", "chat"),
    "7B_sft_chat": (SFT_7B, "chat"),
    "7B_sft_qa": (SFT_7B, "qa"),
    "7B_dpo_chat": ("allenai/OLMo-2-1124-7B-DPO", "chat"),
    "7B_instruct_chat": ("allenai/OLMo-2-1124-7B-Instruct", "chat"),
}

ARMS = ("baseline", "random_drop", "drop_chosen_only", "flip_chosen_only", "length_matched")
TRAIN_SEEDS = (0, 1, 2)
GEN_SEEDS = (0, 1, 2)
N_PAIRS = int(os.environ.get("PIVOT_N_PAIRS", 60_000))
DRY = False  # set by the dryrun command: tiny random models, synthetic pairs
MICRO_BATCH = 4
ACCUM = 8
LR = 1e-6
BETA = 5.0
WARMUP = 0.1
MAX_SEQ = 1024
MAX_NEW = 1024

# Copied from experiments/detector_spec.py so the runner is self-contained.
STRUCTURE = ["here are some", "here are a few", "consider the following",
             "steps you can take", "a few things you can", "first,", "secondly,"]
DISCLAIMER = ["consult a", "talk to a", "speak with a", "see a doctor", "seek medical",
              "i am not a doctor", "i'm not a doctor", "i'm not a medical",
              "medical professional", "healthcare professional",
              "mental health professional", "seek professional", "professional help"]
TARGET = re.compile("|".join(re.escape(p) for p in STRUCTURE + DISCLAIMER))


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def atomic_jsonl(path: Path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_prompts(limit=None):
    corpus = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    out = []
    for fam in corpus["families"]:
        if fam["split"] != "evaluation":
            continue
        for emo in (False, True):
            for req in ("information", "advice"):
                parts = [fam["context"]] + ([fam["emotional_clause"]] if emo else []) + [fam["requests"][req]]
                out.append({"family": fam["scenario_family_id"], "domain": fam["domain"],
                            "emotional": emo, "request": req, "text": " ".join(parts)})
    if len(out) != 160:
        raise ValueError(f"expected 160 prompts, got {len(out)}")
    return out[:limit] if limit else out


def serialize(tok, text, fmt):
    if fmt == "qa":
        return f"Question: {text}\nAnswer:"
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, fmt, prompts, tag, out_dir, max_new=None, batch=32, seeds=None):
    max_new, seeds = max_new or MAX_NEW, seeds or GEN_SEEDS
    path = out_dir / f"gen_{tag}.jsonl"
    if path.exists():
        print("skip", path.name, flush=True)
        return
    model.eval()
    tok.padding_side = "left"
    rows = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok([serialize(tok, p["text"], fmt) for p in chunk], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(model.device)
            enc.pop("token_type_ids", None)
            ids = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9,
                                 max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            new = ids[:, enc["input_ids"].shape[1]:].cpu()
            for p, seq in zip(chunk, new):
                seq = seq.tolist()
                eos = tok.eos_token_id
                stop = "eos" if eos in seq else "cap"
                if eos in seq:
                    seq = seq[:seq.index(eos)]
                rows.append({**p, "tag": tag, "seed": seed, "format": fmt, "stop": stop,
                             "n_tokens": len(seq),
                             "response": tok.decode(seq, skip_special_tokens=True)})
    atomic_jsonl(path, rows)
    print(f"done {tag} n={len(rows)} {time.time() - t0:.0f}s", flush=True)


def tiny_model(model_id):
    """Small random OLMo-2, deterministic per model id (dry runs only)."""
    from transformers import AutoModelForCausalLM, Olmo2Config
    tok = load_tok(SFT_1B)
    torch.manual_seed(sum(map(ord, model_id)))
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return AutoModelForCausalLM.from_config(cfg)


def load_model(model_id, dtype):
    if DRY:
        return tiny_model(model_id).to(dtype if dtype == torch.float32 else torch.float32)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device())


def load_tok(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SFT_1B if DRY else model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def run_atlas(out_dir):
    prompts = load_prompts()
    for tag, (model_id, fmt) in ATLAS.items():
        if (out_dir / f"gen_{tag}.jsonl").exists():
            continue
        tok = load_tok(SFT_7B if tag.startswith("7B") else SFT_1B)  # base lacks a chat template
        model = load_model(model_id, torch.bfloat16)
        generate(model, tok, fmt, prompts, tag, out_dir)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def synthetic_pairs(n=3000):
    """Pairs with a known marker pattern, for dry runs."""
    rng = random.Random(7)
    out = []
    for i in range(n):
        u = {"role": "user", "content": f"synthetic question {i}"}
        words = "plain words " * rng.randint(2, 12)
        lead = rng.choice(["Here are some ideas. ", "Please consult a doctor. ", ""])
        out.append(([u, {"role": "assistant", "content": lead + words}],
                    [u, {"role": "assistant", "content": rng.choice(["", "First, "]) + words}]))
    return out


def load_pairs():
    if DRY:
        return synthetic_pairs()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    pairs = []
    for shard in PREF_SHARDS:
        fp = hf_hub_download(PREF_DATASET, shard, repo_type="dataset", revision=PREF_REVISION)
        table = pq.read_table(fp, columns=["chosen", "rejected"]).to_pylist()
        for row in table:
            c, r = row["chosen"], row["rejected"]
            if not c or not r or c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant":
                continue
            if c[:-1] != r[:-1] or not c[-1]["content"] or not r[-1]["content"]:
                continue
            pairs.append((c, r))
    return pairs


def flags(pair):
    return bool(TARGET.search(pair[0][-1]["content"].lower())), bool(TARGET.search(pair[1][-1]["content"].lower()))


def build_arm(pairs, arm, seed, n=None):
    """Return (pairs for training, counts). The sample S depends only on seed."""
    n = n or N_PAIRS
    rng = random.Random(1000 + seed)
    if arm == "length_matched":
        pool = [p for p in pairs if abs(len(p[0][-1]["content"].split()) - len(p[1][-1]["content"].split())) <= 20]
        if len(pool) < n:
            raise ValueError(f"length pool too small: {len(pool)}")
        return rng.sample(pool, n), {"pool": len(pool)}
    sample = rng.sample(pairs, n)
    only = [i for i, p in enumerate(sample) if flags(p) == (True, False)]
    counts = {"n_chosen_only": len(only), "n_rejected_only": sum(flags(p) == (False, True) for p in sample)}
    if arm == "baseline":
        return sample, counts
    if arm == "random_drop":
        drop = set(random.Random(2000 + seed).sample(range(n), len(only)))
        return [p for i, p in enumerate(sample) if i not in drop], counts
    if arm == "drop_chosen_only":
        drop = set(only)
        return [p for i, p in enumerate(sample) if i not in drop], counts
    if arm == "flip_chosen_only":
        flip = set(only)
        return [(p[1], p[0]) if i in flip else p for i, p in enumerate(sample)], counts
    raise ValueError(arm)


def encode(tok, msgs):
    def ids(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    prompt = ids(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True))
    full = ids(tok.apply_chat_template(msgs, tokenize=False))[:MAX_SEQ]
    return full, min(len(prompt), len(full) - 1)


def avg_logp(model, batch):
    """Mean log-probability per response token for each sequence."""
    logits = model(input_ids=batch["ids"], attention_mask=batch["mask"]).logits
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(2, batch["ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
    m = batch["label_mask"][:, 1:].to(lp.dtype)
    return (lp * m).sum(1) / m.sum(1).clamp(min=1)


def collate(items, pad, dev):
    n = max(len(i[0]) for i in items)
    ids = torch.full((len(items), n), pad)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    label = torch.zeros((len(items), n), dtype=torch.bool)
    for k, (seq, plen) in enumerate(items):
        ids[k, :len(seq)] = torch.tensor(seq)
        mask[k, :len(seq)] = 1
        label[k, plen:len(seq)] = True
    return {"ids": ids.to(dev), "mask": mask.to(dev), "label_mask": label.to(dev)}


def make_optimizer(model):
    """fp32 weights with 8-bit AdamW states: 1.5B params fit on 24 GB beside the bf16 reference."""
    if model.device.type == "cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.0)
        except ImportError:
            print("bitsandbytes missing, using torch AdamW (needs about 24 GB for 1B)", flush=True)
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)


def dpo_train(model, ref, tok, data, seed, log_every=50):
    dev = model.device
    amp = dev.type == "cuda"
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    # group by length inside large chunks to limit padding
    chunks = [sorted(data[i:i + 256 * MICRO_BATCH], key=lambda p: len(p[0][-1]["content"]))
              for i in range(0, len(data), 256 * MICRO_BATCH)]
    batches = [c[i:i + MICRO_BATCH] for c in chunks for i in range(0, len(c), MICRO_BATCH)]
    rng.shuffle(batches)
    total = math.ceil(len(batches) / ACCUM)
    warm = max(1, int(WARMUP * total))
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    model.train()
    model.gradient_checkpointing_enable()
    log, step, acc_loss, acc_right, t0 = [], 0, 0.0, 0.0, time.time()
    for b_idx, chunk in enumerate(batches):
        c = collate([encode(tok, p[0]) for p in chunk], tok.pad_token_id, dev)
        r = collate([encode(tok, p[1]) for p in chunk], tok.pad_token_id, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            with torch.no_grad():
                rc, rr = avg_logp(ref, c), avg_logp(ref, r)
            pc, pr = avg_logp(model, c), avg_logp(model, r)
            margin = (pc - pr) - (rc - rr)
            loss = -F.logsigmoid(BETA * margin).mean()
        (loss / ACCUM).backward()
        acc_loss += loss.item() / ACCUM
        acc_right += (pc > pr).float().mean().item() / ACCUM
        if (b_idx + 1) % ACCUM == 0 or b_idx + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total:
                log.append({"step": step, "loss": acc_loss, "acc": acc_right, "secs": time.time() - t0})
                print(json.dumps(log[-1]), flush=True)
            acc_loss, acc_right = 0.0, 0.0
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return log


def run_train(out_dir, arm, seed, pairs=None):
    tag = f"cf_{arm}_s{seed}"
    if (out_dir / f"gen_{tag}.jsonl").exists():
        print("skip", tag, flush=True)
        return
    torch.manual_seed(seed)
    pairs = pairs if pairs is not None else load_pairs()
    data, counts = build_arm(pairs, arm, seed)
    tok = load_tok(SFT_1B)
    tok.padding_side = "right"
    ref = load_model(SFT_1B, torch.bfloat16).eval().requires_grad_(False)
    model = load_model(SFT_1B, torch.float32)
    t0 = time.time()
    log = dpo_train(model, ref, tok, data, seed)
    (out_dir / f"train_{tag}.json").write_text(json.dumps(
        {"arm": arm, "seed": seed, "n_train": len(data), "counts": counts, "log": log,
         "n_total_pairs": len(pairs), "train_secs": time.time() - t0}), encoding="utf-8")
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    model = model.bfloat16()
    generate(model, tok, "chat", load_prompts(), tag, out_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_smoke(out_dir):
    """Exercises data arms, the DPO step and generation with a tiny random model."""
    from transformers import AutoModelForCausalLM, Olmo2Config
    tok = load_tok(SFT_1B)
    tok.padding_side = "right"
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model = AutoModelForCausalLM.from_config(cfg)
    ref = AutoModelForCausalLM.from_config(cfg).eval().requires_grad_(False)
    ref.load_state_dict(model.state_dict())
    before = next(model.parameters()).clone()

    def pair(i, marker_in_chosen):
        u = {"role": "user", "content": f"question {i}"}
        ch = "Here are some ideas. " * (1 + i % 3) if marker_in_chosen else "Plain reply. " * (1 + i % 3)
        return ([u, {"role": "assistant", "content": ch}],
                [u, {"role": "assistant", "content": "Another plain reply."}])

    pairs = [pair(i, i % 2 == 0) for i in range(400)]
    base, counts = build_arm(pairs, "baseline", 0, n=100)
    assert counts["n_chosen_only"] > 0
    drop, _ = build_arm(pairs, "drop_chosen_only", 0, n=100)
    rand, _ = build_arm(pairs, "random_drop", 0, n=100)
    flip, _ = build_arm(pairs, "flip_chosen_only", 0, n=100)
    assert len(drop) == len(rand) == 100 - counts["n_chosen_only"]
    assert not any(flags(p) == (True, False) for p in drop)
    assert sum(flags(p) == (True, False) for p in flip) == 0
    assert sum(flags(p) == (False, True) for p in flip) == counts["n_chosen_only"] + counts["n_rejected_only"]
    full, plen = encode(tok, pairs[0][0])
    assert 0 < plen < len(full)
    global LR
    saved, LR = LR, 1e-3
    log = dpo_train(model, ref, tok, base[:32], 0, log_every=1)
    LR = saved
    assert log and not torch.equal(before, next(model.parameters()))
    generate(model, tok, "chat", load_prompts(4), "smoke", out_dir, max_new=8, batch=2, seeds=(0,))
    rows = [json.loads(x) for x in (out_dir / "gen_smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and all("response" in r for r in rows)
    print("SMOKE OK", counts, log[-1], flush=True)


def run_all(out):
    run_atlas(out)
    pairs = load_pairs()
    print("pairs", len(pairs), flush=True)
    for seed in TRAIN_SEEDS:
        for arm in ARMS:
            run_train(out, arm, seed, pairs)
    (out / "ALL_DONE").write_text("ok")


def run_dry(out):
    """The full `all` schedule with tiny random models and synthetic pairs, on CPU."""
    global DRY, N_PAIRS, MAX_NEW, GEN_SEEDS
    DRY, N_PAIRS, MAX_NEW, GEN_SEEDS = True, 200, 6, (0,)
    run_all(out)
    names = sorted(x.name for x in out.iterdir())
    expected = [f"gen_{t}.jsonl" for t in ATLAS] + [f"gen_cf_{a}_s{s}.jsonl" for a in ARMS for s in TRAIN_SEEDS]
    missing = [e for e in expected if e not in names]
    assert not missing, missing
    print("DRY RUN OK", len(names), "files", flush=True)


def run_bench(out, n_micro=16):
    """Time real DPO steps and generation on this GPU and project the full run."""
    pairs = load_pairs()
    data, counts = build_arm(pairs, "baseline", 0)
    tok = load_tok(SFT_1B)
    tok.padding_side = "right"
    ref = load_model(SFT_1B, torch.bfloat16).eval().requires_grad_(False)
    model = load_model(SFT_1B, torch.float32)
    torch.cuda.reset_peak_memory_stats()
    global ACCUM
    saved, ACCUM = ACCUM, n_micro
    t0 = time.time()
    dpo_train(model, ref, tok, data[:n_micro * MICRO_BATCH], 0, log_every=1)
    per_micro = (time.time() - t0) / n_micro
    ACCUM = saved
    peak = torch.cuda.max_memory_allocated() / 2**30
    del ref
    model = model.bfloat16()
    prompts = load_prompts(32)
    t0 = time.time()
    generate(model, tok, "chat", prompts, "bench", out, seeds=(0,))
    per_gen_batch = time.time() - t0
    (out / "gen_bench.jsonl").unlink()
    gen_batches_1b = 3 * 5  # 160 prompts x 3 seeds / 32
    train_h = len(ARMS) * len(TRAIN_SEEDS) * (N_PAIRS / MICRO_BATCH) * per_micro / 3600
    gen_1b_h = (7 + 15) * gen_batches_1b * per_gen_batch / 3600
    gen_7b_h = 6 * gen_batches_1b * per_gen_batch * 4 / 3600  # 7B assumed 4x slower
    report = {"sec_per_micro_batch": per_micro, "peak_gb": peak, "sec_per_gen_batch_1b": per_gen_batch,
              "n_pairs": N_PAIRS, "counts": counts, "proj_train_h": train_h, "proj_gen_1b_h": gen_1b_h,
              "proj_gen_7b_h": gen_7b_h, "proj_total_h": train_h + gen_1b_h + gen_7b_h}
    (out / "bench.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["atlas", "train", "all", "smoke", "dryrun", "bench"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"smoke": run_smoke, "atlas": run_atlas, "all": run_all, "dryrun": run_dry, "bench": run_bench,
     "train": lambda o: run_train(o, a.arm, a.seed)}[a.cmd](out)


if __name__ == "__main__":
    main()
