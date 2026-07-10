"""Orchestrate the provenance trace and write the report."""
import json
from pathlib import Path
from config import ROOT, PRETRAIN_INDEXES, PROXY_MODELS, load_token
from .ngrams import attribute
from .infinigram import count
from .hf_datasets import search_instruction, keywords


def _rarest_term(index: str, question: str, floor: int = 1000) -> str:
    """Most distinctive topic word = rarest corpus term, but ABOVE a frequency
    floor so misspellings/junk (≈0 corpus hits, e.g. 'girlfirned') aren't picked.
    If every term is below the floor, fall back to the most common (most 'real')."""
    terms = keywords(question)
    if not terms:
        return ""
    freqs = {w: count(index, w).get("count", 0) for w in terms}
    real = {w: c for w, c in freqs.items() if c >= floor}
    if real:
        return min(real, key=real.get)          # rarest among real words
    return max(freqs, key=freqs.get)             # all junk -> least-junk


def run(model: str, question: str, answer: str, max_calls: int = 80,
        do_instruction: bool = True, save_history: bool = True) -> dict:
    token = load_token()
    index = PRETRAIN_INDEXES[model]

    pretrain = attribute(index, answer, max_calls=max_calls)
    if do_instruction:
        query = _rarest_term(index, question)
        instruct = search_instruction(model, question, token, query=query)
    else:
        instruct = {"query_terms": [], "query_used": "(skipped)", "datasets": []}

    trace = {
        "model": model,
        "is_proxy": model in PROXY_MODELS,
        "question": question,
        "answer": answer,
        "pretraining": pretrain,
        "instruction_tuning": instruct,
    }
    _write(trace)  # latest run, for convenience
    if save_history:
        from .store import save  # lazy to avoid import cycle
        rec = save(trace)
        trace["record_id"] = rec["id"]
    return trace


def _write(trace: dict) -> None:
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "report.md").write_text(render_report(trace), encoding="utf-8")


def _src_tag(src) -> str:
    """Readable provenance tag from infini-gram doc metadata."""
    if not isinstance(src, dict):
        return f"[{str(src)[:80]}]" if src else ""
    meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
    tag = src.get("url") or meta.get("url") or src.get("path") or ""
    return f"`[{str(tag)[:90]}]`" if tag else ""


def render_report(t: dict) -> str:
    L = []
    L.append(f"# Answer provenance trace — `{t['model']}`\n")
    if t["model"] in PROXY_MODELS:
        L.append(
            "> ⚠️ **Proxy trace.** This model's training data is private. The "
            "corpus below (Dolma, open web) is a *plausible-source proxy* — it "
            "shows the answer's phrasings exist on the public web, **not** that "
            "this model was trained on them.\n")
    L.append(f"**Question:** {t['question']}\n")
    L.append(f"**Answer traced:**\n\n> {t['answer'][:500]}\n")

    p = t["pretraining"]
    L.append("\n## 1. Pretraining provenance (corpus n-gram attribution)\n")
    L.append(f"Corpus index: `{p['index']}` · matched spans: **{p['n_matches']}** · "
             f"longest verbatim span: **{p['longest_span_words']} words**\n")
    if not p["matched_spans"]:
        L.append("_No multi-word spans matched — answer phrasing is not verbatim "
                 "in this corpus (paraphrase / novel composition)._\n")
    for m in p["matched_spans"][:12]:
        L.append(f"\n- **{m['words']} words**, {m['count']:,} corpus hits — "
                 f"\"{m['phrase']}\"")
        for ps in m.get("passages", []):
            L.append(f"    - corpus: …{ps['text'][:200]}… {_src_tag(ps.get('source'))}")

    L.append("\n## 2. Instruction-tuning provenance (SFT dataset search)\n")
    ins = t["instruction_tuning"]
    L.append(f"Topic keywords: `{', '.join(ins['query_terms'])}` · "
             f"query: `{ins['query_used']}`\n")
    for d in ins["datasets"]:
        if d.get("error"):
            err = d["error"]
            if "loading" in err.lower() or "rebuilt" in err.lower() or "corrupt" in err.lower():
                note = "HF search index still building/rebuilding server-side — re-run later to populate."
            else:
                note = err
            L.append(f"\n### `{d['dataset']}` — unavailable: {note}")
            continue
        note = f" ({d['scan_note']})" if d.get("scan_note") else ""
        title = d["dataset"]
        if d.get("viewer_url"):
            title = f"[{d['dataset']}]({d['viewer_url']})"
        L.append(f"\n### {title} — {d.get('total_matches', 0):,} matching examples{note}")
        for h in d["hits"]:
            if h.get("turns"):
                L.append("\n  - **example:**")
                for tr in h["turns"]:
                    L.append(f"    - *{tr['role']}:* {tr['content'][:800]}")
            else:
                L.append(f"  - {h.get('full', h['snippet'])[:800]}")

    L.append("\n---\n*Pretraining = where the wording comes from. "
             "Instruction tuning = where the advice-giving behavior comes from.*\n")
    return "\n".join(L)
