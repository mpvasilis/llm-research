"""Local, file-based history of trace runs + exporters.

Every trace is saved as one JSON record under out/history/. Records survive
restarts and can be listed, reloaded, and exported (JSON / Markdown / CSV) for
further investigation. No database — just files you can also open by hand.
"""
import csv
import datetime
import io
import json
import re
from pathlib import Path
from config import ROOT

HIST = ROOT / "out" / "history"


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "untitled").lower()).strip("-")
    return s[:n] or "untitled"


def save(trace: dict) -> dict:
    """Persist a trace as a history record; returns the record (with id, ts)."""
    HIST.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    rid = now.strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ms precision -> unique
    p = trace.get("pretraining", {})
    rec = {
        "id": rid,
        "ts": now.isoformat(timespec="seconds"),
        "model": trace.get("model", ""),
        "question": trace.get("question", ""),
        "slug": _slug(trace.get("question", "")),
        "n_spans": p.get("n_matches", 0),
        "longest": p.get("longest_span_words", 0),
        "trace": trace,
    }
    (HIST / f"{rid}.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def _summary(rec: dict) -> dict:
    return {k: rec.get(k) for k in
            ("id", "ts", "model", "question", "slug", "n_spans", "longest")}


def list_records() -> list[dict]:
    """Newest-first summaries of all saved traces."""
    if not HIST.exists():
        return []
    out = []
    for f in HIST.glob("*.json"):
        try:
            out.append(_summary(json.loads(f.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda r: r["id"], reverse=True)


def load(rid: str) -> dict | None:
    f = HIST / f"{_safe_id(rid)}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def delete(rid: str) -> bool:
    f = HIST / f"{_safe_id(rid)}.json"
    if f.exists():
        f.unlink()
        return True
    return False


def _safe_id(rid: str) -> str:
    return re.sub(r"[^0-9-]", "", rid or "")


# --- exporters -----------------------------------------------------------
def export_json(rec: dict) -> str:
    return json.dumps(rec, indent=2, ensure_ascii=False)


def export_md(rec: dict) -> str:
    from .pipeline import render_report  # lazy to avoid import cycle
    return render_report(rec["trace"])


def export_csv(rec: dict) -> str:
    """Pretraining matched spans as a flat CSV for analysis."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "model", "ts", "span_words", "corpus_count", "phrase"])
    t = rec["trace"]
    for m in t.get("pretraining", {}).get("matched_spans", []):
        w.writerow([rec["question"], rec["model"], rec["ts"],
                    m.get("words"), m.get("count"), m.get("phrase")])
    return buf.getvalue()


def export_all() -> str:
    """Every record bundled into one JSON array."""
    recs = []
    for f in sorted(HIST.glob("*.json")):
        try:
            recs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return json.dumps(recs, indent=2, ensure_ascii=False)
