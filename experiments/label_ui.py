"""Local labeling UI for the stratified detector-validation sheet.

Serves out/results/validation_sheet_v2.csv at http://localhost:8777 as a
clean, keyboard-friendly page: one sentence per card, category checkboxes,
autosave back into the CSV on every change. Deliberately HIDES the detector's
prediction and the prompt condition to avoid anchoring the annotator.

Usage:
    python -m experiments.label_ui                # annotator 1 -> human_label_1
    python -m experiments.label_ui --annotator 2  # annotator 2 -> human_label_2

Then score with:  python -m experiments.score_validation_v2
"""
import argparse
import csv
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import ROOT

SHEET = ROOT / "out" / "results" / "validation_sheet_v2.csv"
CATS = ["empathy_opener", "validation", "disclaimer", "crisis_referral", "structure"]

GUIDE = {
    "empathy_opener": "expresses sympathy or acknowledges the user's feelings; generic politeness and factual descriptions do not count",
    "validation": "normalizes or legitimizes a feeling; advice without affirmation does not count",
    "disclaimer": "recommends consulting a qualified professional; a factual mention of clinicians or treatments does not count",
    "crisis_referral": "refers to a crisis-specific resource such as 988, a hotline, or emergency services",
    "structure": "uses an explicit enumeration or list opener; a bold fragment without an organizing function does not count",
}

DISPLAY = {
    "empathy_opener": "empathy_expression",
    "validation": "validation",
    "disclaimer": "professional_referral",
    "crisis_referral": "crisis_referral",
    "structure": "structuring_language",
}


def read_rows():
    with open(SHEET, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_rows(rows, fieldnames):
    tmp = SHEET.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(SHEET)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Detector validation labeling</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#111;color:#ddd}
 .guide{background:#1a1a2a;border:1px solid #334;border-radius:8px;padding:10px 14px;font-size:13px;margin-bottom:16px}
 .guide b{color:#9cf}
 .card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:12px 16px;margin:10px 0}
 .card.done{border-color:#2a5}
 .sent{white-space:pre-wrap;margin-bottom:8px;font-size:15px;line-height:1.45}
 label{margin-right:14px;font-size:13px;cursor:pointer;user-select:none}
 input[type=checkbox]{accent-color:#4a8;margin-right:4px}
 .none{color:#e90}
 #prog{position:sticky;top:0;background:#111;padding:8px 0;font-weight:600;border-bottom:1px solid #333;z-index:2}
 .idx{color:#777;font-size:12px}
</style></head><body>
<div id="prog"></div>
<div class="guide"><b>Mark every category the sentence EXPRESSES (or “none”).</b><br>__GUIDE__<br>
Autosaves on every click. Annotator: <b>__ANN__</b> → column <code>human_label___ANN__</code>.</div>
<div id="cards"></div>
<script>
const CATS=__CATS__; const DISPLAY=__DISPLAY__;
let ROWS=__ROWS__;
function prog(){const d=ROWS.filter(r=>r.label.trim()!=="").length;
 document.getElementById('prog').textContent=`${d} / ${ROWS.length} labeled`;}
function save(i){fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({i:i,label:ROWS[i].label})}).then(prog);}
function toggle(i,cat,card){let s=new Set(ROWS[i].label.split(',').map(x=>x.trim()).filter(x=>x&&x!=='none'));
 if(cat==='none'){s.clear();ROWS[i].label='none';}
 else{s.has(cat)?s.delete(cat):s.add(cat);ROWS[i].label=[...s].sort().join(',')||'';}
 render(i,card);save(i);}
function render(i,card){const r=ROWS[i];const cur=new Set(r.label.split(',').map(x=>x.trim()));
 card.className='card'+(r.label.trim()!==''?' done':'');
 card.querySelectorAll('input').forEach(cb=>{cb.checked=cur.has(cb.dataset.cat);});}
const wrap=document.getElementById('cards');
ROWS.forEach((r,i)=>{const card=document.createElement('div');card.className='card';
 const s=document.createElement('div');s.className='sent';
 s.innerHTML=`<span class="idx">#${i+1}</span>  `+r.sentence.replace(/&/g,'&amp;').replace(/</g,'&lt;');
 card.appendChild(s);
 CATS.concat(['none']).forEach(cat=>{const l=document.createElement('label');
  if(cat==='none')l.className='none';
  const cb=document.createElement('input');cb.type='checkbox';cb.dataset.cat=cat;
  cb.onclick=()=>toggle(i,cat,card);l.appendChild(cb);l.appendChild(document.createTextNode(DISPLAY[cat]||cat));
  card.appendChild(l);});
 wrap.appendChild(card);render(i,card);});
prog();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", type=int, default=1, choices=(1, 2))
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    col = f"human_label_{args.annotator}"

    rows, fieldnames = read_rows()
    if col not in fieldnames:
        raise SystemExit(f"Column {col} not in {SHEET}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            payload = [{"sentence": r["sentence"], "label": r.get(col, "") or ""} for r in rows]
            guide = "<br>".join(f"<b>{DISPLAY[c]}</b>: {t}" for c, t in GUIDE.items())
            html = (PAGE.replace("__CATS__", json.dumps(CATS))
                        .replace("__DISPLAY__", json.dumps(DISPLAY))
                        .replace("__ROWS__", json.dumps(payload))
                        .replace("__GUIDE__", guide)
                        .replace("__ANN__", str(args.annotator)))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
            rows[d["i"]][col] = d["label"]
            write_rows(rows, fieldnames)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    # Keep one abandoned browser connection from blocking autosave requests.
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    url = f"http://localhost:{args.port}"
    print(f"Labeling {SHEET.name} as annotator {args.annotator} -> column {col}")
    print(f"Open {url}  (Ctrl+C to stop; every click autosaves)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    srv.serve_forever()


if __name__ == "__main__":
    main()
