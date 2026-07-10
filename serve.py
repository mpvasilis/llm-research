"""Local web UI for the provenance tracer. Zero deps (stdlib http.server).

    python serve.py          # then open http://localhost:8000

Paste the question + the LLM's answer, pick the open-data model, hit Trace.
The pretraining layer (infini-gram) is fast and reliable; the instruction layer
(HF datasets-server) is optional and can be slow, so it's off by default.
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from config import PRETRAIN_INDEXES, DEFAULT_MODEL
from trace.pipeline import run
from trace import store

PORT = 8000

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Answer Provenance Tracer</title>
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 system-ui,sans-serif;background:#0d1117;color:#e6edf3}
  header{padding:18px 28px;border-bottom:1px solid #21262d}
  h1{margin:0;font-size:18px}.sub{color:#8b949e;font-size:13px;margin-top:4px}
  main{display:grid;grid-template-columns:380px 1fr;gap:0;height:calc(100vh - 64px)}
  .panel{padding:22px 28px;overflow:auto}
  .panel.left{border-right:1px solid #21262d}
  label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.5px;
    color:#8b949e;margin:16px 0 6px}
  input,textarea,select{width:100%;background:#161b22;border:1px solid #30363d;
    color:#e6edf3;border-radius:8px;padding:10px 12px;font:inherit}
  textarea{min-height:170px;resize:vertical}
  .row{display:flex;gap:10px;align-items:center;margin-top:14px}
  .row label{margin:0;text-transform:none;letter-spacing:0;color:#e6edf3;font-size:14px}
  button{margin-top:20px;width:100%;background:#238636;border:0;color:#fff;
    padding:12px;border-radius:8px;font:600 15px/1 inherit;cursor:pointer}
  button:disabled{background:#30363d;cursor:wait}
  .span{border:1px solid #30363d;border-radius:8px;padding:12px 14px;margin:10px 0;
    background:#161b22}
  .span .ph{color:#7ee787;font-weight:600}
  .meta{color:#8b949e;font-size:12px;margin-top:4px}
  .pass{border-left:2px solid #30363d;margin:8px 0 0 4px;padding:4px 0 4px 12px;
    color:#c9d1d9;font-size:13px}
  .tag{color:#58a6ff;font-size:11px}
  h2{font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;
    border-bottom:1px solid #21262d;padding-bottom:8px;margin-top:26px}
  .hit{border:1px solid #30363d;border-radius:8px;padding:10px 12px;margin:8px 0;
    background:#161b22;font-size:13px;color:#c9d1d9}
  .dslink{color:#58a6ff;text-decoration:none}.dslink:hover{text-decoration:underline}
  .hitx{border:1px solid #30363d;border-radius:8px;margin:8px 0;background:#161b22}
  .hitx>summary{padding:10px 12px;cursor:pointer;font-size:13px;color:#c9d1d9;
    list-style:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hitx>summary::-webkit-details-marker{display:none}
  .hitx>summary::before{content:"▸ ";color:#8b949e}
  .hitx[open]>summary::before{content:"▾ "}
  .hitx[open]>summary{white-space:normal;border-bottom:1px solid #21262d}
  .turn{padding:8px 12px;font-size:13px;color:#c9d1d9;border-bottom:1px solid #21262d;
    white-space:pre-wrap;word-break:break-word}
  .turn:last-child{border-bottom:0}
  .role{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.5px;
    color:#7ee787;background:#1a2a1a;border-radius:4px;padding:1px 6px;margin-right:8px}
  .empty{color:#8b949e;font-style:italic}
  .err{color:#f85149}
  .loading{color:#d29922}
  .bar{display:flex;gap:18px;color:#8b949e;font-size:13px;margin-bottom:8px}
  .bar b{color:#e6edf3}
  .histhead{display:flex;justify-content:space-between;align-items:center;
    margin-top:24px;border-top:1px solid #21262d;padding-top:14px}
  .xall{color:#58a6ff;font-size:12px;text-decoration:none}.xall:hover{text-decoration:underline}
  .hitem{border:1px solid #30363d;border-radius:8px;padding:8px 10px;margin:8px 0;
    background:#161b22;cursor:pointer}.hitem:hover{border-color:#58a6ff}
  .hitem .hq{font-size:13px;color:#e6edf3;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}
  .hitem .hm{font-size:11px;color:#8b949e;margin-top:3px;display:flex;
    justify-content:space-between;align-items:center;gap:6px}
  .hitem a{color:#58a6ff;text-decoration:none;font-size:11px}
  .hitem .del{color:#f85149;cursor:pointer}
  .xbar{display:flex;gap:10px;margin:2px 0 14px}
  .xbar a{color:#58a6ff;font-size:12px;text-decoration:none;border:1px solid #30363d;
    border-radius:6px;padding:4px 10px}.xbar a:hover{border-color:#58a6ff}
</style></head><body>
<header><h1>Answer Provenance Tracer</h1>
<div class="sub">Trace an open-data LLM answer back to its pretraining (Dolma via
 infini-gram) and instruction (HF) data.</div></header>
<main>
 <div class="panel left">
  <label>Question you asked</label>
  <input id="q" placeholder="How do I stop snoring while sleeping with my girlfriend?">
  <label>The model's answer</label>
  <textarea id="a" placeholder="Paste the reply, or tick Generate below..."></textarea>
  <label>Open-data model</label>
  <select id="m">__MODELS__</select>
  <div class="row"><input type="checkbox" id="gen"><label for="gen">
    Generate answer locally (slow; downloads model on first run)</label></div>
  <div class="row"><input type="checkbox" id="ins"><label for="ins">
    Also search instruction data (slower, flaky)</label></div>
  <button id="go" onclick="trace()">Trace provenance</button>
  <div class="histhead">
    <label style="margin:0">History</label>
    <a class="xall" href="/api/export-all" download>export all ⤓</a>
  </div>
  <div id="hist"><div class="empty" style="font-size:13px">No saved traces yet.</div></div>
 </div>
 <div class="panel" id="out"><div class="empty">Results appear here.</div></div>
</main>
<script>
async function trace(){
  const btn=document.getElementById('go'), out=document.getElementById('out');
  const q=document.getElementById('q').value, a=document.getElementById('a').value;
  const gen=document.getElementById('gen').checked;
  if(!a.trim()&&!gen){out.innerHTML='<div class="err">Paste an answer, or tick Generate.</div>';return}
  btn.disabled=true;btn.textContent=gen?'Generating + tracing…':'Tracing… (~30-60s)';
  out.innerHTML='<div class="loading">'+(gen?'Running the model locally (first run downloads ~2.5GB), then ':'')+
    'querying corpora… this can take a while.</div>';
  try{
    const r=await fetch('/api/trace',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,answer:a,generate:gen,model:document.getElementById('m').value,
        instruction:document.getElementById('ins').checked})});
    const t=await r.json();
    if(t.error){out.innerHTML='<div class="err">'+t.error+'</div>';return}
    showResult(t.record_id, t); loadHistory();
  }catch(e){out.innerHTML='<div class="err">'+e+'</div>'}
  finally{btn.disabled=false;btn.textContent='Trace provenance'}
}
function xbar(id){
  if(!id) return '';
  return '<div class="xbar"><span style="color:#8b949e;font-size:12px;align-self:center">export:</span>'+
    '<a href="/api/export/'+id+'.json" download>JSON</a>'+
    '<a href="/api/export/'+id+'.md" download>Markdown</a>'+
    '<a href="/api/export/'+id+'.csv" download>CSV (spans)</a></div>';
}
function showResult(id, t){document.getElementById('out').innerHTML=xbar(id)+render(t)}
async function loadHistory(){
  const h=document.getElementById('hist');
  try{
    const list=await (await fetch('/api/history')).json();
    if(!list.length){h.innerHTML='<div class="empty" style="font-size:13px">No saved traces yet.</div>';return}
    h.innerHTML=list.map(r=>
      '<div class="hitem" data-id="'+r.id+'">'+
        '<div class="hq">'+esc(r.question||'(no question)')+'</div>'+
        '<div class="hm"><span>'+esc(r.ts)+' · '+esc(r.model)+' · '+r.n_spans+' spans</span>'+
        '<span class="del" data-del="'+r.id+'">delete</span></div>'+
      '</div>').join('');
    h.querySelectorAll('.hitem').forEach(el=>el.addEventListener('click',e=>{
      const d=e.target.closest('[data-del]');
      if(d){e.stopPropagation();delRecord(d.getAttribute('data-del'));}
      else viewRecord(el.getAttribute('data-id'));
    }));
  }catch(e){h.innerHTML='<div class="err">'+e+'</div>'}
}
async function viewRecord(id){
  const out=document.getElementById('out');
  out.innerHTML='<div class="loading">Loading saved trace…</div>';
  const rec=await (await fetch('/api/history/'+id)).json();
  if(rec.error){out.innerHTML='<div class="err">'+rec.error+'</div>';return}
  showResult(rec.id, rec.trace);
}
async function delRecord(id){
  await fetch('/api/history/'+id,{method:'DELETE'}); loadHistory();
}
window.addEventListener('load', loadHistory);
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(t){
  const p=t.pretraining;let h='';
  if(t.is_proxy) h+='<div class="err" style="border:1px solid #5a2d2d;background:#2a1a1a;'+
    'padding:10px 12px;border-radius:8px;margin-bottom:10px">⚠ Proxy trace: '+esc(t.model)+
    ' training data is private. Spans below are matches in Dolma (open web) — plausible '+
    'sources, not the actual training data of this model.</div>';
  h+='<div class="bar"><span>model <b>'+esc(t.model)+'</b></span>'+
     '<span>corpus <b>'+esc(p.index)+'</b></span>'+
     '<span>matched spans <b>'+p.n_matches+'</b></span>'+
     '<span>longest <b>'+p.longest_span_words+' words</b></span></div>';
  if(t.answer) h+='<div class="hit" style="border-color:#2d4a2d"><b>answer traced:</b> '+esc(t.answer)+'</div>';
  h+='<h2>1 · Pretraining provenance (verbatim corpus spans)</h2>';
  if(!p.matched_spans.length) h+='<div class="empty">No verbatim spans — answer is paraphrased / novel.</div>';
  for(const m of p.matched_spans.slice(0,15)){
    h+='<div class="span"><div class="ph">"'+esc(m.phrase)+'"</div>'+
       '<div class="meta">'+m.words+' words · '+m.count.toLocaleString()+' corpus hits</div>';
    for(const ps of (m.passages||[])){
      let tag='';const s=ps.source||{};const meta=s.metadata||{};
      const u=s.url||meta.url||s.path||'';
      if(u) tag='<span class="tag">['+esc(String(u)).slice(0,80)+']</span>';
      h+='<div class="pass">…'+esc(ps.text)+'… '+tag+'</div>';
    }
    h+='</div>';
  }
  h+='<h2>2 · Instruction-tuning provenance</h2>';
  const ins=t.instruction_tuning;
  if(ins.query_used==='(skipped)'){h+='<div class="empty">Instruction search skipped.</div>';return h}
  h+='<div class="meta">query: <b>'+esc(ins.query_used)+'</b></div>';
  for(const d of ins.datasets){
    const link=d.viewer_url?'<a class="dslink" href="'+esc(d.viewer_url)+'" target="_blank" rel="noopener">'+esc(d.dataset)+' ↗</a>':esc(d.dataset);
    h+='<h2 style="border:0;color:#e6edf3;text-transform:none;font-size:13px">'+link+'</h2>';
    if(d.error){h+='<div class="err">unavailable: '+esc(d.error)+'</div>';continue}
    h+='<div class="meta">'+(d.total_matches||0).toLocaleString()+' matching examples'+
       (d.scan_note?' · '+esc(d.scan_note):'')+
       (d.viewer_url?' · <a class="dslink" href="'+esc(d.viewer_url)+'" target="_blank" rel="noopener">search on HF ↗</a>':'')+'</div>';
    for(const hit of (d.hits||[])) h+=hitHtml(hit);
  }
  return h;
}
function hitHtml(hit){
  let inner='';
  if(hit.turns){
    inner=hit.turns.map(t=>'<div class="turn"><span class="role">'+esc(t.role)+'</span>'+
      esc(t.content)+'</div>').join('');
  }else{
    inner='<div class="turn">'+esc(hit.full||hit.snippet)+'</div>';
  }
  return '<details class="hitx"><summary>'+esc(hit.snippet)+'</summary>'+inner+'</details>';
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json", filename=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            opts = "".join(
                f'<option value="{m}"{" selected" if m == DEFAULT_MODEL else ""}>{m}</option>'
                for m in PRETRAIN_INDEXES
            )
            return self._send(200, PAGE.replace("__MODELS__", opts),
                              "text/html; charset=utf-8")
        if path == "/api/history":
            return self._send(200, json.dumps(store.list_records(), ensure_ascii=False))
        if path == "/api/export-all":
            return self._send(200, store.export_all(), "application/json",
                              filename="provenance-history.json")
        if path.startswith("/api/history/"):
            rec = store.load(path.rsplit("/", 1)[-1])
            if not rec:
                return self._send(404, json.dumps({"error": "not found"}))
            return self._send(200, json.dumps(rec, ensure_ascii=False))
        if path.startswith("/api/export/"):
            name = path.rsplit("/", 1)[-1]            # <id>.<fmt>
            rid, _, fmt = name.rpartition(".")
            rec = store.load(rid)
            if not rec:
                return self._send(404, json.dumps({"error": "not found"}))
            body, ctype = {
                "json": (store.export_json(rec), "application/json"),
                "md": (store.export_md(rec), "text/markdown; charset=utf-8"),
                "csv": (store.export_csv(rec), "text/csv; charset=utf-8"),
            }.get(fmt, (None, None))
            if body is None:
                return self._send(400, json.dumps({"error": "bad format"}))
            return self._send(200, body, ctype, filename=f"{rec['slug']}-{rid}.{fmt}")
        return self._send(404, "not found", "text/plain")

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/history/"):
            ok = store.delete(path.rsplit("/", 1)[-1])
            return self._send(200, json.dumps({"deleted": ok}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/trace":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            model = req.get("model", DEFAULT_MODEL)
            if model not in PRETRAIN_INDEXES:
                raise ValueError(f"unknown model {model}")
            answer = (req.get("answer") or "").strip()
            if req.get("generate") and not answer:
                from trace.generate import generate
                answer = generate(model, req.get("question", ""))
            if not answer:
                raise ValueError("answer is empty (or tick Generate)")
            trace = run(model, req.get("question", ""), answer,
                        do_instruction=bool(req.get("instruction")))
            self._send(200, json.dumps(trace, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._send(200, json.dumps({"error": str(e)}))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Provenance tracer UI -> {url}  (Ctrl+C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
