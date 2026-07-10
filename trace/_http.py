"""Tiny stdlib HTTP helpers (no external deps)."""
import json
import time
import urllib.request
import urllib.error


def post_json(url: str, payload: dict, timeout: int = 60, retries: int = 3) -> dict:
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    return {"error": last}


def get_json(url: str, token: str = "", timeout: int = 120, retries: int = 6) -> dict:
    """GET JSON. datasets-server builds its search index lazily, so the first
    calls may 500 ("loading") or time out — back off and retry generously."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            last = f"HTTP {e.code}: {body}"
            if "loading" in body.lower():   # steady poll while index builds
                time.sleep(15)
                continue
            if e.code in (500, 502, 503, 504):  # other transient
                time.sleep(min(8 * (attempt + 1), 30))
                continue
        except Exception as e:  # timeouts included — keep retrying
            last = str(e)
            time.sleep(min(8 * (attempt + 1), 30))
            continue
        time.sleep(2 * (attempt + 1))
    return {"error": last}
