"""Mac-side client for the OFFICIAL ACE-Step 1.5 inference engine running on the
Windows GPU box (acestep-api, REST, default :8001). This is the full engine ACE
Studio uses — proper cover (`audio_cover_strength`), native repaint, etc.

Async task API (see ACE-Step docs/en/API.md, RESEARCH §13):
  POST /release_task   submit params (multipart when uploading src_audio) -> task_id
  POST /query_result   {"task_id_list":[id]} -> [{"status":1,"result":"<json>"}]  (1 = done)
  GET  /v1/audio?path= download a produced file
  GET  /health · /v1/models · /v1/stats

NOTE: field names / model ids / status codes below follow the published docs but are
UNVERIFIED against a live server (the box wasn't up when this was written). Verify with
health()/models() once the engine is reachable and adjust if needed — they're isolated
here on purpose.
"""
import json
import time
import requests

HTTP_TIMEOUT = 60          # per call
DEFAULT_DEADLINE = 5400    # max wait for one generation (generous: the FIRST request
                           # also triggers a model download that can take a while)


def _base(host):
    host = (host or "").strip()
    if not host:
        raise RuntimeError("acestep_host not configured")
    return host if host.startswith("http") else f"http://{host}"


def health(host):
    r = requests.get(_base(host) + "/health", timeout=10)
    r.raise_for_status()
    return r.json()


def models(host):
    r = requests.get(_base(host) + "/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()


def stats(host):
    r = requests.get(_base(host) + "/v1/stats", timeout=10)
    r.raise_for_status()
    return r.json()


def submit(host, fields, src_audio=None, ctx_audio=None):
    """POST /release_task. `fields` = dict of generation params. `src_audio`/`ctx_audio`
    = optional (bytes, filename) tuples uploaded as multipart files. Returns task_id."""
    url = _base(host) + "/release_task"
    files = {}
    if src_audio:
        files["src_audio"] = (src_audio[1] or "src.wav", src_audio[0], "application/octet-stream")
    if ctx_audio:
        files["ctx_audio"] = (ctx_audio[1] or "ctx.wav", ctx_audio[0], "application/octet-stream")
    if files:
        # multipart: params ride along as string form fields
        data = {k: ("" if v is None else str(v)) for k, v in fields.items()}
        r = requests.post(url, files=files, data=data, timeout=HTTP_TIMEOUT)
    else:
        r = requests.post(url, json=fields, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    d = j.get("data") or {}
    tid = d.get("task_id") or j.get("task_id")
    if not tid:
        raise RuntimeError(f"no task_id in /release_task response: {j}")
    return tid


def query(host, task_id):
    r = requests.post(_base(host) + "/query_result",
                      json={"task_id_list": [task_id]}, timeout=20)
    r.raise_for_status()
    j = r.json()
    arr = j.get("data") or []
    return arr[0] if arr else {}


def _one_file(r):
    if isinstance(r, dict):
        return r.get("file") or r.get("audio") or r.get("path")
    return None


def result_files(task):
    """All produced audio refs from a completed task's `result` (a JSON string that
    decodes to a LIST of song objects — the engine batches, default 2). Returns a list
    of file refs like '/v1/audio?path=...' (one per take)."""
    res = task.get("result")
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            return []
    if isinstance(res, dict):
        res = [res]
    if isinstance(res, list):
        return [f for f in (_one_file(r) for r in res) if f]
    return []


def result_file(task):
    files = result_files(task)
    return files[0] if files else None


def download(host, file_ref):
    """Download a produced file. `file_ref` may be a full '/v1/audio?path=...' URL or a
    bare path; we normalize to GET /v1/audio?path=<path>."""
    base = _base(host)
    if file_ref.startswith("/v1/audio"):
        url = base + file_ref
        r = requests.get(url, timeout=180)
    elif file_ref.startswith("http"):
        r = requests.get(file_ref, timeout=180)
    else:
        r = requests.get(base + "/v1/audio", params={"path": file_ref}, timeout=180)
    r.raise_for_status()
    return r.content


def wait(host, task_id, deadline=DEFAULT_DEADLINE, poll=5.0):
    """Poll /query_result until the task succeeds; return its file refs.
    Raises on failure/timeout. Status (per API.md): 0=queued/running, 1=succeeded, 2=failed."""
    start = time.time()
    while time.time() - start < deadline:
        t = query(host, task_id)
        st = t.get("status")
        if st == 1 or st == "succeeded":
            files = result_files(t)
            if not files:
                raise RuntimeError(f"task done but no file in result: {t}")
            return files                 # list of refs (one per batch take)
        failed = (st == 2 or (isinstance(st, int) and st < 0)
                  or (isinstance(st, str) and st.lower() in ("failed", "error")))
        if failed:
            raise RuntimeError(t.get("message") or t.get("error") or f"task failed: {t}")
        time.sleep(poll)
    raise RuntimeError(f"timed out after {deadline}s waiting for task {task_id}")
