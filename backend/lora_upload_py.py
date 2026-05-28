"""Mac-side client for the box LoRA dataset upload helper (lora_upload_server.py,
default :5080). Pushes the enriched dataset (audio + {name}.lyrics.txt + {name}.json)
to the box and returns the box-side folder paths the ACE engine's training pipeline
consumes (data_dir → /v1/dataset/scan; tensor_dir → preprocess; adapter_dir → export).
"""
import requests

HTTP_TIMEOUT = 30
UPLOAD_TIMEOUT = 600   # full songs can be large


def _base(host):
    host = (host or "").strip()
    if not host:
        raise RuntimeError("lora_upload_host not configured")
    return host if host.startswith("http") else f"http://{host}"


def health(host):
    r = requests.get(_base(host) + "/health", timeout=10)
    r.raise_for_status()
    return r.json()


def dataset_new(host, name):
    r = requests.post(_base(host) + "/dataset/new", data={"name": name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def dataset_paths(host, name):
    r = requests.get(_base(host) + "/dataset/paths", params={"name": name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def dataset_clear(host, name):
    r = requests.post(_base(host) + "/dataset/clear", data={"name": name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def dataset_delete(host, name):
    """Remove the entire dataset folder (data + tensors + train + adapters)."""
    r = requests.post(_base(host) + "/dataset/delete", data={"name": name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def dataset_list(host, name):
    r = requests.get(_base(host) + "/dataset/list", params={"name": name}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def upload(host, name, files):
    """Upload `files` = list of (filename, bytes) into the box dataset folder.
    Returns {written, count, data_dir}."""
    multipart = [("files", (fn, data)) for (fn, data) in files]
    r = requests.post(_base(host) + "/dataset/upload", data={"name": name},
                      files=multipart, timeout=UPLOAD_TIMEOUT)
    r.raise_for_status()
    return r.json()
