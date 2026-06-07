"""Mac-side client for the official ACE-Step 1.5 engine's LoRA-training + dataset
pipeline (the box, default :8001). The engine exposes the WHOLE pipeline over HTTP
— scan a folder → auto-label captions (box LM) → save dataset → preprocess to
tensors (GPU) → train LoRA/LoKr → export adapter → load/scale at inference — so we
drive it all remotely with no Gradio on the box.

Verified live 2026-05-27 against 192.168.1.201:8001 (openapi.json + status probes).
Full contract + plan: METAL_LORA_PLAN.md §2. Responses are wrapped
{data, code, error, timestamp}; helpers below return the unwrapped `data`.

GPU/VRAM: auto_label needs the LM loaded; preprocess/train need the DiT+decoder
(swap via init()/reinitialize()). Training is GPU-exclusive — free the other box
services first and don't generate concurrently.
"""
import time
import requests

from .acestep_py import _base   # reuse host normalization ("host:port" -> http URL)

HTTP_TIMEOUT = 60
TRAIN_POLL = 5.0


def _unwrap(r):
    """Raise on HTTP error, then return the engine's `data` payload (or the raw json
    if it isn't the standard {data,code,error} envelope)."""
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and "data" in j and "code" in j:
        if j.get("error"):
            raise RuntimeError(str(j["error"]))
        return j["data"]
    return j


def _post(host, path, body=None, timeout=HTTP_TIMEOUT):
    return _unwrap(requests.post(_base(host) + path, json=(body or {}), timeout=timeout))


def _get(host, path, params=None, timeout=HTTP_TIMEOUT):
    return _unwrap(requests.get(_base(host) + path, params=(params or {}), timeout=timeout))


# ---------------- model load / VRAM swap ----------------
def init(host, model=None, init_llm=False, lm_model_path=None, timeout=600):
    """Load models. For auto-labelling pass init_llm=True (+ lm_model_path); for
    preprocess/train load the DiT (init_llm=False)."""
    body = {}
    if model is not None:
        body["model"] = model
    body["init_llm"] = bool(init_llm)
    if lm_model_path:
        body["lm_model_path"] = lm_model_path
    return _post(host, "/v1/init", body, timeout=timeout)


def reinitialize(host, timeout=600):
    """Reload the model/decoder (the API equivalent of 'restart Gradio to free VRAM';
    training requires the decoder present)."""
    return _post(host, "/v1/reinitialize", {}, timeout=timeout)


# ---------------- dataset pipeline ----------------
def dataset_scan(host, audio_dir, dataset_name="crucible_metal", custom_tag="",
                 tag_position="replace", all_instrumental=False):
    return _post(host, "/v1/dataset/scan", {
        "audio_dir": audio_dir, "dataset_name": dataset_name,
        "custom_tag": custom_tag, "tag_position": tag_position,
        "all_instrumental": bool(all_instrumental)})


def dataset_samples(host):
    return _get(host, "/v1/dataset/samples")


def dataset_get_sample(host, idx):
    return _get(host, f"/v1/dataset/sample/{int(idx)}")


def dataset_put_sample(host, idx, sample):
    return _unwrap(requests.put(_base(host) + f"/v1/dataset/sample/{int(idx)}",
                                json=sample, timeout=HTTP_TIMEOUT))


def dataset_auto_label_async(host, only_unlabeled=False, transcribe_lyrics=False,
                             format_lyrics=False, lm_model_path=None, save_path=None):
    """Caption the scanned dataset via the box LM (async). Returns a task handle; poll
    auto_label_status(). NOTE: the LM hallucinates BPM/key — supply those ourselves.

    The engine's openapi advertises chunk_size + batch_size params, but those don't
    actually reach label_all_samples() — passing them errors with 'unexpected keyword
    argument' (engine schema/impl drift, verified 2026-05-28). So we deliberately
    DON'T send them; the engine uses its internal defaults."""
    body = {"only_unlabeled": bool(only_unlabeled),
            "transcribe_lyrics": bool(transcribe_lyrics),
            "format_lyrics": bool(format_lyrics)}
    if lm_model_path:
        body["lm_model_path"] = lm_model_path
    if save_path:
        body["save_path"] = save_path
    return _post(host, "/v1/dataset/auto_label_async", body)


def auto_label_status(host, task_id=None):
    path = f"/v1/dataset/auto_label_status/{task_id}" if task_id else "/v1/dataset/auto_label_status"
    return _get(host, path)


def dataset_save(host, save_path, dataset_name="crucible_metal", custom_tag=None,
                 tag_position=None, all_instrumental=None, genre_ratio=0):
    body = {"save_path": save_path, "dataset_name": dataset_name, "genre_ratio": genre_ratio}
    if custom_tag is not None:
        body["custom_tag"] = custom_tag
    if tag_position is not None:
        body["tag_position"] = tag_position
    if all_instrumental is not None:
        body["all_instrumental"] = bool(all_instrumental)
    return _post(host, "/v1/dataset/save", body)


def dataset_preprocess_async(host, output_dir, skip_existing=False):
    """Encode audio → latents/tensors on the box GPU (async). Poll preprocess_status()."""
    return _post(host, "/v1/dataset/preprocess_async",
                 {"output_dir": output_dir, "skip_existing": bool(skip_existing)})


def preprocess_status(host, task_id=None):
    path = f"/v1/dataset/preprocess_status/{task_id}" if task_id else "/v1/dataset/preprocess_status"
    return _get(host, path)


# ---------------- training ----------------
def train_lora(host, tensor_dir, lora_output_dir, *, lora_rank=64, lora_alpha=128,
               lora_dropout=0.1, learning_rate=1e-4, train_epochs=500, train_batch_size=1,
               gradient_accumulation=4, save_every_n_epochs=5, training_shift=3.0,
               training_seed=42, use_fp8=False, gradient_checkpointing=False):
    return _post(host, "/v1/training/start", {
        "tensor_dir": tensor_dir, "lora_output_dir": lora_output_dir,
        "lora_rank": lora_rank, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "learning_rate": learning_rate, "train_epochs": train_epochs,
        "train_batch_size": train_batch_size, "gradient_accumulation": gradient_accumulation,
        "save_every_n_epochs": save_every_n_epochs, "training_shift": training_shift,
        "training_seed": training_seed, "use_fp8": use_fp8,
        "gradient_checkpointing": gradient_checkpointing})


def train_lokr(host, tensor_dir, output_dir, *, lokr_linear_dim=64, lokr_linear_alpha=128,
               lokr_factor=-1, lokr_weight_decompose=True, lokr_decompose_both=False,
               lokr_use_tucker=False, lokr_use_scalar=False, learning_rate=0.03,
               train_epochs=500, train_batch_size=1, gradient_accumulation=4,
               save_every_n_epochs=5, training_shift=3.0, training_seed=42,
               gradient_checkpointing=False, val_split=None,
               timestep_sampling_mode=None, target_modules=None,
               lora_dropout=None, rank_dropout=None, module_dropout=None):
    # val_split + timestep_sampling_mode + target_modules + *_dropout are
    # Crucible-patched server-side (2026-05-29 / 2026-05-30 / 2026-06-06 /
    # 2026-06-07 engine patches). Only sent when the caller provides explicitly
    # so we stay compatible with unpatched engines (Pydantic ignores unknown
    # fields by default; older engines silently drop -> falls back to the default
    # attention-only target_modules and zero dropout).
    body = {
        "tensor_dir": tensor_dir, "output_dir": output_dir,
        "lokr_linear_dim": lokr_linear_dim, "lokr_linear_alpha": lokr_linear_alpha,
        "lokr_factor": lokr_factor, "lokr_weight_decompose": lokr_weight_decompose,
        "lokr_decompose_both": lokr_decompose_both, "lokr_use_tucker": lokr_use_tucker,
        "lokr_use_scalar": lokr_use_scalar, "learning_rate": learning_rate,
        "train_epochs": train_epochs, "train_batch_size": train_batch_size,
        "gradient_accumulation": gradient_accumulation, "save_every_n_epochs": save_every_n_epochs,
        "training_shift": training_shift, "training_seed": training_seed,
        "gradient_checkpointing": gradient_checkpointing}
    if val_split is not None:
        body["val_split"] = float(val_split)
    if timestep_sampling_mode is not None:
        body["timestep_sampling_mode"] = str(timestep_sampling_mode)
    if target_modules is not None:
        body["target_modules"] = list(target_modules)
    if lora_dropout is not None:
        body["lora_dropout"] = float(lora_dropout)
    if rank_dropout is not None:
        body["rank_dropout"] = float(rank_dropout)
    if module_dropout is not None:
        body["module_dropout"] = float(module_dropout)
    return _post(host, "/v1/training/start_lokr", body)


def training_status(host):
    return _get(host, "/v1/training/status")


def training_stop(host):
    return _post(host, "/v1/training/stop", {})


def training_export(host, export_path, lora_output_dir):
    return _post(host, "/v1/training/export",
                 {"export_path": export_path, "lora_output_dir": lora_output_dir})


# ---------------- inference-time LoRA ----------------
def lora_load(host, lora_path, adapter_name=None):
    body = {"lora_path": lora_path}
    if adapter_name:
        body["adapter_name"] = adapter_name
    return _post(host, "/v1/lora/load", body)


def lora_scale(host, scale, adapter_name=None):
    body = {"scale": float(scale)}
    if adapter_name:
        body["adapter_name"] = adapter_name
    return _post(host, "/v1/lora/scale", body)


def lora_toggle(host, use_lora):
    return _post(host, "/v1/lora/toggle", {"use_lora": bool(use_lora)})


def lora_unload(host):
    return _post(host, "/v1/lora/unload", {})


def lora_status(host):
    return _get(host, "/v1/lora/status")


# ---------------- generic async poller ----------------
def poll_async(host, kind, task_id=None, deadline=7200, poll=TRAIN_POLL, on_progress=None):
    """Poll an async op until it finishes. `kind` in {'auto_label','preprocess','training'}.
    Returns the final status dict. Best-effort 'done' detection across the engine's
    status shapes (training uses is_training=false; dataset ops report a status/progress)."""
    getter = {"auto_label": lambda: auto_label_status(host, task_id),
              "preprocess": lambda: preprocess_status(host, task_id),
              "training": lambda: training_status(host)}[kind]
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline:
        st = getter() or {}
        last = st
        if on_progress:
            try:
                on_progress(st)
            except Exception:
                pass
        if kind == "training":
            if not st.get("is_training", True):
                return st
        else:
            s = str(st.get("status", "")).lower()
            if st.get("done") or st.get("finished") or s in ("done", "complete", "completed", "idle", "error", "failed"):
                return st
            if st.get("error"):
                return st
        time.sleep(poll)
    return last or {}
