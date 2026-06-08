#!/usr/bin/env python3
"""Weight-space overfit-monitor metrics for LoKr/LoRA adapters (METAL_LORA_PLAN §13m).

Computes the cheap, val-free, CPU-side weight-space signals the deep-research pass
recommended (stable rank, effective rank, spectral/Frobenius norm, condition number)
on saved `lokr_weights.safetensors` checkpoints, so we can line them up against the
user's EAR verdicts and see whether any of them actually separates ear-judged-overfit
from ear-judged-good. NO GPU, NO generation, NO retraining.

Usage:
  python tools/lora_overfit_metrics.py <dir-of-safetensors> [--json out.json] [--inspect]

Filenames are expected flattened by the box copy step, e.g.
  train_20260608-130123__lokrv2_150ep_prodigy_cfg0.1__final__lokr_weights.safetensors
  train_20260607-182943__lokr_150ep..__checkpoints__epoch_40__lokr_weights.safetensors
We parse (run, stage/epoch) from the name; --inspect dumps key/shape layout of the
first file so the metric extraction can be confirmed against the real LoKr layout.
"""
import argparse, glob, json, math, os, re, sys
import numpy as np

try:
    from safetensors.numpy import load_file as _load_st
except Exception:
    _load_st = None


def _svals(mat):
    """Singular values of a 2D matrix (float64, CPU)."""
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or min(m.shape) == 0:
        return None
    try:
        return np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return None


def _rank_metrics(s):
    """stable rank, effective (entropy) rank, condition number from singular values."""
    s = s[s > 0]
    if s.size == 0:
        return None
    smax = float(s[0]) if s[0] >= s[-1] else float(s.max())
    fro2 = float((s ** 2).sum())
    spec2 = float(s.max() ** 2)
    stable = fro2 / spec2 if spec2 > 0 else float("nan")
    p = s / s.sum()
    eff = float(math.exp(-(p * np.log(p)).sum()))   # effective rank (Roy & Vetterli)
    cond = float(s.max() / s.min()) if s.min() > 0 else float("inf")
    return {"stable_rank": stable, "effective_rank": eff, "condition": cond,
            "spectral": float(s.max()), "fro": math.sqrt(fro2)}


def _module_groups(tensors):
    """Group tensor keys by module prefix (strip the trailing .lokr_*/.alpha suffix)."""
    groups = {}
    for k in tensors:
        base = re.sub(r'\.(lokr_w1|lokr_w2|lokr_w1_a|lokr_w1_b|lokr_w2_a|lokr_w2_b|alpha|dora_scale).*$', '', k)
        groups.setdefault(base, {})[k.split('.')[-1] if '.' in k else k] = k
    return groups


def _reconstruct_lowrank(tensors, keymap):
    """Return the most informative 2D low-rank matrix for a module:
    w2 = w2_a @ w2_b if decomposed, else lokr_w2, else lokr_w1."""
    def get(suffix):
        for kk, fullk in keymap.items():
            if kk == suffix:
                return tensors[fullk]
        return None
    w2a, w2b = get("lokr_w2_a"), get("lokr_w2_b")
    if w2a is not None and w2b is not None:
        a, b = np.asarray(w2a, np.float64), np.asarray(w2b, np.float64)
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            return a @ b
    for suffix in ("lokr_w2", "lokr_w1"):
        w = get(suffix)
        if w is not None and np.asarray(w).ndim == 2:
            return np.asarray(w, np.float64)
    return None


def analyze_file(path):
    if _load_st is None:
        raise RuntimeError("safetensors not installed: pip install safetensors")
    tensors = _load_st(path)
    total_params = int(sum(np.asarray(v).size for v in tensors.values()))
    global_fro = math.sqrt(float(sum((np.asarray(v, np.float64) ** 2).sum() for v in tensors.values())))
    per_mod = []
    for base, keymap in _module_groups(tensors).items():
        mat = _reconstruct_lowrank(tensors, keymap)
        if mat is None:
            continue
        s = _svals(mat)
        if s is None:
            continue
        rm = _rank_metrics(s)
        if rm:
            per_mod.append(rm)
    if not per_mod:
        return {"total_params": total_params, "global_fro": global_fro, "n_modules": 0}
    agg = {"total_params": total_params, "global_fro": global_fro, "n_modules": len(per_mod)}
    for m in ("stable_rank", "effective_rank", "condition", "spectral", "fro"):
        vals = np.array([d[m] for d in per_mod if np.isfinite(d[m])], dtype=np.float64)
        if vals.size:
            agg[f"{m}_mean"] = float(vals.mean())
            agg[f"{m}_median"] = float(np.median(vals))
    return agg


def parse_name(fn):
    base = os.path.basename(fn)
    run = base.split("__")[0]
    stage, epoch = "?", None
    if "__final__" in base:
        stage = "final"
    elif "__best__" in base or "checkpoints__best" in base:
        stage = "best"
    m = re.search(r'epoch[_-](\d+)', base)
    if m:
        stage, epoch = "epoch", int(m.group(1))
    # carry the config tag (after run timestamp) for readability
    tag = base.split("__")
    cfg = tag[1] if len(tag) > 1 else ""
    return run, cfg, stage, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default="library/lora_train_history/overfit_metrics.json")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "**", "*.safetensors"), recursive=True))
    if not files:
        print("no .safetensors found under", a.dir); sys.exit(1)
    if a.inspect:
        if _load_st is None:
            print("safetensors not installed"); sys.exit(1)
        t = _load_st(files[0])
        print("INSPECT", os.path.basename(files[0]), "-", len(t), "tensors")
        for k in list(t)[:30]:
            print(f"  {k}  {np.asarray(t[k]).shape}  {np.asarray(t[k]).dtype}")
        return
    rows = []
    for f in files:
        run, cfg, stage, epoch = parse_name(f)
        try:
            agg = analyze_file(f)
        except Exception as e:
            print("ERR", os.path.basename(f), e); continue
        rows.append({"run": run, "cfg": cfg, "stage": stage, "epoch": epoch, **agg})
    rows.sort(key=lambda r: (r["run"], r["epoch"] if r["epoch"] is not None else (10**6 if r["stage"]=="final" else 10**5)))
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(rows, open(a.json, "w"), indent=1)
    # table
    cols = ["run","cfg","stage","epoch","global_fro","stable_rank_mean","effective_rank_mean","condition_mean","spectral_mean"]
    print(f"{'run/cfg':42} {'stage':6} {'ep':>4} {'fro':>9} {'stbl_rk':>8} {'eff_rk':>7} {'cond':>9} {'spec':>8}")
    for r in rows:
        name = (r["cfg"] or r["run"])[:42]
        print(f"{name:42} {r['stage']:6} {str(r['epoch'] or ''):>4} "
              f"{r.get('global_fro',0):9.3f} {r.get('stable_rank_mean',float('nan')):8.3f} "
              f"{r.get('effective_rank_mean',float('nan')):7.3f} {r.get('condition_mean',float('nan')):9.1f} "
              f"{r.get('spectral_mean',float('nan')):8.4f}")
    print(f"\nwrote {a.json}  ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
