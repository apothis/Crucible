# COMFY_LORA_PLAN.md - ComfyUI as a LoRA/LoKr inference (and maybe training) target for ACE-Step

Status: **RESEARCH ONLY. No implementation. Too many unknowns to commit.**
Created: 2026-05-31. Author context: Crucible (AI metal studio), ACE-Step 1.5 XL.

Provenance tags used below:
- `[cited]`   = from an upstream source (linked in §12), not tested by us.
- `[ours]`    = an established fact about OUR codebase/setup.
- `[hyp]`     = hypothesis / inference, not yet verified.
- `[verify]`  = an explicit thing to test before relying on it.

---

## 1. Why this exists

We currently run **generation + LoRA inference on the official ACE-Step 1.5 engine** (HTTP, the box). Its recurring pain is **VRAM-stickiness / opaque memory state**: multi-init sessions accumulate orphan tensors, base output degrades, and only an OS-level restart truly clears it (see [[engine-fresh-boot-for-lora]], and the whole 2026-05-31 debugging saga). The question: can we move **LoRA inference to ComfyUI** (which we already use for the pre-engine generation path) to get cleaner, per-generation model state - while optionally keeping the ACE-Step engine for **training** if we must.

The honest one-line conclusion: **ComfyUI can run our model and can do plain LoRA cleanly, but it CANNOT load our engine-trained LoKr as-is, and it does NOT clearly solve the memory pain.** So this is a "switch the adapter FORMAT" question more than a "switch the host" question.

---

## 2. What we already know works `[ours]` + `[cited]`

- **ComfyUI runs ACE-Step 1.5 XL.** Our `backend/comfy.py` `build_t2m` already drives 1.5 XL generation via the verified DualCLIPLoader(qwen_0.6b, qwen_4b) wiring. It is our documented fallback path (gated by `acestep_dcw_ok`). `[ours]`
- **ComfyUI has an official ACE-Step LoRA workflow** using the standard **`LoraLoaderModelOnly`** node + a `.safetensors` LoRA, `lora_weight` 0.0-1.0. `[cited]`
- **Plain LoRA adapters load and work** in ComfyUI for ACE-Step (community collection exists). `[cited]`
- **Multi-LoRA blending is native**: chain multiple `LoraLoaderModelOnly` nodes, each with its own strength. Cleaner than the engine's load/scale/unload dance. `[cited]`

---

## 3. The blocker: our LoKr will NOT load in ComfyUI `[cited]` (high confidence)

ComfyUI's loader rejects ACE-Step **LoKr** with `lora key not loaded` on exactly:
```
lycoris_condition_embedder.alpha
lycoris_condition_embedder.dora_scale
lycoris_condition_embedder.lokr_w1
lycoris_condition_embedder.lokr_w2_a
lycoris_condition_embedder.lokr_w2_b
```
(ComfyUI issue #12638, **open/unresolved**.) These are the **exact LyCORIS keys our engine writes** - we saw `lokr_w1/w2` and `dora_scale` when inspecting our own adapters on 2026-05-29. The reporter confirms the same file **loads in ACE-Step's Gradio/engine but not ComfyUI**, so it is a ComfyUI key-mapping gap, not a corrupt file. `[cited]` + matches `[ours]`

This is not ACE-Step-specific bad luck: LoKr trips ComfyUI across models (Flux #4476, Z-Image #10973). **LoKr is a poor ComfyUI citizen right now.**

**No off-the-shelf conversion fixes it.** The FL repo's `convert_lora_keys.py` only strips a doubled `base_model.model.` prefix; it has **no LyCORIS/LoKr logic** (no `lokr_w1`/`dora_scale` handling). `[cited]`

---

## 4. Plain LoRA is the realistic escape `[cited]`/`[hyp]`

Plain LoRA (not LoKr) loads cleanly in ComfyUI with strength + chaining. So the lever is **format**, not host. Our engine's trainer also exposes a **plain-LoRA path** (`PreprocessedLoRAModule` alongside `PreprocessedLoKRModule` in the patched `trainer.py`), so plain-LoRA training on the engine is likely reachable. `[ours]` (code exists) / `[verify]` (export key-compat with ComfyUI unconfirmed)

Trade-off to weigh: per [[engine-lokr-defaults]], LoKr/DoRA tends to extract a bit more from tiny datasets than plain LoRA, so a plain-LoRA adapter might be marginally weaker in character. Unmeasured on our data. `[hyp]`

---

## 5. The routes (pick later, after §9 verification)

### Route A - plain LoRA on the engine, infer in ComfyUI  (lowest change)
- Train **LoRA (not LoKr)** on the ACE-Step engine (same dataset/config), export.
- Load in our existing ComfyUI 1.5 workflow via `LoraLoaderModelOnly`.
- Keep ALL current training infra; only the adapter type + inference host change.
- Unknowns: does engine plain-LoRA export use ComfyUI-compatible keys? `[verify]` Quality vs LoKr? `[verify]`

### Route B - train AND infer entirely in ComfyUI  (drops the engine)
- Use `filliptm/ComfyUI-FL-AceStep-Training` (end-to-end ACE-Step **1.5** train->infer in ComfyUI's graph; "use ComfyUI's native LoRA loading nodes to apply your trained LoRA"). `[cited]`
- Removes the ACE-Step engine entirely (no more VRAM-stickiness from it).
- Unknowns: output format (LoRA vs LoKr) and clean-load are NOT explicitly documented; training quality, captioning pipeline, and our metadata-enrichment integration all unproven. `[verify]`

### Route C - make ComfyUI accept LoKr  (most work, least certain)
- Either (c1) patch ComfyUI's lora loader to map the ACE-Step LyCORIS LoKr keys, or (c2) write a real LyCORIS-LoKr -> ComfyUI converter (bake the LoKr Kronecker delta into a loadable form).
- We already patch box-side source (engine patches), so patching ComfyUI is in-character - but it is an upstream gap we'd be carrying, with maintenance burden, and (c2) may still not apply LoKr math even with renamed keys.
- Lowest priority unless A and B both fail and we specifically need to reuse existing LoKr adapters.

---

## 6. Multi-LoRA blending

- **ComfyUI (plain LoRA):** chain `LoraLoaderModelOnly` nodes, per-node strength. Native, clean, stateless per run. `[cited]`
- **Engine (current):** our `lora_runtime.reconcile()` already does deterministic multi-adapter + per-scale, verified - so we are NOT blocked on blending today; this is about doing it on a cleaner host.
- The `billwuhao/ComfyUI_ACE-Step` custom `ACELoRALoader` is **single-slot, no stacking** (unload before load) - so prefer the standard `LoraLoaderModelOnly` chain, not that node, for blends. `[cited]`

---

## 7. Memory reality - ComfyUI is NOT a clean win `[cited]` (the key caveat)

Moving to ComfyUI trades one set of memory problems for another:
- **#12440**: ACE-Step 1.5 "LM sampling" in ComfyUI can fall back to **CPU-speed**, dramatically slower than the engine's Gradio fast path. This directly threatens the reason we'd switch. `[cited]` `[verify on our 3090]`
- **#12541**: general ComfyUI memory-management regressions. `[cited]`
- ComfyUI **offloads VRAM<->RAM** each run; community unload nodes exist because people fight this. `[cited]`

Net: the **engine's** issue is VRAM-stickiness needing OS restarts; **ComfyUI's** is offload churn + a possible CPU-LM fallback. ComfyUI does give explicit per-node model control + unload nodes (fits our "clean state per generation" philosophy), but it is a different set of sharp edges, not a free lunch. **Do not assume the switch fixes the pain - measure it.**

---

## 8. Open unknowns (must resolve before committing) `[verify]`

1. Does an engine-trained **plain LoRA** export with ComfyUI-loadable keys, or does it also carry incompatible prefixes/format?
2. Quality: plain LoRA vs LoKr on OUR small datasets (BiB, Nightwish) - by ear + CLAP.
3. Is ComfyUI ACE-Step **1.5 XL inference fast enough on our 3090** (the #12440 CPU-LM risk), or is it a regression vs the engine?
4. FL ComfyUI trainer: what format does it output, does it load clean, and can its captioning/training match our enrichment pipeline quality?
5. Does ComfyUI multi-LoRA chaining behave for ACE-Step 1.5 the way it does for SD/Flux?
6. Memory: does ComfyUI actually hold a cleaner state across many generations on our box than the engine does?

---

## 9. Cheap verification experiments (NOT yet run - this session made NO moves)

Ordered by information-per-effort. Each is a small, reversible probe; none rips anything out.

- **E1 (answers #1, #2, #3 at once):** Train ONE plain-LoRA on the engine (BiB, same data as the 150ep LoKr). Export. Load it in our existing ComfyUI 1.5 workflow with `LoraLoaderModelOnly`. Generate the same BiB-style prompt/seed used in the 2026-05-31 LoKr A/B.
  - If it loads + sounds comparable + ComfyUI 1.5 inference is acceptably fast on the 3090 -> Route A is viable.
  - If "lora key not loaded" -> engine plain-LoRA export also needs key conversion (note the failing keys).
- **E2 (answers #3, #6 in isolation):** Run a handful of ComfyUI ACE-Step 1.5 generations (no LoRA) back-to-back, watch nvidia-smi + timing for the CPU-LM fallback and offload churn. Compares ComfyUI inference health vs the engine on OUR hardware.
- **E3 (answers #4):** Smoke-test `ComfyUI-FL-AceStep-Training` on a tiny (3-6 track) set; inspect the output adapter format + try loading it for inference in the same ComfyUI graph.
- **E4 (only if A and B fail):** Prototype Route C - inspect a ComfyUI lora-key map and see whether the ACE-Step LyCORIS LoKr keys can be remapped/converted.

Decision after E1-E3: pick Route A (keep engine training, ComfyUI inference), Route B (full ComfyUI), or **stay on the engine** (if ComfyUI's inference health is worse on our box - a legitimate outcome).

---

## 10. How this maps to our code (so a switch is cheap)

- `backend/lora_runtime.py` is the **modular seam**: today it reconciles adapters on the ACE-Step engine. A ComfyUI target would be a sibling implementation (build the LoRA-loader chain into the comfy graph) behind the same interface. The **LoRA picker UI does not change** - it just emits `[{path, scale}]`.
- `backend/comfy.py` `build_t2m` is where LoRA-loader nodes would be inserted for the ComfyUI path.
- `/api/lora/adapters/all` enumeration would point at a ComfyUI `models/loras/` dir instead of (or in addition to) the box `lora_data` runs.

So none of the picker/reconcile work from 2026-05-31 is wasted regardless of which route wins.

---

## 11. Recommendation

1. **Keep training on the ACE-Step engine for now** (it works; the metadata-enrichment + captioning pipeline is mature).
2. **Treat ComfyUI as an inference target to VALIDATE, not adopt** - via E1/E2 - before changing anything.
3. **Do not try to force our existing LoKr into ComfyUI** - the format gap is real, upstream, unfixed. If we want ComfyUI, switch the adapter format to plain LoRA.
4. **Make no moves until the unknowns in §8 are tested.** This document is the parking spot for that.

---

## 12. Sources

- ACE-Step ComfyUI native + LoRA workflow: https://comfyui-wiki.com/en/tutorial/advanced/audio/ace-step/ace-step-v1 ; https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1
- `LoraLoaderModelOnly` node: https://comfyui-wiki.com/en/comfyui-nodes/loaders/lora-loader-model-only
- LoKr fails to load (ACE-Step 1.5), the blocker: https://github.com/Comfy-Org/ComfyUI/issues/12638
- LoKr fails on other models too: https://github.com/Comfy-Org/ComfyUI/issues/4476 (Flux) ; https://github.com/Comfy-Org/ComfyUI/issues/10973 (Z-Image)
- billwuhao ComfyUI_ACE-Step LoRA customization (single-slot ACELoRALoader): https://deepwiki.com/billwuhao/ComfyUI_ACE-Step/4.4-lora-customization
- ComfyUI-native ACE-Step 1.5 LoRA TRAINING nodes (Route B): https://github.com/filliptm/ComfyUI-FL-AceStep-Training
- The (insufficient) key converter: https://github.com/filliptm/ComfyUI-FL-AceStep-Training/blob/master/convert_lora_keys.py
- Plain ACE-Step LoRA collection (works in ComfyUI): https://huggingface.co/woctordho/ACE-Step-v1-LoRA-collection
- ComfyUI ACE-Step 1.5 LM-sampling CPU fallback (memory/perf caveat): https://github.com/Comfy-Org/ComfyUI/issues/12440
- ComfyUI memory-management regression: https://github.com/Comfy-Org/ComfyUI/issues/12541
- ComfyUI dynamic VRAM / offload behavior: https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local
- ComfyUI unload-model node (people fight offload): https://github.com/SeanScripts/ComfyUI-Unload-Model
