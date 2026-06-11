# Engine patch 2026-06-11 - LyCORIS LoKr rank_dropout device-bug fix

THIRD-PARTY (pip) file patch, not engine source. Fixes a genuine bug in the installed
`lycoris-lora` so `rank_dropout` works for LoKr on CUDA.

## Why
`rank_dropout` is the only FINE-GRAINED regularizer LyCORIS actually implements for LoKr
(normal `dropout` is a no-op, module_dropout is coarse - see [[lokr-dropout-lycoris]]). But it
CRASHES at step 0 with `RuntimeError: Expected all tensors to be on the same device, cuda:0 and
cpu`. Cause (lycoris/modules/lokr.py:376): `drop = (torch.rand(weight.size(0)) > self.rank_dropout)
.to(dtype)` - `torch.rand` defaults to CPU and `.to(dtype)` casts dtype only, NOT device - then
`weight *= drop` (line 380) multiplies a CPU mask against the cuda weight. Fires every forward via
get_weight (even on disabled non-target modules).

## The fix (ONE line, 376)
- before: `drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(dtype)`
- after:  `drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(dtype)`
`diff` proves exactly one line changed; `py_compile` clean. Built from LyCORIS GitHub main, which
matches the box's installed version EXACTLY (the box traceback line numbers - 376 rand / 380
`weight *= drop` / 553 get_weight call - align line-for-line, = our box-diff-check).

## Deploy (box) + caveat
Copy over, then OS-restart the engine ([[engine-restart-is-user-only]]):
- `lokr.py` -> `E:\AI\MusicGen\AceStep\ACE-Step-1.5\.venv\Lib\site-packages\lycoris\modules\lokr.py`
CAVEAT: this is in site-packages, so ANY `uv add`/`pip install` that reinstalls lycoris-lora
REVERTS it (more fragile than our engine-source patches). Re-apply after any lycoris reinstall.

## Verify after restart
Fire a run with `rank_dropout: 0.1`. Engine console should show NO `cuda:0 and cpu` crash, NO
`[WARN]...normal dropout` spam (that's only for normal `dropout`), `LoKr dropout config: ...
rank_dropout=0.1`, and it should clear step 0 into real training.
