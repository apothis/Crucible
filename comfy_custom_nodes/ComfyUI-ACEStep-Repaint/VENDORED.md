# Vendored: ACE-Step repaint/extend nodes

These files are vendored verbatim from the **`nodes/acestep/`** folder of
[`ryanontheinside/ComfyUI_RyanOnTheInside`](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside)
(MIT — see `LICENSE`). Only this self-contained subfolder is included, to avoid
the parent pack's heavy unrelated dependencies (pygame, pymunk, opencv, openunmix,
scikit-image, matplotlib). These nodes depend only on `torch`, `numpy`, and core
`comfy.*` modules already present in ComfyUI.

**Why vendored:** Crucible's repaint/extend features (`/api/repaint`, `/api/extend`)
drive the `ACEStep15NativeEditGuider` node. Pinning it here keeps the box install
lean and version-locked to what Crucible was tested against.

## Install on the ComfyUI (GPU) box
Copy this whole folder into the box's `ComfyUI/custom_nodes/`, then restart ComfyUI.
No `pip install` needed. The key nodes register as:
- **ACEStep15NativeEditGuider** — unified extend + repaint (the one Crucible uses)
- ACEStep15NativeCoverGuider / ExtractGuider / LegoGuider, plus 1.0 guiders & utils.

## Updating
Re-copy from upstream `nodes/acestep/` and re-apply the one local change:
`__init__.py` appends `from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS`
so the folder registers as a standalone custom node (upstream exposes the mappings
from its top-level package instead).

Local edits to upstream files: **none** except the `__init__.py` re-export above.
