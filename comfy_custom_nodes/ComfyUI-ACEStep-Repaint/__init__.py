"""
ACE-Step nodes for ComfyUI.

This package provides nodes for ACE-Step 1.0 and 1.5 audio generation,
including extend, repaint, cover, and extract operations.
"""

import logging
import os

# Create logger for the acestep module
logger = logging.getLogger("comfyui_ryanontheinside.acestep")

# Set default level from environment variable or default to INFO
# Use ACESTEP_LOG_LEVEL=DEBUG for verbose output
log_level = os.environ.get("ACESTEP_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Only add handler if none exist (avoid duplicate handlers on reload)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)

# Prevent propagation to root logger to avoid duplicate messages
logger.propagate = False

# --- Standalone custom-node registration (vendored into Crucible) ---
# Upstream (ryanontheinside/ComfyUI_RyanOnTheInside) aggregates these mappings in
# its top-level package __init__. Vendored on its own, ComfyUI loads THIS folder's
# __init__ and looks for NODE_CLASS_MAPPINGS here, so re-export them from .nodes.
# (logger is defined above first, so nodes.py's `from . import logger` resolves.)
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
