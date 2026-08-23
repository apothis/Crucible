"""Deterministic band wordmark (Pillow, GPU-free, Mac).

Krea2 misspelled the invented band name in 5 of 6 cover renders [OBSERVED 2026-08-23]:
diffusion text rendering pulls unfamiliar words toward its language prior, worst on the
small artist line. So the model is taken out of the loop for the band name: the art is
generated with only the song title in-model, and the wordmark is composited here with a
real font - the spelling is code, 6/6 by construction, and the band gets one consistent
logo across releases. Plain ASCII.
"""
import io
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Curated display faces present on macOS (checked on this Mac 2026-08-23). Only fonts
# whose file exists are offered, so a missing face degrades to a shorter list, not an error.
FONTS = [
    {"id": "luminari",    "label": "Luminari",       "path": "/System/Library/Fonts/Supplemental/Luminari.ttf"},
    {"id": "herculanum",  "label": "Herculanum",     "path": "/System/Library/Fonts/Supplemental/Herculanum.ttf"},
    {"id": "trattatello", "label": "Trattatello",    "path": "/System/Library/Fonts/Supplemental/Trattatello.ttf"},
    {"id": "copperplate", "label": "Copperplate",    "path": "/System/Library/Fonts/Supplemental/Copperplate.ttc"},
    {"id": "didot",       "label": "Didot",          "path": "/System/Library/Fonts/Supplemental/Didot.ttc"},
    {"id": "baskerville", "label": "Baskerville",    "path": "/System/Library/Fonts/Supplemental/Baskerville.ttc"},
    {"id": "cochin",      "label": "Cochin",         "path": "/System/Library/Fonts/Supplemental/Cochin.ttc"},
    {"id": "optima",      "label": "Optima",         "path": "/System/Library/Fonts/Optima.ttc"},
]

# Fill gradient (top -> bottom) + glow behind the letters. "shadow" is a flat fill with a
# heavy dark halo - the safe pick over busy artwork.
TREATMENTS = {
    "bone":   {"top": (242, 239, 228), "bottom": (196, 189, 168), "glow": (10, 8, 6),  "glow_alpha": 200, "glow_blur": 0.10},
    "silver": {"top": (240, 241, 246), "bottom": (134, 140, 155), "glow": (5, 6, 10),  "glow_alpha": 200, "glow_blur": 0.10},
    "gold":   {"top": (248, 231, 166), "bottom": (166, 123, 45),  "glow": (25, 12, 0), "glow_alpha": 200, "glow_blur": 0.10},
    "ember":  {"top": (255, 217, 160), "bottom": (226, 87, 28),   "glow": (45, 6, 0),  "glow_alpha": 225, "glow_blur": 0.14},
    "shadow": {"top": (245, 243, 236), "bottom": (245, 243, 236), "glow": (0, 0, 0),   "glow_alpha": 255, "glow_blur": 0.18},
}

DEFAULTS = {"text": "Apotheon", "font": "luminari", "treatment": "bone",
            "position": "bottom", "scale": 0.40}


def available_fonts():
    return [{"id": f["id"], "label": f["label"]} for f in FONTS if os.path.exists(f["path"])]


def _font_path(font_id):
    for f in FONTS:
        if f["id"] == font_id and os.path.exists(f["path"]):
            return f["path"]
    for f in FONTS:                                   # first available = fallback
        if os.path.exists(f["path"]):
            return f["path"]
    raise RuntimeError("no wordmark fonts available on this system")


def _render_mask(text, font, tracking):
    """Tightly-cropped L-mode mask of the text, drawn letter by letter so the
    wordmark carries a little extra tracking (display lettering breathes)."""
    dummy = ImageDraw.Draw(Image.new("L", (8, 8)))
    widths = [dummy.textlength(ch, font=font) for ch in text]
    ascent, descent = font.getmetrics()
    total = int(sum(widths) + tracking * max(0, len(text) - 1)) + 16
    mask = Image.new("L", (total, ascent + descent + 16), 0)
    d = ImageDraw.Draw(mask)
    x = 8.0
    for ch, w in zip(text, widths):
        d.text((x, 8), ch, font=font, fill=255)
        x += w + tracking
    bbox = mask.getbbox()
    return mask.crop(bbox) if bbox else mask


def _gradient(size, top, bottom):
    w, h = size
    g = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        g.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return g.resize((w, h))


def render_wordmark(text, font_id, treatment, height=240):
    """The wordmark as a tight RGBA image (transparent background): gradient-filled
    letters over a soft glow. `height` is the letter size; the canvas adds glow padding."""
    spec = TREATMENTS.get(treatment) or TREATMENTS["bone"]
    font = ImageFont.truetype(_font_path(font_id), size=height)
    mask = _render_mask(text, font, tracking=round(height * 0.06))
    w, h = mask.size
    pad = max(8, round(height * 0.30))
    W, H = w + pad * 2, h + pad * 2
    canvas = Image.new("L", (W, H), 0)
    canvas.paste(mask, (pad, pad))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # glow: the blurred mask as the alpha of a solid glow-color layer
    blur = canvas.filter(ImageFilter.GaussianBlur(max(1.0, height * spec["glow_blur"])))
    glow_a = blur.point(lambda v: min(255, round(v * spec["glow_alpha"] / 255)))
    glow = Image.new("RGBA", (W, H), spec["glow"] + (0,))
    glow.putalpha(glow_a)
    out = Image.alpha_composite(out, glow)
    # fill: vertical gradient clipped by the letter mask
    fill = _gradient((W, H), spec["top"], spec["bottom"]).convert("RGBA")
    fill.putalpha(canvas)
    return Image.alpha_composite(out, fill)


def preview_png(text, font_id, treatment, width=640, height=160):
    """The wordmark on a dark neutral card - what the picker grid shows."""
    card = Image.new("RGBA", (width, height), (18, 20, 26, 255))
    wm = render_wordmark(text or DEFAULTS["text"], font_id, treatment, height=140)
    scale = min((width * 0.88) / wm.width, (height * 0.80) / wm.height)
    wm = wm.resize((max(1, round(wm.width * scale)), max(1, round(wm.height * scale))), Image.LANCZOS)
    card.alpha_composite(wm, ((width - wm.width) // 2, (height - wm.height) // 2))
    buf = io.BytesIO()
    card.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# Placement grid: "<vertical>" or "<vertical>-<horizontal>". Vertical: top (below the
# in-model title band, which ends ~23% down), middle, bottom. Horizontal: left, center
# (default), right. e.g. "bottom", "bottom-right", "top-left", "middle".
POSITIONS = ["top", "top-left", "top-right", "middle", "middle-left", "middle-right",
             "bottom", "bottom-left", "bottom-right"]


def stamp(image_path, out_path, text, font_id, treatment, position="bottom", scale=0.40):
    """Composite the wordmark onto a cover still -> out_path (PNG). `scale` = wordmark
    width as a fraction of the image width (also capped to 16% of the image height so a
    tall face never dominates). `position` is a POSITIONS grid slot."""
    img = Image.open(image_path).convert("RGBA")
    wm = render_wordmark(text, font_id, treatment, height=260)
    scale = min(0.9, max(0.1, float(scale)))
    f = min((img.width * scale) / wm.width, (img.height * 0.16) / wm.height)
    wm = wm.resize((max(1, round(wm.width * f)), max(1, round(wm.height * f))), Image.LANCZOS)
    parts = str(position or "bottom").split("-")
    vert, horiz = parts[0], (parts[1] if len(parts) > 1 else "center")
    if horiz == "left":
        x = round(img.width * 0.06)
    elif horiz == "right":
        x = round(img.width * 0.94) - wm.width
    else:
        x = (img.width - wm.width) // 2
    if vert == "top":
        y = round(img.height * 0.24)
    elif vert == "middle":
        y = (img.height - wm.height) // 2
    else:
        y = round(img.height * 0.93) - wm.height
    img.alpha_composite(wm, (max(0, x), max(0, y)))
    img.convert("RGB").save(out_path, format="PNG")
    return out_path
