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

# Curated display faces. The repo `fonts/` dir carries OFL faces from google/fonts
# (downloaded 2026-08-24; far better display range than what macOS ships - blackletter,
# carved epic serifs, elegant scripts - which is what made the model's in-image lettering
# attractive). macOS Supplemental faces fill out the list. Only fonts whose file exists
# are offered, so a missing face degrades to a shorter list, not an error.
_REPO_FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONTS = [
    # gothic / blackletter (the look the Krea2 renders used for "Angel")
    {"id": "unifraktur",  "label": "UnifrakturMaguntia", "path": os.path.join(_REPO_FONTS, "UnifrakturMaguntia.ttf")},
    {"id": "pirataone",   "label": "Pirata One",         "path": os.path.join(_REPO_FONTS, "PirataOne.ttf")},
    {"id": "grenze",      "label": "Grenze Gotisch",     "path": os.path.join(_REPO_FONTS, "GrenzeGotisch.ttf")},
    {"id": "metalmania",  "label": "Metal Mania",        "path": os.path.join(_REPO_FONTS, "MetalMania.ttf")},
    {"id": "newrocker",   "label": "New Rocker",         "path": os.path.join(_REPO_FONTS, "NewRocker.ttf")},
    # carved / epic serif
    {"id": "cinzel",      "label": "Cinzel",             "path": os.path.join(_REPO_FONTS, "Cinzel.ttf")},
    {"id": "cinzeldeco",  "label": "Cinzel Decorative",  "path": os.path.join(_REPO_FONTS, "CinzelDecorative.ttf")},
    {"id": "eaglelake",   "label": "Eagle Lake",         "path": os.path.join(_REPO_FONTS, "EagleLake.ttf")},
    # elegant script / italic (the Bitter Honey look)
    {"id": "greatvibes",  "label": "Great Vibes",        "path": os.path.join(_REPO_FONTS, "GreatVibes.ttf")},
    {"id": "italianno",   "label": "Italianno",          "path": os.path.join(_REPO_FONTS, "Italianno.ttf")},
    {"id": "cormorant",   "label": "Cormorant Italic",   "path": os.path.join(_REPO_FONTS, "CormorantItalic.ttf")},
    # macOS faces
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
    "crimson": {"top": (238, 110, 100), "bottom": (120, 8, 16),   "glow": (30, 0, 2),  "glow_alpha": 230, "glow_blur": 0.12},
    "shadow": {"top": (245, 243, 236), "bottom": (245, 243, 236), "glow": (0, 0, 0),   "glow_alpha": 255, "glow_blur": 0.18},
}

# `title_*` = the SONG TITLE's stamped lettering (large, top band); the bare keys are the
# band wordmark. Both live in one saved config blob (app_config `yt_wordmark`).
DEFAULTS = {"text": "Apotheon", "font": "luminari", "treatment": "bone",
            "position": "bottom", "scale": 0.40,
            "title_font": "unifraktur", "title_treatment": "ember",
            "title_position": "header", "title_scale": 0.72}


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


# Placement grid: "<vertical>" or "<vertical>-<horizontal>". Vertical: header (the title
# band at the very top), top (below that band, ~24% down), middle, bottom. Horizontal:
# left, center (default), right. e.g. "bottom", "bottom-right", "header", "top-left".
POSITIONS = ["header", "header-left", "header-right",
             "top", "top-left", "top-right", "middle", "middle-left", "middle-right",
             "bottom", "bottom-left", "bottom-right"]


def _place(img, wm, position):
    parts = str(position or "bottom").split("-")
    vert, horiz = parts[0], (parts[1] if len(parts) > 1 else "center")
    if horiz == "left":
        x = round(img.width * 0.06)
    elif horiz == "right":
        x = round(img.width * 0.94) - wm.width
    else:
        x = (img.width - wm.width) // 2
    if vert == "header":
        y = round(img.height * 0.055)
    elif vert == "top":
        y = round(img.height * 0.24)
    elif vert == "middle":
        y = (img.height - wm.height) // 2
    else:
        y = round(img.height * 0.93) - wm.height
    return max(0, x), max(0, y)


def compose_layers(img, layers):
    """Composite text layers onto an RGBA image IN PLACE. Each layer:
    {text, font, treatment, position, scale, max_h?}. `scale` = layer width as a fraction
    of the image width; `max_h` caps the height as a fraction (default 0.16 - the song
    TITLE passes ~0.22 so a tall blackletter face can carry a full headline)."""
    for l in layers:
        text = str(l.get("text") or "").strip()
        if not text:
            continue
        wm = render_wordmark(text, str(l.get("font") or "luminari"),
                             str(l.get("treatment") or "bone"), height=260)
        scale = min(0.95, max(0.1, float(l.get("scale") or 0.40)))
        max_h = min(0.35, max(0.06, float(l.get("max_h") or 0.16)))
        f = min((img.width * scale) / wm.width, (img.height * max_h) / wm.height)
        wm = wm.resize((max(1, round(wm.width * f)), max(1, round(wm.height * f))), Image.LANCZOS)
        img.alpha_composite(wm, _place(img, wm, l.get("position")))
    return img


def layers_plate(width, height, layers, out_path):
    """The same text layers on a TRANSPARENT canvas -> RGBA PNG. Used as a static
    ffmpeg overlay so lettering can sit on top of VIDEO (living covers)."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    compose_layers(img, layers).save(out_path, format="PNG")
    return out_path


def stamp_layers(image_path, out_path, layers):
    """Composite one or more text layers onto a cover still -> out_path (PNG)."""
    img = Image.open(image_path).convert("RGBA")
    compose_layers(img, layers).convert("RGB").save(out_path, format="PNG")
    return out_path


def stamp(image_path, out_path, text, font_id, treatment, position="bottom", scale=0.40):
    """Single-layer convenience wrapper (the band wordmark) over stamp_layers."""
    return stamp_layers(image_path, out_path,
                        [{"text": text, "font": font_id, "treatment": treatment,
                          "position": position, "scale": scale}])


# Layout B ("card visualizer") geometry on a 1920x1080 canvas - the layout the user
# picked from the mock PDF (2026-09-02): art card left, band + title column right,
# waveform under the text. All consumers (plate below, musicvideo.card_video) share it.
CARD_LAYOUT = {"card_x": 106, "card_y": 288, "card_w": 896, "card_h": 504,   # exact 16:9, EVEN dims (yuv420p)
               "col_x": 1088, "col_w": 760,
               "band_y": 300, "band_max_h": 210,
               "title_y": 480, "title_max_h": 240,
               "wave_x": 1118, "wave_y": 660, "wave_w": 700, "wave_h": 70}


# YouTube end-screen landing zone for layout B (user-approved mock 2026-09-16): for the
# final N seconds the right column drops the title + waveform and shows two 16:9 panels
# (YouTube "video" elements) side by side with the subscribe circle under them. The card
# window and the band wordmark stay exactly where they are. Studio elements are placed
# over these rectangles ONCE, then "Import from video" copies them to every upload.
ENDCARD_LAYOUT = {"more_y": 470, "more_max_h": 60,
                  "slot_w": 360, "slot_h": 203, "slot_y": 540, "slot1_x": 1108, "slot2_x": 1488,
                  "sub_cx": 1478, "sub_cy": 860, "sub_r": 72}


def endcard_layout(width=1920, height=1080):
    """ENDCARD_LAYOUT scaled to a canvas, plus the element rectangles as fractions of the
    frame (what YouTube Studio's end-screen editor shows while you drag elements)."""
    s = width / 1920
    L = {k: round(v * s) for k, v in ENDCARD_LAYOUT.items()}
    frac = lambda x, y, w, h: {"x": round(x / width, 4), "y": round(y / height, 4),
                               "w": round(w / width, 4), "h": round(h / height, 4)}
    L["elements"] = {
        "video_1": frac(L["slot1_x"], L["slot_y"], L["slot_w"], L["slot_h"]),
        "video_2": frac(L["slot2_x"], L["slot_y"], L["slot_w"], L["slot_h"]),
        "subscribe": frac(L["sub_cx"] - L["sub_r"], L["sub_cy"] - L["sub_r"], 2 * L["sub_r"], 2 * L["sub_r"]),
    }
    return L


def endcard_plate(out_path, band_text, band_font, band_treatment, more_text="",
                  width=1920, height=1080):
    """The static overlay for the end-card segment of the card-visualizer video: card
    border + band wordmark (as card_plate) + a 'More from ...' line + the two dark video
    panels + the subscribe circle. Transparent elsewhere, so the background loop and the
    living cover keep playing underneath."""
    L = {k: round(v * width / 1920) for k, v in CARD_LAYOUT.items()}
    E = endcard_layout(width, height)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([L["card_x"] - 4, L["card_y"] - 4,
                 L["card_x"] + L["card_w"] + 4, L["card_y"] + L["card_h"] + 4],
                outline=(210, 190, 160, 90), width=4)
    if str(band_text or "").strip():
        wm = render_wordmark(band_text, band_font, band_treatment, height=260)
        f = min(L["col_w"] / wm.width, L["band_max_h"] / wm.height, 1.0)
        wm = wm.resize((max(1, round(wm.width * f)), max(1, round(wm.height * f))), Image.LANCZOS)
        img.alpha_composite(wm, (L["col_x"] + (L["col_w"] - wm.width) // 2, L["band_y"]))
    if str(more_text or "").strip():
        # same family as the title lettering, bone treatment, small
        mt = render_wordmark(more_text, band_font, "bone", height=120)
        f = min(L["col_w"] / mt.width, E["more_max_h"] / mt.height, 1.0)
        mt = mt.resize((max(1, round(mt.width * f)), max(1, round(mt.height * f))), Image.LANCZOS)
        img.alpha_composite(mt, (E["slot1_x"], E["more_y"]))
    for x0 in (E["slot1_x"], E["slot2_x"]):
        d.rounded_rectangle([x0, E["slot_y"], x0 + E["slot_w"], E["slot_y"] + E["slot_h"]],
                            radius=round(10 * width / 1920), fill=(0, 0, 0, 170),
                            outline=(210, 190, 160, 110), width=3)
    cx, cy, r = E["sub_cx"], E["sub_cy"], E["sub_r"]
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 170),
              outline=(210, 190, 160, 110), width=3)
    img.save(out_path, format="PNG")
    return out_path


def card_plate(out_path, band_text, band_font, band_treatment,
               title_text, title_font, title_treatment, width=1920, height=1080):
    """The static overlay for the card-visualizer video: card border + band wordmark +
    song title on a transparent canvas. The card interior stays transparent (the living
    cover / still plays underneath); ffmpeg overlays this plate above it."""
    L = {k: round(v * width / 1920) for k, v in CARD_LAYOUT.items()}
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([L["card_x"] - 4, L["card_y"] - 4,
                 L["card_x"] + L["card_w"] + 4, L["card_y"] + L["card_h"] + 4],
                outline=(210, 190, 160, 90), width=4)

    def put(text, font_id, treatment, y, max_h):
        wm = render_wordmark(text, font_id, treatment, height=260)
        f = min(L["col_w"] / wm.width, max_h / wm.height, 1.0)
        wm = wm.resize((max(1, round(wm.width * f)), max(1, round(wm.height * f))), Image.LANCZOS)
        img.alpha_composite(wm, (L["col_x"] + (L["col_w"] - wm.width) // 2, y))
    if str(band_text or "").strip():
        put(band_text, band_font, band_treatment, L["band_y"], L["band_max_h"])
    if str(title_text or "").strip():
        put(title_text, title_font, title_treatment, L["title_y"], L["title_max_h"])
    img.save(out_path, format="PNG")
    return out_path
