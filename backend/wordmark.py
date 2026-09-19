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
    # modern sans (the "modern" look, 2026-09-19) - variable weight
    {"id": "josefin",     "label": "Josefin Sans",       "path": os.path.join(_REPO_FONTS, "JosefinSans.ttf")},
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
    # flat, no glow - the modern look's lettering
    "flat":   {"top": (236, 231, 220), "bottom": (236, 231, 220), "glow": (0, 0, 0),   "glow_alpha": 0,   "glow_blur": 0.01},
    "muted":  {"top": (150, 145, 136), "bottom": (150, 145, 136), "glow": (0, 0, 0),   "glow_alpha": 0,   "glow_blur": 0.01},
}

# The LOOK: a preset that the video plate, end card and thumbnail consult. "classic" =
# everything as it was (the saved fonts/treatments); "modern" (user-approved mocks
# 2026-09-19) = Josefin Sans, flat ivory, tracked caps, a short ember accent line, the
# cover art blurred as the video backdrop, 1px window border.
LOOKS = {
    "classic": {},
    "modern": {"font": "josefin", "band_weight": 300, "band_tracking": 0.60,
               "title_weight": 200, "title_tracking": 0.12, "title_upper": True,
               "treatment": "flat", "accent": (220, 120, 60), "ivory": (236, 231, 220),
               "backdrop": "blur", "border_alpha": 70, "sub_line": "OFFICIAL AUDIO"},
}


def look_of(cfg):
    return LOOKS.get(str((cfg or {}).get("look") or "classic")) or {}

# `title_*` = the SONG TITLE's stamped lettering (large, top band); the bare keys are the
# band wordmark. Both live in one saved config blob (app_config `yt_wordmark`).
DEFAULTS = {"look": "classic",
            "text": "Apotheon", "font": "luminari", "treatment": "bone",
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


def _set_weight(font, weight):
    """Pick a weight on a variable font (no-op on static faces)."""
    if weight is None:
        return font
    try:
        vals = []
        for a in font.get_variation_axes():
            nm = a.get("name")
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            vals.append(weight if "eight" in nm else a.get("default", a.get("minimum")))
        font.set_variation_by_axes(vals)
    except Exception:
        pass
    return font


def render_wordmark(text, font_id, treatment, height=240, weight=None, tracking=None):
    """The wordmark as a tight RGBA image (transparent background): gradient-filled
    letters over a soft glow. `height` is the letter size; the canvas adds glow padding.
    weight = variable-font weight (modern look); tracking = letter-spacing as a fraction
    of the size (default 0.06)."""
    spec = TREATMENTS.get(treatment) or TREATMENTS["bone"]
    font = _set_weight(ImageFont.truetype(_font_path(font_id), size=height), weight)
    mask = _render_mask(text, font, tracking=round(height * (0.06 if tracking is None else tracking)))
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


# YouTube end-screen landing zone for layout B. v6 (2026-09-16, after the user fought
# Studio's editor): the elements have MINIMUM sizes (video ~34% of the frame wide,
# subscribe ~16%), a bottom safe margin (~88% of the height) and side margins, and the
# editor cannot place them precisely - drawn target panels were a losing game. So the
# plate draws NO panels: the card window is dimmed, title + waveform are off, the
# wordmark and a "More from ..." line stay top-right, and the rest of the frame is a
# clean dark stage. YouTube's elements bring their own frames, so any Studio TEMPLATE
# (e.g. two videos + subscribe) lands looking designed, with no dragging.
ENDCARD_LAYOUT = {"more_y": 470, "more_max_h": 60, "window_dim": 165, "frame_dim": 60}


def endcard_layout(width=1920, height=1080):
    """ENDCARD_LAYOUT scaled to a canvas (kept for callers; no element rectangles any
    more - positions come from the Studio template)."""
    s = width / 1920
    return {k: (round(v * s) if k.endswith("_y") or k.endswith("_h") else v) for k, v in ENDCARD_LAYOUT.items()}


def endcard_plate(out_path, band_text, band_font, band_treatment, more_text="",
                  width=1920, height=1080, look=None):
    """The static overlay for the end-card segment of the card-visualizer video: a light
    darkening of the whole frame, a heavier dim over the card window (the loop keeps
    moving underneath, muted), the card border, the band wordmark and a 'More from ...'
    line. Transparent otherwise; no element panels (see ENDCARD_LAYOUT)."""
    L = {k: round(v * width / 1920) for k, v in CARD_LAYOUT.items()}
    E = endcard_layout(width, height)
    img = Image.new("RGBA", (width, height), (0, 0, 0, E["frame_dim"]))
    d = ImageDraw.Draw(img)
    d.rectangle([L["card_x"], L["card_y"], L["card_x"] + L["card_w"], L["card_y"] + L["card_h"]],
                fill=(0, 0, 0, E["window_dim"]))
    if look and look.get("backdrop"):
        d.rectangle([L["card_x"] - 1, L["card_y"] - 1, L["card_x"] + L["card_w"], L["card_y"] + L["card_h"]],
                    outline=look["ivory"] + (look["border_alpha"],), width=1)
        _modern_column(img, look, band_text, None, None, width, height, more_text=more_text)
        img.save(out_path, format="PNG")
        return out_path
    d.rectangle([L["card_x"] - 4, L["card_y"] - 4,
                 L["card_x"] + L["card_w"] + 4, L["card_y"] + L["card_h"] + 4],
                outline=(210, 190, 160, 70), width=4)
    if str(band_text or "").strip():
        wm = render_wordmark(band_text, band_font, band_treatment, height=260)
        f = min(L["col_w"] / wm.width, L["band_max_h"] / wm.height, 1.0)
        wm = wm.resize((max(1, round(wm.width * f)), max(1, round(wm.height * f))), Image.LANCZOS)
        img.alpha_composite(wm, (L["col_x"] + (L["col_w"] - wm.width) // 2, L["band_y"]))
    if str(more_text or "").strip():
        mt = render_wordmark(more_text, band_font, "bone", height=120)
        f = min(L["col_w"] / mt.width, E["more_max_h"] / mt.height, 1.0)
        mt = mt.resize((max(1, round(mt.width * f)), max(1, round(mt.height * f))), Image.LANCZOS)
        img.alpha_composite(mt, (L["col_x"] + round(30 * width / 1920), E["more_y"]))
    img.save(out_path, format="PNG")
    return out_path


def layout_for(look, width=1920):
    """CARD_LAYOUT scaled to the canvas; the modern look moves the waveform to a thin
    strip under the sub-line (user-approved mock 2026-09-19)."""
    L = dict(CARD_LAYOUT)
    if look.get("backdrop"):
        L.update({"wave_x": 1110, "wave_y": 660, "wave_w": 700, "wave_h": 48})
    return {k: round(v * width / 1920) for k, v in L.items()}


def blur_backdrop(src_png, out_path, width=1920, height=1080, dark=0.74, sat=0.5, blur=70):
    """The modern look's video backdrop: the cover art itself, blurred, desaturated,
    darkened and vignetted (each video gets its own palette instead of one shared loop)."""
    from PIL import ImageEnhance
    im = Image.open(src_png).convert("RGB").resize((width, height), Image.LANCZOS)
    im = im.filter(ImageFilter.GaussianBlur(round(blur * width / 1920)))
    im = ImageEnhance.Color(im).enhance(sat)
    im = Image.blend(im, Image.new("RGB", (width, height), (0, 0, 0)), dark)
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).ellipse([-width * 0.25, -height * 0.35, width * 1.25, height * 1.35], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(round(220 * width / 1920)))
    im = Image.composite(im, Image.blend(im, Image.new("RGB", (width, height), (0, 0, 0)), 0.55), m)
    im.save(out_path, format="PNG")
    return out_path


def _modern_column(img, look, band_text, title_text, sub_text, width, height, more_text=None):
    """The modern look's right column: tracked band name, hairline accent, big tracked
    caps title (or a 'More from' line on the end card), small sub-line. Positions are the
    approved mock's, scaled from 1920x1080."""
    s = width / 1920
    L = {k: round(v * s) for k, v in CARD_LAYOUT.items()}
    x = L["col_x"] + round(20 * s)
    ivory, accent = look["ivory"], look["accent"]
    if str(band_text or "").strip():
        wm = render_wordmark(band_text.upper(), look["font"], look["treatment"], height=round(34 * s),
                             weight=look["band_weight"], tracking=look["band_tracking"])
        img.alpha_composite(wm, (x, round(348 * s)))
    if title_text:
        t = title_text.upper() if look.get("title_upper") else title_text
        size = round(88 * s)
        while size > round(36 * s):
            wm = render_wordmark(t, look["font"], look["treatment"], height=size,
                                 weight=look["title_weight"], tracking=look["title_tracking"])
            if wm.width <= L["col_w"] - round(20 * s):
                break
            size -= round(4 * s)
        img.alpha_composite(wm, (x, round(430 * s)))
    elif more_text:
        wm = render_wordmark(more_text.upper(), look["font"], "muted", height=round(26 * s),
                             weight=300, tracking=0.35)
        img.alpha_composite(wm, (x, round(440 * s)))
    d = ImageDraw.Draw(img)
    d.line([(x + round(2 * s), round(560 * s)), (x + round(62 * s), round(560 * s))], fill=accent + (255,), width=max(1, round(2 * s)))
    if sub_text:
        wm = render_wordmark(sub_text, look["font"], "muted", height=round(18 * s), weight=300, tracking=0.5)
        img.alpha_composite(wm, (x + round(2 * s), round(588 * s)))
    return img


def card_plate(out_path, band_text, band_font, band_treatment,
               title_text, title_font, title_treatment, width=1920, height=1080, look=None):
    """The static overlay for the card-visualizer video: card border + band wordmark +
    song title on a transparent canvas. The card interior stays transparent (the living
    cover / still plays underneath); ffmpeg overlays this plate above it.
    look (wordmark.look_of(cfg)): the modern preset draws its own column + 1px border."""
    L = {k: round(v * width / 1920) for k, v in CARD_LAYOUT.items()}
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if look and look.get("backdrop"):
        d.rectangle([L["card_x"] - 1, L["card_y"] - 1, L["card_x"] + L["card_w"], L["card_y"] + L["card_h"]],
                    outline=look["ivory"] + (look["border_alpha"],), width=1)
        _modern_column(img, look, band_text, title_text, look.get("sub_line"), width, height)
        img.save(out_path, format="PNG")
        return out_path
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
