"""YouTube thumbnail: the "panel" layout chosen 2026-09-06 (Vanadis round two, D2).

The cover's face is cropped large on one side, the other side falls into a dark panel
carrying the song title, the band wordmark and an optional genre cue. Everything is
Pillow on the Mac; the source is any library still (or the first frame of a living
cover, extracted by the caller). Fonts/treatments come from the saved wordmark config
so the thumbnail matches the video lettering.

Why this shape (measured on the feed-scale sheet): the card-video frame shrinks the
face to a thumb of a thumb; a half-frame face + three big words is what survives at
336 px on desktop and 168 px on a phone. The grade crushes contrast and split-tones
(cool shadows / warm highs) so the image does not vanish on YouTube's dark UI.
"""
import os
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from . import wordmark as wordmark_mod

W, H = 1280, 720          # YouTube's recommended thumbnail size (16:9)


def _crop_at(img, fx, fy, zoom, anchor):
    """16:9 window of width zoom*source placed so the face point (fractions) sits at
    `anchor` (fractions of the window), clamped inside the source, resized to 1280x720."""
    cw = max(64, int(img.width * zoom))
    ch = int(cw * 9 / 16)
    if ch > img.height:
        ch = img.height
        cw = int(ch * 16 / 9)
    cx, cy = fx * img.width, fy * img.height
    x0 = int(cx - cw * anchor[0]); y0 = int(cy - ch * anchor[1])
    x0 = max(0, min(img.width - cw, x0)); y0 = max(0, min(img.height - ch, y0))
    out = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)
    # the crop is an upscale on every source we have (720p clips to 2848-wide stills);
    # a light unsharp keeps the eyes crisp at feed size
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2)), (cx - x0) * W / cw, (cy - y0) * H / ch


def _radial(size, cx, cy, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    return m.filter(ImageFilter.GaussianBlur(r * 0.55))


def _grade(img, strength=1.0, cool=(150, 175, 235), warm=(255, 205, 150)):
    """Punchy split-tone: crushed contrast, cool shadows, warm highlights. `strength`
    0 = untouched, 1 = the chosen look."""
    if strength <= 0:
        return img
    im = ImageEnhance.Contrast(img).enhance(1 + 0.30 * strength)
    im = ImageEnhance.Color(im).enhance(1 + 0.25 * strength)
    lum = im.convert("L")
    shadows = lum.point(lambda v: int(max(0, 140 - v) * 255 / 140 * 0.45 * strength))
    im = Image.composite(ImageChops.multiply(im, Image.new("RGB", im.size, cool)), im, shadows)
    highs = lum.point(lambda v: int(max(0, v - 150) * 255 / 105 * 0.35 * strength))
    warm_layer = ImageChops.screen(im, Image.new("RGB", im.size, tuple(int(c * 0.35) for c in warm)))
    return Image.composite(warm_layer, im, highs)


def _lift_face(img, cx, cy, r=300, amount=1.30):
    bright = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(amount)).enhance(1.10)
    return Image.composite(bright, img, _radial(img.size, cx, cy, r))


def _vignette(img, strength=0.35):
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).ellipse((-img.width * 0.1, -img.height * 0.2, img.width * 1.1, img.height * 1.2), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(img.width * 0.12)).point(lambda v: int(255 - (255 - v) * strength))
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), m)


def _panel(img, x0, x1, color=(6, 5, 9), soft=90, opacity=0.9):
    p = Image.new("RGB", img.size, color)
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rectangle((x0, 0, x1, H), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(soft)).point(lambda v: int(v * opacity))
    return Image.composite(p, img, m)


def _put_centered(img, text, font, treatment, cx, y, height, max_w):
    """Render a wordmark at `height`, shrinking until it fits `max_w`, centred on cx."""
    wm = wordmark_mod.render_wordmark(text, font, treatment, height=height)
    while wm.width > max_w and height > 24:
        height = int(height * 0.9)
        wm = wordmark_mod.render_wordmark(text, font, treatment, height=height)
    img.paste(wm, (int(cx - wm.width / 2), int(y)), wm)
    return wm.height


def video_frame(video_path, out_png):
    """First frame of a clip as PNG (for living covers that never had a still)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_path, "-frames:v", "1", out_png],
                   check=True)
    return out_png


def panel_thumbnail(src_path, out_path, title, band, cue="", face=(0.5, 0.35), zoom=0.5,
                    side="right", grade=1.0, lift=1.3, wm_cfg=None):
    """Write the panel thumbnail. `face` = (x, y) fractions of the source where the face
    is; `zoom` = crop width as a fraction of the source (smaller = bigger face);
    `side` = which side the text panel takes ("right" puts the face on the left).
    Fonts: title -> wm_cfg title_font/title_treatment, band -> font/treatment."""
    cfg = {**wordmark_mod.DEFAULTS, **(wm_cfg or {})}
    src = Image.open(src_path).convert("RGB")
    zoom = min(1.0, max(0.2, float(zoom)))
    panel_right = side != "left"
    anchor = (0.30, 0.40) if panel_right else (0.70, 0.40)
    img, fcx, fcy = _crop_at(src, face[0], face[1], zoom, anchor)
    img = _lift_face(img, fcx, fcy, r=300, amount=lift)
    img = _grade(img, grade)
    if panel_right:
        img = _panel(img, 720, W)
        tx, max_w = 1000, 540
    else:
        img = _panel(img, 0, 560)
        tx, max_w = 280, 540
    img = _vignette(img, 0.35)
    look = wordmark_mod.look_of(cfg)
    if look.get("backdrop"):
        # modern look (user-approved mock 2026-09-19): flat tracked caps fitted inside the
        # panel's text column - title small on top, band as the big element, accent
        # hairline, cue at the bottom; right-aligned on a right panel, left on a left one
        x0, x1 = (770, 1235) if panel_right else (45, 510)
        col = x1 - x0

        def fitted(text, size, weight, tracking, treatment, floor=14):
            while size > floor:
                wm = wordmark_mod.render_wordmark(text, look["font"], treatment, height=size,
                                                  weight=weight, tracking=tracking)
                if wm.width <= col:
                    return wm
                size -= 2
            return wm

        def put(wm, y):
            img.paste(wm, (x1 - wm.width if panel_right else x0, y), wm)
        if title:
            put(fitted(title.upper(), 40, 300, 0.14, look["treatment"]), 160)
        if band:
            put(fitted(band.upper(), 72, 300, 0.28, look["treatment"]), 235)
        d = ImageDraw.Draw(img)
        lx = (x1 - 70, x1) if panel_right else (x0, x0 + 70)
        d.line([(lx[0], 340), (lx[1], 340)], fill=look["accent"], width=2)
        if cue:
            put(fitted(cue.upper(), 24, 300, 0.45, "muted"), 520)
        img.save(out_path)
        return out_path
    y = 130
    if title:
        h = _put_centered(img, title, cfg["title_font"], cfg["title_treatment"], tx, y, 150, max_w)
        y += h + 20
    else:
        y = 200
    if band:
        h = _put_centered(img, band, cfg["font"], cfg["treatment"], tx, y, 92, max_w)
        y += h + 40
    if cue:
        _put_centered(img, cue, "cinzel", "silver", tx, max(y, 500), 52, max_w)
    img.save(out_path)
    return out_path
