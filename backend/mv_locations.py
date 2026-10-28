"""Reusable location library for the music-video pipeline (docs/MV_H3_REBUILD_PLAN.md s.19).

A location is one physical place at one camera distance class, with a person-free still that
the chain hands to the video model as the reference picture of the place. The library keeps
every location that has been written and photographed (id, name, description, still prompt,
library still id) so that:
  - the writer REUSES known places across re-runs of the same song and across songs, inventing
    a new location only where no known one fits (before this, every run invented fresh names and
    orphaned the stills already generated);
  - the render route finds a scene's still from its location id, with no per-request map.

Store: .mvwork/mv_locations/locations.json, a JSON list of rows:
  {"id": slug, "name": "the parlour", "description": one sentence of what the still shows,
   "still_prompt": the prompt the still was made from, "still_id": library still or None,
   "tags": [..], "source": {"title": song, "script": file}, "created": epoch, "updated": epoch}
Nothing here touches the GPU.
"""
import json
import os
import re
import time

FIELDS = ("id", "name", "description", "still_prompt", "still_id", "tags", "source", "created", "updated")


def slug(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x or "").lower()).strip("_")


def lib_dir(library_dir):
    d = os.path.join(os.path.dirname(os.path.abspath(library_dir)), ".mvwork", "mv_locations")
    os.makedirs(d, exist_ok=True)
    return d


def store_path(library_dir):
    return os.path.join(lib_dir(library_dir), "locations.json")


def load(library_dir):
    """All rows, oldest first. Missing or unreadable store = empty library."""
    p = store_path(library_dir)
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            rows = json.load(f)
    except Exception:
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("id")]


def save(library_dir, rows):
    p = store_path(library_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, p)
    return p


def get(rows, lid):
    lid = slug(lid)
    for r in rows:
        if r["id"] == lid:
            return r
    return None


def with_stills(rows):
    """Rows the writer may offer: a still exists, so the place can be referenced."""
    return [r for r in rows if r.get("still_id")]


def still_for(rows, lid):
    r = get(rows, lid)
    return (r or {}).get("still_id") or None


def upsert(library_dir, row):
    """Add or merge one row (id required; given fields overwrite, missing fields are kept).
    Returns the stored row."""
    lid = slug(row.get("id") or row.get("name"))
    if not lid:
        raise ValueError("a location needs an id or a name")
    rows = load(library_dir)
    now = time.time()
    cur = get(rows, lid)
    new = {k: v for k, v in row.items() if k in FIELDS and v is not None}
    new["id"] = lid
    if cur is None:
        cur = {"id": lid, "name": lid.replace("_", " "), "description": "", "still_prompt": "",
               "still_id": None, "tags": [], "source": {}, "created": now}
        rows.append(cur)
    cur.update(new)
    cur["name"] = (cur.get("name") or lid.replace("_", " ")).strip()
    cur["tags"] = sorted({slug(t) for t in (cur.get("tags") or []) if slug(t)})
    cur["updated"] = now
    save(library_dir, rows)
    return cur


def remove(library_dir, lid):
    rows = load(library_dir)
    keep = [r for r in rows if r["id"] != slug(lid)]
    if len(keep) == len(rows):
        return False
    save(library_dir, keep)
    return True


def import_from_script(library_dir, script, location_id, still_id=None, script_name=None, tags=None):
    """File a script's location row in the library, with the still that was generated from its
    prompt. The script's description and still prompt are kept verbatim so later scenes can be
    written to match the photograph."""
    lid = slug(location_id)
    src = None
    for L in script.get("locations") or []:
        if L.get("id") == lid:
            src = L
            break
    if src is None:
        raise ValueError(f"the script has no location '{lid}'")
    row = {"id": lid, "name": src.get("name"), "description": src.get("description"),
           "still_prompt": src.get("still_prompt"),
           "source": {"title": script.get("title"), "script": script_name or ""}}
    if still_id:
        row["still_id"] = os.path.basename(str(still_id))
    if tags:
        row["tags"] = list(tags)
    return upsert(library_dir, row)


def known_text(rows):
    """The block the writer shows the model: one line per known place, id first."""
    lines = []
    for r in rows:
        lines.append(f"- id `{r['id']}` ({r.get('name')}): {r.get('description') or '(no description)'}")
    return "\n".join(lines)


def same_song(rows, title):
    """Rows whose source song is this one (case-insensitive title match)."""
    t = (title or "").strip().lower()
    return [r for r in rows if t and ((r.get("source") or {}).get("title") or "").strip().lower() == t]


def select(rows, want, title=None):
    """Which rows the writer is offered.
    None/"auto" [default] = the rows with a still that came from THIS song (re-runs keep their
      stills; nothing from other songs);
    "library" = every row with a still (cross-song reuse, explicit opt-in: MEASURED 2026-10-28 that
      offering another video's places pulls them into a song they do not fit);
    "none" = nothing; a list of ids = those rows (with or without stills)."""
    if want in (None, "", "auto"):
        return with_stills(same_song(rows, title))
    if want in ("library", True):
        return with_stills(rows)
    if want in ("none", False):
        return []
    if isinstance(want, (list, tuple)):
        ids = {slug(w) for w in want}
        return [r for r in rows if r["id"] in ids]
    raise ValueError("locations must be 'auto', 'library', 'none' or a list of ids")
