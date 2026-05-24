const $ = (id) => document.getElementById(id);
let MODE = "generate";
let VARIANTS = [];
let polling = null;

async function init() {
  const cfg = await (await fetch("/api/config")).json();
  $("host").textContent = "ComfyUI @ " + cfg.comfy_host;
  VARIANTS = cfg.variants;
  const vsel = $("variant");
  vsel.innerHTML = "";
  cfg.variants.forEach(v => {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = v.label + (v.available ? "" : " — not installed");
    o.disabled = !v.available;
    vsel.appendChild(o);
  });
  const firstAvail = cfg.variants.find(v => v.available);
  if (firstAvail) vsel.value = firstAvail.id;
  cfg.keys.forEach(k => {
    const o = document.createElement("option");
    o.value = k; o.textContent = k;
    if (k === "E minor") o.selected = true;
    $("keyscale").appendChild(o);
  });
  updateDefaults();
  loadVoices();
  loadLibrary();
}

async function loadVoices() {
  let v;
  try {
    v = await (await fetch("/api/rvc/voices")).json();
  } catch (e) {
    v = { available: false, voices: [] };
  }
  const status = (v.available ? (v.voices.length ? "" : "no voices installed") : "RVC unreachable");
  ["voiceStatus", "swapVoiceStatus"].forEach(id => { if ($(id)) $(id).textContent = status; });
  document.querySelectorAll(".voicesel").forEach(sel => {
    const cur = sel.value;
    sel.innerHTML = "";
    (v.voices || []).forEach(name => {
      const o = document.createElement("option");
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    });
    sel.value = cur;
  });
}

async function loadSwapSongs() {
  const items = await (await fetch("/api/library")).json();
  const sel = $("swapJob");
  const cur = sel.value;
  sel.innerHTML = "<option value=''>— choose —</option>";
  items.filter(it => (it.mode === "generate" || it.mode === "restyle") && !it.params.instrumental)
    .forEach(it => {
      const o = document.createElement("option");
      o.value = it.id;
      o.textContent = `${it.mode}: ${(it.params.tags || "").slice(0, 40)}`;
      sel.appendChild(o);
    });
  sel.value = cur;
}

function updateDefaults() {
  const v = VARIANTS.find(x => x.id === $("variant").value);
  if (!v) return;
  $("stepsDef").textContent = "(def " + v.steps + ")";
  $("cfgDef").textContent = "(def " + v.cfg + ")";
  $("steps").placeholder = v.steps;
  $("cfg").placeholder = v.cfg;
}

// tabs
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  MODE = t.dataset.mode;
  $("restyle-input").classList.toggle("hidden", MODE !== "restyle");
  $("gen-fields").classList.toggle("hidden", MODE === "vocals" || MODE === "stems");
  $("vocals-fields").classList.toggle("hidden", MODE !== "vocals");
  $("swap-fields").classList.toggle("hidden", MODE !== "swap");
  $("stems-fields").classList.toggle("hidden", MODE !== "stems");
  $("mix-fields").classList.toggle("hidden", MODE !== "mix");
  $("gen-fields").classList.toggle("hidden", ["vocals", "swap", "stems", "mix"].includes(MODE));
  $("go").textContent = { generate: "Generate", restyle: "Restyle", vocals: "Convert voice", swap: "Swap voice", stems: "Separate", mix: "Mix down" }[MODE];
  if (MODE === "stems") loadStemSources();
  if (MODE === "mix") loadMixSources();
  if (MODE === "swap") { loadVoices(); loadSwapSongs(); }
});

let MIX_SOURCES = [];
async function loadMixSources() {
  MIX_SOURCES = await (await fetch("/api/sources")).json();
  if (!$("mixRows").children.length) { addMixRow(); addMixRow(); }
  else document.querySelectorAll(".mixrow select").forEach(fillSourceSelect);
}
function fillSourceSelect(sel) {
  const cur = sel.value;
  sel.innerHTML = "<option value=''>— choose track —</option>";
  MIX_SOURCES.forEach(s => {
    const o = document.createElement("option");
    o.value = s.url; o.textContent = s.label;
    sel.appendChild(o);
  });
  sel.value = cur;
}
function addMixRow() {
  const row = document.createElement("div");
  row.className = "mixrow";
  const sel = document.createElement("select");
  fillSourceSelect(sel);
  row.appendChild(sel);
  row.insertAdjacentHTML("beforeend",
    `<span class="mixctl">gain <input type="number" class="gain" value="0" step="1" title="dB"> dB</span>`
    + `<span class="mixctl">offset <input type="number" class="offset" value="0" step="0.1" title="seconds"> s</span>`);
  const rm = document.createElement("button");
  rm.className = "ghost"; rm.textContent = "✕";
  rm.onclick = () => row.remove();
  row.appendChild(rm);
  $("mixRows").appendChild(row);
}
$("mixAdd").onclick = addMixRow;

async function loadStemSources() {
  const items = await (await fetch("/api/library")).json();
  const sel = $("stemJob");
  const cur = sel.value;
  sel.innerHTML = "<option value=''>— choose —</option>";
  items.filter(it => it.mode === "generate" || it.mode === "restyle").forEach(it => {
    const o = document.createElement("option");
    o.value = it.id;
    o.textContent = `${it.mode} · ${(it.params.tags || "").slice(0, 40)}`;
    sel.appendChild(o);
  });
  sel.value = cur;
}

// ---- voice search / install ----
const vstatus = (msg, cls) => { $("voiceAddStatus").textContent = msg; $("voiceAddStatus").className = "status" + (cls ? " " + cls : ""); };

$("voiceSearchBtn").onclick = async () => {
  const q = $("voiceSearch").value.trim();
  if (!q) return;
  $("voiceResults").innerHTML = "<span class='hint'>searching…</span>";
  try {
    const d = await (await fetch(`/api/voices/search?q=${encodeURIComponent(q)}&sort=${$("voiceSort").value}`)).json();
    if (!d.results || !d.results.length) { $("voiceResults").innerHTML = "<span class='hint'>no results</span>"; return; }
    $("voiceResults").innerHTML = "";
    d.results.forEach(r => {
      const row = document.createElement("div");
      row.className = "vresult";
      row.innerHTML = `<span class="rid">${r.id} <span class="hint">♥${r.likes} ⤓${r.downloads}</span></span>`;
      const btn = document.createElement("button");
      btn.className = "ghost"; btn.textContent = "files";
      btn.onclick = () => listRepo(r.id, row);
      row.appendChild(btn);
      $("voiceResults").appendChild(row);
    });
  } catch (e) { $("voiceResults").innerHTML = "<span class='err'>search failed</span>"; }
};

async function listRepo(id, afterEl) {
  let box = afterEl.nextElementSibling;
  if (box && box.classList.contains("vfiles")) { box.remove(); return; }
  box = document.createElement("div");
  box.className = "vfiles";
  box.innerHTML = "<span class='hint'>loading…</span>";
  afterEl.after(box);
  try {
    const d = await (await fetch(`/api/voices/repo?id=${encodeURIComponent(id)}`)).json();
    if (!d.voices.length) { box.innerHTML = "<span class='hint'>no .pth models found</span>"; return; }
    box.innerHTML = "";
    d.voices.forEach(v => {
      const f = document.createElement("div");
      f.className = "vfile";
      const tag = v.zip ? " <span class='hint'>(zip)</span>" : (v.index ? " <span class='hint'>+index</span>" : "");
      f.innerHTML = `<span>${v.name}${tag}</span>`;
      const b = document.createElement("button");
      b.className = "ghost"; b.textContent = "install";
      const body = v.zip ? { repo: id, zip: v.zip, name: v.name }
                         : { repo: id, pth: v.pth, index: v.index, name: v.name };
      b.onclick = () => installVoice(body, b);
      f.appendChild(b);
      box.appendChild(f);
    });
  } catch (e) { box.innerHTML = "<span class='err'>failed to list files</span>"; }
}

async function installVoice(body, btn) {
  if (btn) { btn.disabled = true; btn.textContent = "installing…"; }
  vstatus("installing " + (body.name || body.url) + " to the PC… (downloads can be large)");
  try {
    const res = await fetch("/api/voices/install", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "install failed");
    vstatus("✓ installed " + d.name, "ok");
    loadVoices();
  } catch (e) {
    vstatus("✗ " + e.message, "err");
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "install"; }
  }
}

function renderStems(stems) {
  const box = $("stemResults");
  box.innerHTML = "<h3 class='tune'>Stems</h3>";
  stems.forEach(s => {
    const d = document.createElement("div");
    d.className = "libitem";
    d.innerHTML = `<div class="meta"><span class="tag">${s.name}</span>`
      + ` <a href="${s.url}" download class="hint">download</a></div>`
      + `<audio controls src="${s.url}"></audio>`;
    box.appendChild(d);
  });
}

$("voiceUrlBtn").onclick = () => {
  const url = $("voiceUrl").value.trim();
  if (!url) return;
  installVoice({ url, name: $("voiceUrlName").value.trim() || undefined }, $("voiceUrlBtn"));
};

$("variant").onchange = updateDefaults;
$("instrumental").onchange = (e) => $("lyrics-field").classList.toggle("hidden", e.target.checked);
$("restyle_amount").oninput = (e) => $("restyleOut").textContent = parseFloat(e.target.value).toFixed(2);
$("index_rate").oninput = (e) => $("irOut").textContent = parseFloat(e.target.value).toFixed(2);
$("rms_mix_rate").oninput = (e) => $("rmsOut").textContent = parseFloat(e.target.value).toFixed(2);
$("protect").oninput = (e) => $("protOut").textContent = parseFloat(e.target.value).toFixed(2);

function gatherParams() {
  const p = {
    variant: $("variant").value,
    tags: $("tags").value.trim(),
    instrumental: $("instrumental").checked,
    lyrics: $("lyrics").value,
    duration: parseFloat($("duration").value) || 40,
    bpm: parseInt($("bpm").value) || 120,
    keyscale: $("keyscale").value,
    timesignature: $("timesignature").value,
  };
  if ($("steps").value) p.steps = parseInt($("steps").value);
  if ($("cfg").value) p.cfg = parseFloat($("cfg").value);
  if ($("seed").value) p.seed = parseInt($("seed").value);
  if (MODE === "restyle") p.restyle_amount = parseFloat($("restyle_amount").value);
  return p;
}

$("go").onclick = async () => {
  const p = gatherParams();
  $("go").disabled = true;
  $("cancel").disabled = false;
  $("progress").classList.remove("hidden");
  $("now").classList.add("hidden");
  setStatus("submitting…", 0, 0);
  try {
    let res;
    if ((MODE === "generate" || MODE === "restyle") && !p.tags) {
      throw new Error("Add style tags first (e.g. \"symphonic power metal, distorted guitars, double-bass drums\") — an empty prompt produces noise.");
    }
    if (MODE === "swap") {
      const f = $("swapsrc").files[0];
      const jid = $("swapJob").value;
      if (!f && !jid) throw new Error("choose a vocal song (library or upload)");
      if (!$("swapVoice").value) throw new Error("no target voice available (is RVC running?)");
      const fd = new FormData();
      fd.append("voice", $("swapVoice").value);
      fd.append("transpose", parseInt($("swapTranspose").value) || 0);
      fd.append("vocal_gain", parseFloat($("swapVocalGain").value) || 0);
      fd.append("instr_gain", parseFloat($("swapInstrGain").value) || 0);
      if (f) fd.append("file", f); else fd.append("job_id", jid);
      setStatus("split → re-timbre → remix… (this takes ~20–40s)", 0, 0);
      const res = await fetch("/api/voiceswap", { method: "POST", body: fd });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || "voice swap failed");
      setStatus("✓ done", 1, 1);
      $("player").src = d.audio_url + "?t=" + Date.now();
      $("now").classList.remove("hidden");
      finish();
      loadLibrary();
      return;
    }
    if (MODE === "mix") {
      const tracks = [...document.querySelectorAll(".mixrow")].map(r => ({
        src: r.querySelector("select").value,
        gain_db: parseFloat(r.querySelector(".gain").value) || 0,
        offset: parseFloat(r.querySelector(".offset").value) || 0,
      })).filter(t => t.src);
      if (tracks.length < 1) throw new Error("choose at least one track to mix");
      setStatus("mixing down…", 0, 0);
      const res = await fetch("/api/mix", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks, normalize: $("mixNorm").checked })
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || "mix failed");
      setStatus("✓ mixed", 1, 1);
      $("player").src = d.audio_url + "?t=" + Date.now();
      $("now").classList.remove("hidden");
      finish();
      loadLibrary();
      return;
    }
    if (MODE === "stems") {
      const f = $("stemsrc").files[0];
      const jid = $("stemJob").value;
      if (!f && !jid) throw new Error("choose a library track or upload one to separate");
      const fd = new FormData();
      fd.append("mode", $("stemMode").value);
      if (f) fd.append("file", f); else fd.append("job_id", jid);
      setStatus("separating stems on the Mac GPU… (can take a while for long tracks)", 0, 0);
      const res = await fetch("/api/stems/separate", { method: "POST", body: fd });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail || "separation failed");
      renderStems(d.stems);
      setStatus("✓ separated", 1, 1);
      finish();
      return;
    }
    if (MODE === "vocals") {
      const f = $("vocalsrc").files[0];
      if (!f) throw new Error("choose a guide vocal first");
      if (!$("voice").value) throw new Error("no target voice available");
      const p = {
        voice: $("voice").value,
        transpose: parseInt($("transpose").value) || 0,
        f0_method: $("f0_method").value,
        index_rate: parseFloat($("index_rate").value),
        rms_mix_rate: parseFloat($("rms_mix_rate").value),
        protect: parseFloat($("protect").value),
      };
      const fd = new FormData();
      fd.append("file", f);
      fd.append("params", JSON.stringify(p));
      setStatus("converting voice… (synchronous, please wait)", 0, 0);
      res = await fetch("/api/rvc/convert", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "convert failed");
      setStatus("✓ done", 1, 1);
      $("player").src = data.audio_url + "?t=" + Date.now();
      $("now").classList.remove("hidden");
      finish();
      loadLibrary();
      return;
    }
    if (MODE === "restyle") {
      const f = $("source").files[0];
      if (!f) throw new Error("choose a source track first");
      const fd = new FormData();
      fd.append("file", f);
      fd.append("params", JSON.stringify(p));
      res = await fetch("/api/restyle", { method: "POST", body: fd });
    } else {
      res = await fetch("/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(p)
      });
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "submit failed");
    setStatus("seed " + data.seed + " · queued…", 0, 0);
    pollJob(data.job_id);
  } catch (e) {
    setStatus("✗ " + e.message, 0, 0, true);
    finish();
  }
};

$("cancel").onclick = () => fetch("/api/cancel", { method: "POST" });

function pollJob(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    const j = await (await fetch("/api/job/" + id)).json();
    if (j.status === "running" || j.status === "finalizing") {
      const pct = j.max ? Math.round(100 * j.progress / j.max) : 0;
      setStatus(j.status + (j.max ? ` · ${j.progress}/${j.max}` : "…"), j.progress, j.max);
    } else if (j.status === "pending") {
      setStatus("queued…", 0, 0);
    } else if (j.status === "done" && j.audio_url) {
      setStatus("✓ done", 1, 1);
      $("player").src = j.audio_url + "?t=" + Date.now();
      $("now").classList.remove("hidden");
      $("nowMeta").textContent = "";
      clearInterval(polling);
      finish();
      loadLibrary();
    } else if (j.status === "error") {
      setStatus("✗ " + (j.error || "error"), 0, 0, true);
      clearInterval(polling);
      finish();
    }
  }, 1000);
}

function setStatus(msg, val, max, err) {
  $("status").textContent = msg;
  $("status").className = "status" + (err ? " err" : "");
  $("barfill").style.width = (max ? 100 * val / max : (err ? 0 : 5)) + "%";
}

function finish() {
  $("go").disabled = false;
  $("cancel").disabled = true;
}

async function loadLibrary() {
  const items = await (await fetch("/api/library")).json();
  const lib = $("lib");
  lib.innerHTML = items.length ? "" : "<p class='hint'>No tracks yet.</p>";
  items.forEach(it => {
    const d = document.createElement("div");
    d.className = "libitem";
    const p = it.params || {};
    const when = new Date(it.created * 1000).toLocaleString();
    const tags = (p.tags || "").slice(0, 70);
    d.innerHTML =
      `<div class="meta"><span class="tag">${it.mode}</span>`
      + `<span class="tag">${p.variant || ""}</span>`
      + `${p.bpm || "?"}bpm · ${p.keyscale || ""} · seed ${p.seed || "?"}<br>${tags}`
      + `<br><span class="hint">${when}</span></div>`
      + `<audio controls src="${it.audio_url}"></audio>`;
    lib.appendChild(d);
  });
}

init();
