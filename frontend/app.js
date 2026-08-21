// Farm Pal kiosk UI.
//
// Talks to the backend:
//   - text question         -> POST /chat      (JSON)
//   - photo (+optional text) -> POST /classify  (multipart)
// Farmer profiles and crop history are kept on this device in
// localStorage (offline). Bilingual English / Kiswahili.

(() => {
  "use strict";

  const I18N = {
    en: {
      thinking: "Thinking…", analyzing: "Analyzing the photo…",
      errGeneric: "Something went wrong. Please try again.",
      errNetwork: "Cannot reach the service. Check that the backend is running.",
      errEmpty: "Please type a question first.",
      needPhoto: "Please add a photo first.",
      llmDown: "The assistant is temporarily unavailable. Here is the reference material found:",
      invalidImage: "That file couldn't be read as an image. Please try another photo.",
      classifierOff: "Photo analysis isn't available on this device yet. Please type your question instead.",
      detected: "Detected", alsoPossible: "Also possible", healthy: "looks healthy",
      nextAction: "Next action", saveProfile: "Save to farmer profile", saved: "Saved to profile",
      noCrops: "No crop records yet.", noFarmers: "No farmers yet. Add one to start.",
      badge: { low: "Low", medium: "Medium", high: "High" },
      cropsTracked: "Crop records", farmers: "Farmers", cases: "Active cases", counties: "Counties",
    },
    sw: {
      thinking: "Inafikiri…", analyzing: "Inachambua picha…",
      errGeneric: "Hitilafu imetokea. Tafadhali jaribu tena.",
      errNetwork: "Imeshindwa kufikia huduma. Hakikisha seva inaendeshwa.",
      errEmpty: "Tafadhali andika swali kwanza.",
      needPhoto: "Tafadhali ongeza picha kwanza.",
      llmDown: "Msaidizi hapatikani kwa sasa. Hapa kuna marejeo yaliyopatikana:",
      invalidImage: "Faili hilo halikusomeka kama picha. Jaribu picha nyingine.",
      classifierOff: "Uchambuzi wa picha haupatikani bado. Tafadhali andika swali.",
      detected: "Imegundulika", alsoPossible: "Yawezekana pia", healthy: "unaonekana mzima",
      nextAction: "Hatua inayofuata", saveProfile: "Hifadhi kwa mkulima", saved: "Imehifadhiwa",
      noCrops: "Hakuna rekodi za mazao bado.", noFarmers: "Hakuna wakulima bado. Ongeza mmoja.",
      badge: { low: "Mdogo", medium: "Wastani", high: "Mkubwa" },
      cropsTracked: "Rekodi za mazao", farmers: "Wakulima", cases: "Kesi hai", counties: "Kaunti",
    },
  };
  let lang = "en";
  const t = () => I18N[lang];
  const sessionId = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
  const $ = (id) => document.getElementById(id);

  // ---------- Persistence (localStorage) ----------
  const KEY = "farmpal.v1";
  const PALETTE = ["#254A34", "#2F5C41", "#6E4A2C", "#8a5f14", "#3B6D11", "#7a3f18"];

  function uid() { return (crypto.randomUUID && crypto.randomUUID()) || "id" + Math.random().toString(36).slice(2); }
  function today() {
    const d = new Date(), m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return d.getDate() + " " + m[d.getMonth()] + " " + d.getFullYear();
  }
  function initials(n) { return (n || "?").trim().split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase(); }
  function colorFor(n) { let h = 0; for (const c of (n || "")) h = (h * 31 + c.charCodeAt(0)) >>> 0; return PALETTE[h % PALETTE.length]; }
  function diseaseClass(d) {
    d = (d || "").toLowerCase();
    if (!d || d.includes("healthy")) return "b-healthy";
    if (d.includes("angular")) return "b-als";
    if (d.includes("rust")) return "b-rust";
    if (d.includes("borer") || d.includes("stalk")) return "b-borer";
    if (d.includes("gray") || d.includes("grey") || d.includes("blight") || d.includes("spot")) return "b-gray";
    return "b-stage";
  }
  function isHealthy(d) { return !d || String(d).toLowerCase().includes("healthy"); }

  function seed() {
    const mk = (name, county, crops) => ({ id: uid(), name, county, crops: crops.map((c) => Object.assign({ id: uid() }, c)) });
    return { farmers: [
      mk("Wanjiru Kamau", "Nakuru", [
        { crop: "Beans", stage: "Flowering", disease: "Bean Rust", date: "12 Aug 2026", next: "Remove infected lower leaves and keep 45–50 cm row spacing for airflow. Scout at flowering; spray copper or mancozeb only if pustules spread." },
        { crop: "Maize", stage: "Whorl", disease: "Healthy", date: "3 Aug 2026", next: "No disease seen. Keep watching the whorl for stem borer windowpane holes." },
      ]),
      mk("Otieno Ochieng", "Kisumu", [
        { crop: "Maize", stage: "Whorl", disease: "Stem Borer", date: "15 Aug 2026", next: "Destroy old crop residues and try push-pull with Napier and Desmodium. A pinch of dry soil or ash in the funnel helps young plants without costly sprays." },
      ]),
      mk("Njeri Mwangi", "Kiambu", [
        { crop: "Beans", stage: "Pod fill", disease: "Angular Leaf Spot", date: "9 Aug 2026", next: "Use certified clean seed next season and avoid walking through the crop when wet. Rotate with maize for two seasons." },
      ]),
      mk("Amina Hassan", "Machakos", [
        { crop: "Maize", stage: "Germinating", disease: "Healthy", date: "18 Aug 2026", next: "Seedlings look healthy. Plant early and thin to recommended spacing to reduce pest pressure." },
      ]),
    ] };
  }
  function load() { try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && Array.isArray(s.farmers)) return s; } catch (e) {} const s = seed(); save(s); return s; }
  function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} }
  let state = load();
  const farmerById = (id) => state.farmers.filter((f) => f.id === id)[0];

  function esc(s) { return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }

  // ---------- i18n ----------
  function applyLanguage(next) {
    lang = next;
    document.documentElement.lang = next;
    document.querySelectorAll(".lang-btn").forEach((b) => b.classList.toggle("is-active", b.dataset.lang === next));
    document.querySelectorAll("[data-en]").forEach((el) => { const v = el.getAttribute("data-" + next); if (v !== null) el.textContent = v; });
    document.querySelectorAll("[data-ph-en]").forEach((el) => { const v = el.getAttribute("data-ph-" + next); if (v !== null) el.placeholder = v; });
  }
  document.querySelectorAll(".lang-btn").forEach((b) => b.addEventListener("click", () => applyLanguage(b.dataset.lang)));

  // ---------- Router ----------
  const titles = { home: ["Home","Nyumbani"], ask: ["Ask","Uliza"], diagnose: ["Diagnose","Tambua"], farmers: ["Farmers","Wakulima"], profile: ["Farmer profile","Mkulima"], history: ["Crop history","Historia"], sources: ["Sources","Marejeo"], settings: ["Settings","Mipangilio"] };
  function go(page) {
    document.querySelectorAll(".page").forEach((p) => { p.hidden = p.dataset.p !== page; });
    document.querySelectorAll("#nav a, .side-foot .nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    const c = $("crumb"); if (titles[page]) c.textContent = titles[page][lang === "sw" ? 1 : 0];
    const el = document.querySelector('.page[data-p="' + page + '"]');
    if (el) { el.style.animation = "none"; void el.offsetHeight; el.style.animation = ""; }
    if (page === "home") renderHome();
    if (page === "farmers") renderFarmers();
    if (page === "history") renderHistory();
    document.querySelector(".main").scrollTo({ top: 0, behavior: "smooth" });
  }
  document.querySelectorAll("[data-page]").forEach((a) => a.addEventListener("click", () => go(a.dataset.page)));
  document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));

  // ---------- Home ----------
  function allRecords() { const r = []; state.farmers.forEach((f) => f.crops.forEach((c) => r.push({ f, c }))); return r; }
  function animateCounts() {
    document.querySelectorAll("#stats .n").forEach((el) => {
      const target = +el.dataset.count || 0; const t0 = performance.now();
      (function step(now) { const p = Math.min(1, (now - t0) / 800); el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))); if (p < 1) requestAnimationFrame(step); })(performance.now());
    });
  }
  function renderHome() {
    const recs = allRecords();
    const cases = recs.filter((x) => !isHealthy(x.c.disease)).length;
    const counties = new Set(state.farmers.map((f) => f.county)).size;
    const stats = [
      [t().farmers, state.farmers.length, "ti"],
      [t().cropsTracked, recs.length, "ti"],
      [t().cases, cases, "ti"],
      [t().counties, counties, "ti"],
    ];
    $("stats").innerHTML = stats.map((s) =>
      '<div class="stat"><div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2 2 7l10 5 10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/></svg></div><div class="k">' + esc(s[0]) + '</div><div class="n" data-count="' + s[1] + '">0</div></div>').join("");
    const recent = recs.slice().reverse().slice(0, 5);
    $("feed").innerHTML = recent.length ? recent.map((x) =>
      '<div class="row"><div class="av" style="background:' + colorFor(x.f.name) + '">' + esc(initials(x.f.name)) + '</div><div><b>' + esc(x.f.name) + '</b><br><span>' + esc(x.c.crop + " · " + x.c.disease) + '</span></div><span class="when">' + esc(x.c.date) + '</span></div>').join("") : '<div class="empty">' + esc(t().noCrops) + "</div>";
    const flags = recs.filter((x) => !isHealthy(x.c.disease));
    $("attn").innerHTML = flags.length ? flags.map((x) =>
      '<div class="row"><div class="av" style="background:' + colorFor(x.f.name) + '">' + esc(initials(x.f.name)) + '</div><div><b>' + esc(x.c.crop + " · " + x.f.name.split(" ")[0]) + '</b><br><span>' + esc(x.f.county) + '</span></div><span class="badge ' + diseaseClass(x.c.disease) + '">' + esc(x.c.disease) + "</span></div>").join("") : '<div class="empty">All clear.</div>';
    animateCounts();
  }

  // ---------- Farmers ----------
  function renderFarmers() {
    const g = $("farmgrid");
    if (!state.farmers.length) { g.innerHTML = '<div class="empty">' + esc(t().noFarmers) + "</div>"; return; }
    g.innerHTML = state.farmers.map((f) => {
      const alerts = f.crops.filter((c) => !isHealthy(c.disease)).length;
      const chips = f.crops.slice(0, 3).map((c) => '<span class="badge ' + diseaseClass(c.disease) + '">' + esc(c.crop + " · " + c.disease) + "</span>").join("");
      return '<div class="farm" data-f="' + f.id + '"><div class="h"><div class="av" style="background:' + colorFor(f.name) + '">' + esc(initials(f.name)) + '</div><div><b>' + esc(f.name) + '</b><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>' + esc(f.county) + ' County</span></div></div><div class="cropline">' + chips + '</div><div class="foot"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 2 7l10 5 10-5z"/></svg>' + f.crops.length + ' crops' + (alerts ? '<span class="alert"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>' + alerts + " to act on</span>" : "") + "</div></div>";
    }).join("");
    g.querySelectorAll(".farm").forEach((el) => el.addEventListener("click", () => openProfile(el.dataset.f)));
  }

  function entryHTML(c) {
    const warn = !isHealthy(c.disease);
    return '<div class="entry' + (warn ? " warn" : "") + '"><div class="eh"><b>' + esc(c.crop) + '</b><span class="badge b-stage">' + esc(c.stage) + '</span><span class="badge ' + diseaseClass(c.disease) + '">' + esc(c.disease) + '</span><span class="date">' + esc(c.date) + '</span></div><div class="kv"><div class="c"><div class="k">Crop</div><div class="v">' + esc(c.crop) + '</div></div><div class="c"><div class="k">Stage</div><div class="v">' + esc(c.stage) + '</div></div><div class="c"><div class="k">Diagnosis</div><div class="v">' + esc(c.disease) + '</div></div></div><div class="next' + (warn ? " warn" : "") + '"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg><div><b>' + esc(t().nextAction) + ':</b> ' + esc(c.next) + "</div></div></div>";
  }
  function openProfile(id) {
    const f = farmerById(id); if (!f) return;
    const alerts = f.crops.filter((c) => !isHealthy(c.disease)).length;
    const body = '<div class="profhead"><div class="av" style="background:' + colorFor(f.name) + '">' + esc(initials(f.name)) + '</div><div><h2>' + esc(f.name) + '</h2><div class="m"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>' + esc(f.county) + ' County</span></div></div><div class="profstats"><div><div class="n">' + f.crops.length + '</div><div class="k">Crops</div></div><div><div class="n" style="color:' + (alerts ? "#C56A3E" : "#5C8A3A") + '">' + alerts + '</div><div class="k">To act on</div></div></div></div>' +
      '<div class="timeline">' + (f.crops.length ? f.crops.map(entryHTML).join("") : '<div class="empty">' + esc(t().noCrops) + "</div>") + "</div>" +
      '<button class="addbtn" id="addRecord"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>Add crop record</button>';
    $("profbody").innerHTML = body;
    $("addRecord").addEventListener("click", () => openModal("addRecord", { farmerId: f.id }));
    go("profile");
  }

  function renderHistory() {
    const recs = allRecords().slice().reverse();
    $("allhistory").innerHTML = recs.length ? '<div class="timeline">' + recs.map((x) => {
      const c = x.c, warn = !isHealthy(c.disease);
      return '<div class="entry' + (warn ? " warn" : "") + '"><div class="eh"><b>' + esc(x.f.name.split(" ")[0] + " · " + c.crop) + '</b><span class="badge b-stage">' + esc(c.stage) + '</span><span class="badge ' + diseaseClass(c.disease) + '">' + esc(c.disease) + '</span><span class="date">' + esc(c.date) + '</span></div><div class="next' + (warn ? " warn" : "") + '"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg><div><b>' + esc(t().nextAction) + ':</b> ' + esc(c.next) + "</div></div></div>";
    }).join("") + "</div>" : '<div class="empty">' + esc(t().noCrops) + "</div>";
  }

  // ---------- Sources (static list) ----------
  (function renderSources() {
    const docs = [["KALRO manuals", "Crop management"], ["Infonet-Biovision", "Pests and diseases"], ["Bean Rust.txt", "Beans · rust"], ["Angular Leaf Spot.txt", "Beans · ALS"], ["African Maize stalkborer.txt", "Maize · stem borer"], ["MAIZE_TIMPS Vol.1", "Maize technical"], ["AFA crop strategies", "Value chains"], ["KAMIS market data", "Prices"]];
    $("doclist").innerHTML = docs.map((d) => '<div class="doc"><div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg></div><div><b>' + esc(d[0]) + "</b><br><span>" + esc(d[1]) + "</span></div></div>").join("");
  })();

  // ---------- API ----------
  function postChat(text) {
    return fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: text, language: lang, session_id: sessionId }) }).then((r) => r.json());
  }
  function postClassify(file, text) {
    const form = new FormData(); form.append("file", file); if (text) form.append("query", text); form.append("language", lang); form.append("session_id", sessionId);
    return fetch("/classify", { method: "POST", body: form }).then((r) => r.json());
  }
  function statusEl(el, msg, err) {
    el.hidden = false; el.classList.toggle("is-error", !!err);
    el.innerHTML = err ? esc(msg) : '<span class="spinner"></span><span>' + esc(msg) + "</span>";
  }

  // ---------- Ask ----------
  const q = $("q");
  document.querySelectorAll("[data-fill]").forEach((b) => b.addEventListener("click", () => { q.value = b.dataset.fill; q.focus(); }));
  function renderSources_(list) {
    const wrap = $("srcWrap");
    if (!list || !list.length) { wrap.hidden = true; return; }
    wrap.hidden = false;
    $("srcList").innerHTML = list.map((s) => {
      const name = s.source_filename || s.crop || "reference";
      const score = typeof s.score === "number" ? '<span class="sc">' + s.score.toFixed(2) + "</span>" : "";
      return '<div class="src"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>' + esc(name) + score + "</div>";
    }).join("");
  }
  function showAnswer(answer, confidence, sources, grounded) {
    $("ans").hidden = false;
    const badge = $("ansConf");
    if (confidence) { badge.hidden = false; badge.textContent = (t().badge[confidence] || confidence); badge.className = "badge b-" + confidence; } else badge.hidden = true;
    $("ansbody").textContent = answer || "";
    $("ansCaveat").hidden = !(confidence === "low" || grounded === false);
    renderSources_(sources || []);
    $("ans").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  async function ask() {
    const text = q.value.trim();
    if (!text) { statusEl($("askStatus"), t().errEmpty, true); return; }
    $("askbtn").disabled = true; $("ans").hidden = true;
    statusEl($("askStatus"), t().thinking, false);
    try {
      const body = await postChat(text);
      $("askStatus").hidden = true;
      if (body && body.success === false) {
        const code = (body.error || {}).code, det = (body.error || {}).details || {};
        if (code === "LLM_UNAVAILABLE") { showAnswer(t().llmDown, null, det.sources || [], true); }
        else { statusEl($("askStatus"), (body.error || {}).message || t().errGeneric, true); }
        return;
      }
      const d = (body && body.data) || {};
      showAnswer(d.answer, d.confidence, d.sources, d.grounded);
    } catch (e) {
      statusEl($("askStatus"), e && e.name === "TypeError" ? t().errNetwork : t().errGeneric, true);
    } finally { $("askbtn").disabled = false; }
  }
  $("askbtn").addEventListener("click", ask);
  q.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask(); });

  // ---------- Diagnose ----------
  let photoFile = null;
  const drop = $("drop"), photo = $("photo");
  drop.addEventListener("click", () => photo.click());
  photo.addEventListener("change", () => {
    const f = photo.files && photo.files[0]; if (!f) return;
    photoFile = f; drop.innerHTML = '<img alt="leaf" />'; drop.querySelector("img").src = URL.createObjectURL(f);
  });
  let lastDx = null;
  function meterWidth(level) { return level === "high" ? 88 : level === "medium" ? 66 : 42; }
  async function analyze() {
    if (!photoFile) { statusEl($("dxStatus"), t().needPhoto, true); return; }
    $("analyze").disabled = true;
    statusEl($("dxStatus"), t().analyzing, false);
    try {
      const body = await postClassify(photoFile, q.value.trim());
      $("dxStatus").hidden = true;
      if (body && body.success === false) {
        const code = (body.error || {}).code;
        statusEl($("dxStatus"), code === "INVALID_IMAGE" || code === "IMAGE_NOT_FOUND" ? t().invalidImage : code === "CLASSIFIER_UNAVAILABLE" ? t().classifierOff : ((body.error || {}).message || t().errGeneric), true);
        return;
      }
      renderDx((body && body.data) || {});
    } catch (e) {
      statusEl($("dxStatus"), e && e.name === "TypeError" ? t().errNetwork : t().errGeneric, true);
    } finally { $("analyze").disabled = false; }
  }
  function renderDx(d) {
    const c = d.classification || {};
    const crop = c.crop || "—";
    const cond = c.condition || "—";
    const level = c.confidence_level || d.confidence || "medium";
    const alts = (c.alternatives || []).map((a) => [a.crop, a.condition].filter(Boolean).join(" ") || a.label).filter(Boolean);
    const dx = $("dx"); dx.classList.remove("dim");
    dx.querySelector(".top").innerHTML = '<span class="badge b-' + level + '">' + (t().badge[level] || level) + '</span><span class="big">' + esc(crop) + (isHealthy(cond) ? " · " + t().healthy : " — " + esc(cond)) + "</span>";
    $("dxRows").innerHTML =
      '<div class="r"><span class="k">Crop</span><span style="font-weight:600">' + esc(crop) + '</span></div>' +
      '<div class="r"><span class="k">Condition</span><span class="badge ' + diseaseClass(cond) + '">' + esc(cond) + '</span></div>' +
      (alts.length ? '<div class="r"><span class="k">' + t().alsoPossible + '</span><span style="color:var(--ink-soft)">' + esc(alts.join(", ")) + "</span></div>" : "") +
      '<div class="r"><span class="k">Confidence</span><div class="conf-meter"><i id="cbar"></i></div></div>';
    let nb = dx.querySelector(".nextbox");
    if (!nb) { dx.insertAdjacentHTML("beforeend", '<div class="nextbox"><div class="k">' + t().nextAction + " · grounded</div><p></p></div>"); nb = dx.querySelector(".nextbox"); }
    nb.querySelector(".k").textContent = t().nextAction + " · grounded";
    nb.querySelector("p").textContent = d.answer || "—";
    let btn = dx.querySelector("#saveDx");
    if (!btn) { dx.insertAdjacentHTML("beforeend", '<div style="padding:0 20px 20px"><button class="btn" id="saveDx" style="width:100%;justify-content:center"></button></div>'); btn = dx.querySelector("#saveDx"); btn.addEventListener("click", () => openModal("saveDx", lastDx)); }
    btn.textContent = t().saveProfile;
    setTimeout(() => { const b = dx.querySelector("#cbar"); if (b) b.style.width = meterWidth(level) + "%"; }, 60);
    lastDx = { crop: crop, disease: isHealthy(cond) ? "Healthy" : cond, next: d.answer || "" };
  }
  $("analyze").addEventListener("click", analyze);

  // ---------- Modal (save / add / new farmer) ----------
  const modal = $("modal");
  let modalMode = null, modalFarmerId = null;
  function showFields(opts) {
    $("fldFarmer").hidden = !opts.farmer;
    $("fldName").hidden = !opts.name; $("fldCounty").hidden = !opts.name;
    $("fldCrop").hidden = !opts.crop;
    [$("mStage"), $("mDisease"), $("mNext")].forEach((el) => { el.closest(".field").hidden = !opts.crop; });
  }
  function fillFarmerSelect() {
    $("mFarmer").innerHTML = state.farmers.map((f) => '<option value="' + f.id + '">' + esc(f.name) + " · " + esc(f.county) + "</option>").join("") + '<option value="__new__">+ New farmer</option>';
  }
  function openModal(mode, prefill) {
    modalMode = mode; modalFarmerId = (prefill && prefill.farmerId) || null; prefill = prefill || {};
    if (mode === "newFarmer") {
      $("modalTitle").textContent = lang === "sw" ? "Mkulima mpya" : "New farmer";
      showFields({ farmer: false, name: true, crop: false });
    } else if (mode === "addRecord") {
      $("modalTitle").textContent = lang === "sw" ? "Ongeza rekodi" : "Add crop record";
      showFields({ farmer: false, name: false, crop: true });
    } else { // saveDx
      $("modalTitle").textContent = t().saveProfile;
      fillFarmerSelect();
      showFields({ farmer: true, name: false, crop: true });
      onFarmerChange();
    }
    $("mCrop").value = prefill.crop && /bean/i.test(prefill.crop) ? "Beans" : "Maize";
    $("mDisease").value = prefill.disease || "";
    $("mNext").value = prefill.next || "";
    $("mName").value = ""; $("mCounty").value = "";
    modal.hidden = false;
  }
  function onFarmerChange() {
    const isNew = $("mFarmer").value === "__new__";
    $("fldName").hidden = !isNew; $("fldCounty").hidden = !isNew;
  }
  $("mFarmer").addEventListener("change", onFarmerChange);
  $("mCancel").addEventListener("click", () => (modal.hidden = true));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.hidden = true; });
  function ensureFarmer() {
    if (modalMode === "addRecord") return farmerById(modalFarmerId);
    if (modalMode === "newFarmer") { const f = { id: uid(), name: $("mName").value.trim(), county: $("mCounty").value.trim() || "—", crops: [] }; state.farmers.push(f); return f; }
    // saveDx
    if ($("mFarmer").value === "__new__") { const f = { id: uid(), name: $("mName").value.trim(), county: $("mCounty").value.trim() || "—", crops: [] }; state.farmers.push(f); return f; }
    return farmerById($("mFarmer").value);
  }
  $("mSave").addEventListener("click", () => {
    if (modalMode === "newFarmer" && !$("mName").value.trim()) { $("mName").focus(); return; }
    if (modalMode === "saveDx" && $("mFarmer").value === "__new__" && !$("mName").value.trim()) { $("mName").focus(); return; }
    const f = ensureFarmer(); if (!f) { modal.hidden = true; return; }
    if (modalMode !== "newFarmer") {
      f.crops.push({ id: uid(), crop: $("mCrop").value, stage: $("mStage").value, disease: $("mDisease").value.trim() || "Healthy", next: $("mNext").value.trim() || "—", date: today() });
    }
    save(state); modal.hidden = true; openProfile(f.id);
  });
  $("addFarmer").addEventListener("click", () => openModal("newFarmer", {}));

  // ---------- Boot ----------
  applyLanguage("en");
  renderHome();
})();
