// Shamba Rafiki kiosk UI behaviour (Phase 4, Output 2).
//
// Talks to the existing backend:
//   - text question         -> POST /chat      (JSON)
//   - photo (+optional text) -> POST /classify  (multipart)
// and renders the standard response envelope: the verified answer, a
// confidence badge, the classification summary (for photos), and the
// corpus sources. Bilingual (English / Kiswahili): the toggle swaps every
// data-en/data-sw string AND sets the `language` field sent to the API.
//
// Dependency-free, offline-first: no framework, no network beyond the
// local backend. Degraded states (model down, empty retrieval, bad image,
// classifier missing) are surfaced clearly rather than failing silently.

(() => {
  "use strict";

  // ---- Localized strings for messages the DOM can't carry statically ----
  const I18N = {
    en: {
      thinking: "Thinking…",
      analyzing: "Analyzing your photo…",
      errGeneric: "Something went wrong. Please try again.",
      errNetwork: "Cannot reach the service. Check that the backend is running.",
      errEmpty: "Please type a question or add a photo.",
      llmDown: "The assistant is temporarily unavailable. Here is the reference material found:",
      noSources: "No matching local references were found for this question.",
      detected: "Detected",
      alsoPossible: "Also possible",
      lowConfNote: "The photo wasn't a clear match, so this is a best guess — confirm with the details below.",
      conf: { low: "Low confidence", medium: "Medium confidence", high: "High confidence" },
      badge: { low: "Low", medium: "Medium", high: "High" },
      invalidImage: "That file couldn't be read as an image. Please try another photo.",
      classifierOff: "Photo analysis isn't available on this device yet. Please type your question instead.",
      healthy: "looks healthy",
    },
    sw: {
      thinking: "Inafikiri…",
      analyzing: "Inachambua picha yako…",
      errGeneric: "Hitilafu imetokea. Tafadhali jaribu tena.",
      errNetwork: "Imeshindwa kufikia huduma. Hakikisha seva inaendeshwa.",
      errEmpty: "Tafadhali andika swali au ongeza picha.",
      llmDown: "Msaidizi hapatikani kwa sasa. Hapa kuna marejeo yaliyopatikana:",
      noSources: "Hakuna marejeo ya eneo yaliyopatikana kwa swali hili.",
      detected: "Imegundulika",
      alsoPossible: "Yawezekana pia",
      lowConfNote: "Picha haikulingana vizuri, hivyo hili ni kisio bora — thibitisha na maelezo hapa chini.",
      conf: { low: "Uhakika mdogo", medium: "Uhakika wa wastani", high: "Uhakika mkubwa" },
      badge: { low: "Mdogo", medium: "Wastani", high: "Mkubwa" },
      invalidImage: "Faili hilo halikusomeka kama picha. Tafadhali jaribu picha nyingine.",
      classifierOff: "Uchambuzi wa picha haupatikani kwenye kifaa hiki bado. Tafadhali andika swali lako.",
      healthy: "unaonekana mzima",
    },
  };

  let lang = "en";
  const sessionId = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());

  // ---- Element handles ----
  const $ = (id) => document.getElementById(id);
  const els = {
    question: $("question"),
    photo: $("photo"),
    photoPreview: $("photo-preview"),
    photoThumb: $("photo-thumb"),
    photoClear: $("photo-clear"),
    ask: $("ask"),
    status: $("status"),
    answerCard: $("answer-card"),
    answer: $("answer"),
    confidence: $("confidence"),
    classification: $("classification"),
    sourcesWrap: $("sources-wrap"),
    sources: $("sources"),
  };

  const t = () => I18N[lang];

  // ---- Language toggle ----
  function applyLanguage(next) {
    lang = next;
    document.documentElement.lang = next;
    document.querySelectorAll(".lang-btn").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.lang === next);
    });
    document.querySelectorAll("[data-en]").forEach((el) => {
      const val = el.getAttribute("data-" + next);
      if (val !== null) el.textContent = val;
    });
    document.querySelectorAll("[data-ph-en]").forEach((el) => {
      const val = el.getAttribute("data-ph-" + next);
      if (val !== null) el.placeholder = val;
    });
  }

  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyLanguage(btn.dataset.lang));
  });

  // ---- Photo selection ----
  let photoFile = null;

  els.photo.addEventListener("change", () => {
    const file = els.photo.files && els.photo.files[0];
    if (!file) return;
    photoFile = file;
    els.photoThumb.src = URL.createObjectURL(file);
    els.photoPreview.hidden = false;
  });

  els.photoClear.addEventListener("click", () => {
    photoFile = null;
    els.photo.value = "";
    els.photoPreview.hidden = true;
    if (els.photoThumb.src) URL.revokeObjectURL(els.photoThumb.src);
    els.photoThumb.removeAttribute("src");
  });

  // ---- Status helpers ----
  function showLoading(message) {
    els.status.hidden = false;
    els.status.classList.remove("is-error");
    els.status.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${escapeHtml(
      message
    )}</span>`;
  }

  function showError(message) {
    els.status.hidden = false;
    els.status.classList.add("is-error");
    els.status.textContent = message;
  }

  function clearStatus() {
    els.status.hidden = true;
    els.status.classList.remove("is-error");
    els.status.innerHTML = "";
  }

  function revealAnswer() {
    els.answerCard.hidden = false;
    // Bring the answer into view on a small kiosk screen.
    if (typeof els.answerCard.scrollIntoView === "function") {
      els.answerCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // ---- Submit ----
  async function ask() {
    const text = els.question.value.trim();
    if (!text && !photoFile) {
      showError(t().errEmpty);
      return;
    }

    els.ask.disabled = true;
    els.answerCard.hidden = true;
    showLoading(photoFile ? t().analyzing : t().thinking);

    try {
      const body = photoFile ? await postClassify(text) : await postChat(text);
      clearStatus();
      renderResponse(body);
    } catch (err) {
      if (err && err.name === "TypeError") {
        showError(t().errNetwork); // fetch failed to reach the server
      } else {
        showError(t().errGeneric);
      }
      // eslint-disable-next-line no-console
      console.error(err);
    } finally {
      els.ask.disabled = false;
    }
  }

  function postChat(text) {
    return fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, language: lang, session_id: sessionId }),
    }).then((r) => r.json());
  }

  function postClassify(text) {
    const form = new FormData();
    form.append("file", photoFile);
    if (text) form.append("query", text);
    form.append("language", lang);
    form.append("session_id", sessionId);
    return fetch("/classify", { method: "POST", body: form }).then((r) => r.json());
  }

  // ---- Rendering ----
  function renderResponse(body) {
    if (!body || typeof body !== "object") {
      showError(t().errGeneric);
      return;
    }

    if (body.success === false) {
      renderError(body.error || {});
      return;
    }

    const data = body.data || {};
    revealAnswer();

    // Confidence badge
    setConfidenceBadge(data.confidence);

    // Classification (photo path only)
    renderClassification(data.classification);

    // Answer
    els.answer.textContent = data.answer || "";

    // Sources
    renderSources(data.sources || []);
  }

  function renderError(error) {
    const code = error.code || "";
    const details = error.details || {};

    if (code === "LLM_UNAVAILABLE") {
      // Degraded: no generated answer, but we may still have references.
      revealAnswer();
      setConfidenceBadge(null);
      renderClassification(details.classification);
      els.answer.textContent = t().llmDown;
      renderSources(details.sources || []);
      return;
    }

    if (code === "INVALID_IMAGE" || code === "IMAGE_NOT_FOUND") {
      showError(t().invalidImage);
      return;
    }
    if (code === "CLASSIFIER_UNAVAILABLE") {
      showError(t().classifierOff);
      return;
    }
    showError(error.message || t().errGeneric);
  }

  function setConfidenceBadge(level) {
    const badge = els.confidence;
    if (!level) {
      badge.hidden = true;
      return;
    }
    badge.hidden = false;
    badge.textContent = (t().badge[level] || level).toString();
    badge.className = "badge is-" + level;
  }

  function renderClassification(c) {
    const box = els.classification;
    if (!c) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    box.hidden = false;

    const crop = c.crop || "";
    const isHealthy = c.condition && String(c.condition).toLowerCase() === "healthy";
    const dxText = isHealthy
      ? `${crop} ${t().healthy}`.trim()
      : [crop, c.condition].filter(Boolean).join(" — ");

    const confWord = t().conf[c.confidence_level] || "";
    let html = `<div class="dx">${t().detected}: ${escapeHtml(dxText)} <span class="alts">(${escapeHtml(
      confWord
    )})</span></div>`;

    if (c.confidence_level === "low") {
      html += `<div class="alts">${escapeHtml(t().lowConfNote)}</div>`;
    }

    const alts = (c.alternatives || []).filter((a) => a && a.label);
    if (alts.length) {
      const altText = alts
        .map((a) => [a.crop, a.condition].filter(Boolean).join(" ") || a.label)
        .join(", ");
      html += `<div class="alts">${t().alsoPossible}: ${escapeHtml(altText)}</div>`;
    }
    box.innerHTML = html;
  }

  function renderSources(sources) {
    if (!sources || !sources.length) {
      els.sourcesWrap.hidden = true;
      els.sources.innerHTML = "";
      return;
    }
    els.sourcesWrap.hidden = false;
    els.sources.innerHTML = sources
      .map((s) => {
        const name = s.source_filename || s.crop || "reference";
        const bits = [s.crop, s.county].filter((x) => x && x !== "unknown").join(", ");
        const score = typeof s.score === "number" ? ` <span class="score">(${s.score})</span>` : "";
        const meta = bits ? ` — ${escapeHtml(bits)}` : "";
        return `<li>${escapeHtml(name)}${meta}${score}</li>`;
      })
      .join("");
  }

  // ---- Utilities ----
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ---- Wire up ----
  els.ask.addEventListener("click", ask);
  els.question.addEventListener("keydown", (e) => {
    // Ctrl/Cmd+Enter submits (Enter alone allows multi-line typing).
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
  });

  applyLanguage("en");
})();
