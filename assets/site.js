// InstantTrack page: reveal on scroll, video buttons, highlights carousel, return chart, ablation steps.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const PAUSE = "M4 2.5h3.4v13H4zM10.6 2.5H14v13h-3.4z", PLAY = "M5 2.5v13l10-6.5z";

  // reveal
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px" });
  $$(".rv, .tcard, .steps").forEach((el) => io.observe(el));

  // play/pause buttons on videos
  $$(".vbtn").forEach((b) => {
    const v = document.getElementById(b.dataset.video);
    const sync = () => { b.querySelector("path").setAttribute("d", v.paused ? PLAY : PAUSE); b.setAttribute("aria-label", v.paused ? "Play the video" : "Pause the video"); };
    b.addEventListener("click", () => { v.paused ? v.play() : v.pause(); });
    v.addEventListener("play", sync); v.addEventListener("pause", sync);
    if (reduce) v.pause();
  });

  // highlights carousel: one card in focus, its video plays, auto-advance after a dwell
  const track = $("#hlTrack"), cards = $$(".hcard", track), pager = $("#hlPager"), play = $("#hlPlay");
  let cur = 0, auto = !reduce, timer = null, inView = false;
  cards.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("aria-label", `Highlight ${i + 1}`);
    b.addEventListener("click", () => go(i, true));
    pager.appendChild(b);
  });
  const dots = $$("button", pager);
  function show(i) {
    cur = i;
    dots.forEach((d, k) => d.setAttribute("aria-current", k === i ? "true" : "false"));
    cards.forEach((c, k) => { const v = $("video", c); if (!v) return; if (k === i && inView) { v.currentTime = 0; v.play().catch(() => {}); } else v.pause(); });
    clearTimeout(timer);
    if (auto && inView) timer = setTimeout(() => go((cur + 1) % cards.length), dwell(cards[i]));
  }
  function dwell(c) { const v = $("video", c); return v && v.duration ? Math.max(6000, v.duration * 1000) : 7000; }
  function go(i, user) {
    if (user) { auto = false; syncPlay(); }
    track.scrollTo({ left: cards[i].offsetLeft - cards[0].offsetLeft, behavior: reduce ? "auto" : "smooth" });
    show(i);
  }
  function syncPlay() { $("#hlPlayIcon").setAttribute("d", auto ? PAUSE : PLAY); play.setAttribute("aria-label", auto ? "Pause highlights" : "Play highlights"); }
  play.addEventListener("click", () => { auto = !auto; syncPlay(); show(cur); });
  let st = null;
  track.addEventListener("scroll", () => { clearTimeout(st); st = setTimeout(() => {
    const x = track.scrollLeft, i = cards.reduce((b, c, k) => Math.abs(c.offsetLeft - cards[0].offsetLeft - x) < Math.abs(cards[b].offsetLeft - cards[0].offsetLeft - x) ? k : b, 0);
    if (i !== cur) show(i);
  }, 120); }, { passive: true });
  new IntersectionObserver((es) => { inView = es[0].isIntersecting; show(cur); }, { threshold: 0.35 }).observe(track);
  syncPlay(); show(0);

  // return chart
  const D = JSON.parse($("#curves").textContent);
  const S = [{ k: "sm", n: "SuperMimic", c: "--sm" }, { k: "ema", n: "SuperMimic + EMA", c: "--ema" }, { k: "it", n: "InstantTrack", c: "--it" }];
  const on = { sm: true, ema: true, it: true };
  const el = $("#chart"), W = 900, H = 420, m = { l: 44, r: 18, t: 14, b: 40 };
  const x = (t) => m.l + (t / 0.75) * (W - m.l - m.r), y0 = 0.3, y1 = 0.92;
  const y = (r) => m.t + (1 - (Math.max(r, y0) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const NS = "http://www.w3.org/2000/svg";
  const mk = (n, a = {}, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, el);
  const ax = mk("g", { class: "ax" }, svg);
  for (let r = 0.3; r <= 0.901; r += 0.1) { mk("line", { class: "gl", x1: m.l, x2: W - m.r, y1: y(r), y2: y(r) }, ax); mk("text", { x: m.l - 10, y: y(r) + 4, "text-anchor": "end" }, ax).textContent = r.toFixed(1); }
  for (let t = 0; t <= 0.751; t += 0.125) mk("text", { x: x(t), y: H - 14, "text-anchor": "middle" }, ax).textContent = t === 0 ? "0 h" : `${+t.toFixed(3)} h`;
  mk("line", { class: "tgt", x1: m.l, x2: W - m.r, y1: y(0.82), y2: y(0.82) }, svg);
  mk("text", { x: m.l + 8, y: y(0.82) - 7, class: "lab", fill: "var(--ink2)" }, svg).textContent = "target 0.82";
  const G = {};
  S.forEach((s) => {
    const g = mk("g", {}, svg), d = { ...D[s.k], t: D.t }, col = `var(${s.c})`;
    if (s.k !== "sm") {
      const up = d.t.map((t, i) => `${i ? "L" : "M"}${x(t)},${y(d.rhi[i])}`).join("");
      const lo = d.t.slice().reverse().map((t, j) => { const i = d.t.length - 1 - j; return `L${x(t)},${y(d.rlo[i])}`; }).join("");
      mk("path", { d: up + lo + "Z", fill: col, opacity: 0.14, class: "bd" }, g);
    }
    mk("path", { d: d.t.map((t, i) => `${i ? "L" : "M"}${x(t)},${y(d.r[i])}`).join(""), stroke: col, class: "ln" }, g);
    G[s.k] = g;
  });
  // direct labels at the right edge (above/below their lines)
  [["it", -10], ["ema", 15], ["sm", 18]].forEach(([k, dy]) => {
    const d = { ...D[k], t: D.t }, li = d.t.length - 1, s = S.find((q) => q.k === k);
    mk("text", { x: x(d.t[li]) - 4, y: y(d.r[li]) + dy, class: "lab", fill: `var(${s.c})`, "text-anchor": "end" }, G[k]).textContent = s.n;
  });
  const xh = mk("line", { class: "xh", y1: m.t, y2: H - m.b, opacity: 0 }, svg);
  const pts = S.map((s) => mk("circle", { r: 5, class: "dot", fill: `var(${s.c})`, opacity: 0 }, svg));
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; el.appendChild(tip);
  let ti = -1;
  function read(i) {
    ti = Math.max(0, Math.min(D.t.length - 1, i));
    const t = D.t[ti], px = x(t);
    xh.setAttribute("x1", px); xh.setAttribute("x2", px); xh.setAttribute("opacity", 1);
    S.forEach((s, k) => { pts[k].setAttribute("cx", px); pts[k].setAttribute("cy", y(D[s.k].r[ti])); pts[k].setAttribute("opacity", on[s.k] ? 1 : 0); });
    tip.innerHTML = `<div class="h">${t.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")} h of training</div>` +
      S.slice().reverse().filter((s) => on[s.k]).map((s) => `<div class="r"><span><i style="background:var(${s.c})"></i>${s.n}</span><span>${D[s.k].r[ti].toFixed(3)} · ${D[s.k].f[ti]} failed</span></div>`).join("");
    tip.hidden = false;
    const box = el.getBoundingClientRect(), sx = box.width / W;
    const left = Math.min(Math.max(px * sx - tip.offsetWidth / 2, 0), box.width - tip.offsetWidth);
    tip.style.transform = `translate(${left}px, ${Math.max(0, y(0.92) * sx - 8)}px)`;
    el.setAttribute("aria-valuenow", t); el.setAttribute("aria-valuetext", `${t} hours`);
  }
  function hide() { xh.setAttribute("opacity", 0); pts.forEach((d) => d.setAttribute("opacity", 0)); tip.hidden = true; }
  el.addEventListener("pointermove", (e) => { const b = el.getBoundingClientRect(); const t = ((e.clientX - b.left) / b.width * W - m.l) / (W - m.l - m.r) * 0.75; read(Math.round(t / 0.025)); });
  el.addEventListener("pointerleave", hide);
  el.addEventListener("keydown", (e) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); read((ti < 0 ? 8 : ti) + (e.key === "ArrowRight" ? 1 : -1)); } });
  el.addEventListener("blur", hide);
  $$("#seg button").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.s; if (on[k] && Object.values(on).filter(Boolean).length === 1) return;
    on[k] = !on[k]; b.setAttribute("aria-pressed", on[k]); G[k].classList.toggle("off", !on[k]); if (ti >= 0 && !tip.hidden) read(ti);
  }));

  // ablation steps
  const STEPS = [
    ["SuperMimic", "paper hyperparameters", [1.5], "--sm"],
    ["+ EMA policy", "--pi_ema 0.995", [0.5, 0.54, 0.6], "--ema"],
    ["+ lr 4e-4, 5 WM batches, no syncs, fused Adam", "hyperparameters + systems", [0.29, 0.3, 0.35], "--sm"],
    ["+ fused 3×3 rotations", "1 seed", [0.3], "--sm"],
    ["+ compiled reward / obs / goal", "", [0.25, 0.25, 0.49], "--sm"],
    ["+ world model 1024³", "was 2048³", [0.3, 0.2, 0.3], "--sm"],
    ["+ failure-focused resets", "", [0.2, 0.2, 0.25], "--sm"],
    ["+ contact reward = InstantTrack", "main configuration", [0.25, 0.25, 0.25], "--it"],
  ];
  const sw = $("#steps"), MAX = 1.6;
  STEPS.forEach(([nm, sub, ts, c], i) => {
    const mean = ts.reduce((a, b) => a + b) / ts.length, row = document.createElement("div");
    row.className = "step" + (i === STEPS.length - 1 ? " us" : "");
    const col = i === STEPS.length - 1 ? "var(--it)" : i === 0 ? "var(--sm)" : i === 1 ? "var(--ema)" : "color-mix(in srgb, var(--it) 45%, var(--sm))";
    row.innerHTML = `<div class="nm">${nm}${sub ? `<small>${sub}</small>` : ""}</div>
      <div class="sbar" role="img" aria-label="${ts.map((t) => t.toFixed(2)).join(", ")} hours"><span class="b" style="--w:${(mean / MAX) * 100}%;background:${col};opacity:.28"></span>${ts.map((t) => `<span class="d" style="left:${(t / MAX) * 100}%;background:${col}"></span>`).join("")}</div>
      <div class="v">${mean.toFixed(2)} h<small>${ts.length} seed${ts.length > 1 ? "s" : ""}</small></div>`;
    sw.appendChild(row);
  });

  // copy buttons
  $$("button.copy").forEach((b) => b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(document.getElementById(b.dataset.copy).innerText); b.textContent = "Copied"; } catch { b.textContent = "Select to copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1600);
  }));
})();
