// Reveal-on-scroll, bar-chart grow-in, copy buttons, and pausing off-screen videos.
(() => {
  const els = document.querySelectorAll(".rv, .bars");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { rootMargin: "0px 0px -8% 0px" });
    els.forEach((el) => io.observe(el));
    const vio = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const v = e.target;
        if (!v.autoplay && !v.dataset.played) continue;
        if (e.isIntersecting) v.play().catch(() => {}); else v.pause();
      }
    }, { threshold: 0.25 });
    document.querySelectorAll("video").forEach((v) => {
      v.addEventListener("play", () => { v.dataset.played = "1"; });
      vio.observe(v);
    });
  } else {
    els.forEach((el) => el.classList.add("in"));
  }
  document.querySelectorAll("button.copy").forEach((b) => {
    b.addEventListener("click", async () => {
      const text = document.getElementById(b.dataset.copy).innerText;
      try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; } catch { b.textContent = "Select and copy"; }
      setTimeout(() => { b.textContent = "Copy"; }, 1600);
    });
  });
})();
