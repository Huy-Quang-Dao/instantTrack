/* InstantGMT project page: scroll reveal, FAQ "expand all", BibTeX copy. */
(function(){
  "use strict";
  var rv = document.querySelectorAll(".rv");
  if ("IntersectionObserver" in window){
    var io = new IntersectionObserver(function(es){
      es.forEach(function(e){ if (e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target); } });
    }, {rootMargin:"0px 0px -8% 0px"});
    rv.forEach(function(el){ io.observe(el); });
  } else {
    rv.forEach(function(el){ el.classList.add("in"); });
  }

  var all = document.getElementById("faqAll");
  if (all){
    var dets = document.querySelectorAll("#faqList details");
    all.addEventListener("click", function(){
      var open = all.getAttribute("aria-expanded") !== "true";
      dets.forEach(function(d){ d.open = open; });
      all.setAttribute("aria-expanded", String(open));
      all.firstChild.nodeValue = open ? "Collapse all " : "Expand all ";
    });
  }

  var btn = document.getElementById("bibCopy"), txt = document.getElementById("bibText");
  if (btn && txt){
    btn.addEventListener("click", function(){
      var done = function(){ btn.textContent = "Copied"; setTimeout(function(){ btn.textContent = "Copy"; }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt.textContent).then(done, function(){});
    });
  }
})();

/* Hero morph: scroll progress through the pinned stage drives --p (0 -> 1). */
(function(){
  "use strict";
  var stage = document.getElementById("morphStage"), pin = document.getElementById("morphPin");
  if (!stage || !pin) return;
  var xs = pin.querySelectorAll(".x");
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function measure(){
    // natural width of each hidden fragment, in em of the title, so it scales with the font size
    var h1 = pin.querySelector(".morph"), fs = parseFloat(getComputedStyle(h1).fontSize) || 1;
    xs.forEach(function(x){
      x.style.maxWidth = "none";
      var w = x.getBoundingClientRect().width / fs;
      x.style.maxWidth = "";
      x.style.setProperty("--w", (w + 0.02).toFixed(3) + "em");
    });
  }
  function update(){
    if (reduce){ pin.style.setProperty("--p", "1"); return; }
    var r = stage.getBoundingClientRect(), span = stage.offsetHeight - pin.offsetHeight;
    var p = span > 0 ? Math.min(1, Math.max(0, -r.top / (span * 0.85))) : 1;
    p = p * p * (3 - 2 * p);  // smoothstep
    pin.style.setProperty("--p", p.toFixed(4));
  }
  var queued = false;
  function onScroll(){ if (!queued){ queued = true; requestAnimationFrame(function(){ queued = false; update(); }); } }
  pin.style.setProperty("--p", "1"); measure(); update();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function(){ pin.style.setProperty("--p", "1"); measure(); update(); });
  window.addEventListener("scroll", onScroll, {passive:true});
  window.addEventListener("resize", function(){ pin.style.setProperty("--p", "1"); measure(); update(); });
})();
