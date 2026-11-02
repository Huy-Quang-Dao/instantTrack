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
