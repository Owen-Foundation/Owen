(function () {
  "use strict";

  // Theme: stored preference wins, otherwise follow the OS, default dark.
  var root = document.documentElement;
  var toggle = document.getElementById("themeToggle");
  try {
    var stored = localStorage.getItem("owen-theme");
    if (stored === "light" || stored === "dark") {
      root.setAttribute("data-theme", stored);
    } else if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches
    ) {
      root.setAttribute("data-theme", "light");
    }
  } catch (e) {
    /* private mode: stay on the default theme */
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      var next =
        root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("owen-theme", next);
      } catch (e) {
        /* ignore */
      }
    });
  }

  // Copy buttons (build-from-source, run-a-worker).
  Array.prototype.forEach.call(
    document.querySelectorAll("button.copy"),
    function (btn) {
      btn.addEventListener("click", function () {
        var text = btn.getAttribute("data-copy") || "";
        var done = function () {
          var old = btn.textContent;
          btn.textContent = "Copied";
          window.setTimeout(function () {
            btn.textContent = old;
          }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            fallbackCopy(text);
            done();
          });
        } else {
          fallbackCopy(text);
          done();
        }
      });
    }
  );

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "absolute";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch (e) {
      /* ignore */
    }
    document.body.removeChild(ta);
  }

  // The DefenceTest server URL is injected at deploy time. Until then,
  // keep the button from navigating anywhere.
  var dt = document.getElementById("dtLink");
  if (dt && (!dt.getAttribute("href") || dt.getAttribute("href") === "#")) {
    dt.addEventListener("click", function (ev) {
      ev.preventDefault();
    });
    dt.setAttribute("aria-disabled", "true");
    dt.style.opacity = "0.55";
  }
})();
