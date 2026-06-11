/* Panneau d'aide des raccourcis clavier.
   100 % côté client : n'interagit avec aucun callback Dash ni aucune logique
   applicative. Documente les raccourcis déjà définis dans keyboard_shortcuts.js.
   Ouverture : touche « ? », bouton flottant, ou ancre #shortcuts. */
(function () {
    "use strict";

    var SHORTCUTS = [
        { keys: ["G"], fr: "Grande lésion", en: "Large lesion" },
        { keys: ["A"], fr: "Atrophie", en: "Atrophy" },
        { keys: ["P", "M"], fr: "Pigment", en: "Pigment" },
        { keys: ["N"], fr: "Nerf optique", en: "Optic nerve" },
        { keys: ["Z"], fr: "Annuler la dernière action", en: "Undo last action" },
        { keys: ["H"], fr: "Afficher / masquer les zones", en: "Toggle shapes" },
        { keys: ["⌫", "Suppr"], fr: "Supprimer la zone active", en: "Delete active shape" },
        { keys: ["?"], fr: "Afficher cette aide", en: "Show this help" }
    ];

    function isTyping() {
        var el = document.activeElement;
        return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
    }

    function buildOverlay() {
        if (document.getElementById("ft-shortcuts-overlay")) return;

        var overlay = document.createElement("div");
        overlay.id = "ft-shortcuts-overlay";
        overlay.className = "ft-shortcuts-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");
        overlay.setAttribute("aria-label", "Keyboard shortcuts");

        var panel = document.createElement("div");
        panel.className = "ft-shortcuts-panel";

        var title = document.createElement("div");
        title.className = "ft-shortcuts-title";
        title.innerHTML = "<span>Raccourcis clavier · Keyboard shortcuts</span>";

        var close = document.createElement("button");
        close.className = "ft-shortcuts-close";
        close.setAttribute("aria-label", "Close");
        close.innerHTML = "✕";
        close.addEventListener("click", hide);
        title.appendChild(close);
        panel.appendChild(title);

        var subtitle = document.createElement("div");
        subtitle.className = "ft-shortcuts-subtitle";
        subtitle.textContent = "Segmentation manuelle · Manual segmentation";
        panel.appendChild(subtitle);

        SHORTCUTS.forEach(function (s) {
            var row = document.createElement("div");
            row.className = "ft-shortcuts-row";
            var keys = document.createElement("span");
            keys.className = "ft-shortcuts-keys";
            s.keys.forEach(function (k, i) {
                if (i > 0) {
                    var sep = document.createElement("span");
                    sep.className = "ft-shortcuts-sep";
                    sep.textContent = "/";
                    keys.appendChild(sep);
                }
                var kbd = document.createElement("kbd");
                kbd.textContent = k;
                keys.appendChild(kbd);
            });
            var label = document.createElement("span");
            label.className = "ft-shortcuts-label";
            label.innerHTML = s.fr + " <span class='ft-shortcuts-en'>· " + s.en + "</span>";
            row.appendChild(keys);
            row.appendChild(label);
            panel.appendChild(row);
        });

        var hint = document.createElement("div");
        hint.className = "ft-shortcuts-hint";
        hint.textContent = "Échap pour fermer · Esc to close";
        panel.appendChild(hint);

        overlay.appendChild(panel);
        overlay.addEventListener("click", function (e) {
            if (e.target === overlay) hide();
        });
        document.body.appendChild(overlay);
    }

    function buildButton() {
        if (document.getElementById("ft-shortcuts-btn")) return;
        var btn = document.createElement("button");
        btn.id = "ft-shortcuts-btn";
        btn.className = "ft-shortcuts-btn";
        btn.type = "button";
        btn.title = "Raccourcis clavier (?)";
        btn.setAttribute("aria-label", "Keyboard shortcuts");
        btn.textContent = "?";
        btn.addEventListener("click", show);
        document.body.appendChild(btn);
    }

    function show() {
        buildOverlay();
        document.getElementById("ft-shortcuts-overlay").classList.add("is-open");
    }
    function hide() {
        var o = document.getElementById("ft-shortcuts-overlay");
        if (o) o.classList.remove("is-open");
    }
    function toggle() {
        var o = document.getElementById("ft-shortcuts-overlay");
        if (o && o.classList.contains("is-open")) hide(); else show();
    }

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") { hide(); return; }
        if (isTyping() || e.ctrlKey || e.metaKey || e.altKey) return;
        if (e.key === "?") { e.preventDefault(); toggle(); }
    });

    function init() {
        buildButton();
        if (window.location.hash === "#shortcuts") show();
    }

    if (document.body) init();
    else document.addEventListener("DOMContentLoaded", init);
})();
