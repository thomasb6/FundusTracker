// assets/export_pdf.js

function attachPrintButton() {
    // Cherche le bouton à intervalle régulier et ajoute le handler une seule fois
    const btn = document.getElementById('export-report-btn');
    if (btn && !btn.classList.contains("pdf-export-attached")) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            window.print();
        });
        btn.classList.add("pdf-export-attached"); // Marque pour éviter les doublons
    }
}

// Essaie d'accrocher le bouton toutes les 500ms
setInterval(attachPrintButton, 500);
