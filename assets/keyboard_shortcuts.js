document.addEventListener('keydown', function(event) {

    // 1. SÉCURITÉ : Ne rien faire si on est en train d'écrire du texte
    const activeElem = document.activeElement;
    if (activeElem && (activeElem.tagName === 'INPUT' || activeElem.tagName === 'TEXTAREA')) {
        return;
    }

    // 2. GESTION DE LA SUPPRESSION (Backspace / Delete)
    // On détecte la touche et on clique virtuellement sur le bouton "Erase active shape"
    if (event.key === 'Backspace' || event.key === 'Delete') {

        // On cherche le bouton gomme dans la page via son attribut data-title
        // (C'est comme ça que Plotly nomme ses boutons en interne)
        const eraseButton = document.querySelector('[data-title="Erase active shape"]');

        if (eraseButton) {
            eraseButton.click(); // CLIC VIRTUEL
            event.preventDefault(); // Empêche le navigateur de revenir à la page précédente
            console.log("Suppression déclenchée via clavier");
        } else {
            console.log("Bouton gomme introuvable (vérifiez config_graph)");
        }
        return; // On s'arrête là pour cette touche
    }

    // 3. GESTION DES CLASSIFICATIONS (Vos raccourcis g, a, p, n)
    let shortcuts = {
        'g': 'grande',
        'a': 'atrophie',
        'p': 'pigment',
        'm': 'pigment',
        'n': 'nerf optique'
    };

    let key = event.key.toLowerCase();
    if (key in shortcuts) {
        let desiredText = shortcuts[key];
        let buttons = document.getElementsByClassName('classification-button');

        for (let button of buttons) {
            // On cherche le bouton qui contient le texte
            if (button.innerText.trim().toLowerCase().includes(desiredText)) {
                event.preventDefault();
                button.click();
                break;
            }
        }
    }
});