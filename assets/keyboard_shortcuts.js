document.addEventListener('keydown', function(event) {

    // 1. SÉCURITÉ : Ne rien faire si on écrit dans un champ texte
    // C'est CRUCIAL maintenant que 'z' est un raccourci sans Ctrl
    const activeElem = document.activeElement;
    if (activeElem && (activeElem.tagName === 'INPUT' || activeElem.tagName === 'TEXTAREA')) {
        return;
    }

    // Récupération de la touche
    let key = event.key.toLowerCase();

    // 2. SUPPRESSION (Backspace / Delete)
    if (key === 'backspace' || key === 'delete') {
        const eraseButton = document.querySelector('[data-title="Erase active shape"]');
        if (eraseButton) {
            eraseButton.click();
            event.preventDefault();
            console.log("Suppression déclenchée via clavier");
        }
        return;
    }

    // 3. CLASSIFICATIONS (Raccourcis lettres simples)
    let shortcuts = {
        'g': 'grande',
        'a': 'atrophie',
        'p': 'pigment',
        'm': 'pigment',
        'n': 'nerf optique'
    };

    // On s'assure qu'on ne fait pas Ctrl+A ou Cmd+A par erreur
    if (!event.ctrlKey && !event.metaKey && key in shortcuts) {
        let desiredText = shortcuts[key];
        let buttons = document.getElementsByTagName('button');

        for (let button of buttons) {
            if (button.innerText && button.innerText.trim().toLowerCase().includes(desiredText)) {
                event.preventDefault();
                button.click();
                break;
            }
        }
    }

    // 4. VISIBILITÉ (Touche 'h')
    if (key === 'h') {
        event.preventDefault();
        const btn = document.getElementById('toggle-visibility-btn');
        if (btn) {
            btn.click();
            console.log("Raccourci H activé");
        }
    }

    // 5. UNDO (Touche 'z' seule)
    // On vérifie que Ctrl et Cmd ne sont PAS pressés pour éviter les conflits
    if (key === 'z' && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        const undoBtn = document.getElementById('undo-button');
        if (undoBtn) {
            undoBtn.click();
            console.log("Touche Z -> Annulation déclenchée");
        }
    }

});
