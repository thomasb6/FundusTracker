document.addEventListener('keydown', function(event) {
    const activeElem = document.activeElement;
    if (activeElem && (activeElem.tagName === 'INPUT' || activeElem.tagName === 'TEXTAREA')) {
        return;
    }
    if (event.key === 'Backspace' || event.key === 'Delete') {
        const eraseButton = document.querySelector('[data-title="Erase active shape"]');
        if (eraseButton) {
            eraseButton.click(); 
            event.preventDefault(); 
            console.log("Suppression déclenchée via clavier");
        } else {
            console.log("Bouton gomme introuvable (vérifiez config_graph)");
        }
        return; 
    }
    let shortcuts = {
        'g': 'grande',
        'a': 'atrophie',
        'p': 'pigment',
        'n': 'nerf optique'
    };
    let key = event.key.toLowerCase();
    if (key in shortcuts) {
        let desiredText = shortcuts[key];
        let buttons = document.getElementsByClassName('classification-button');

        for (let button of buttons) {
            if (button.innerText.trim().toLowerCase().includes(desiredText)) {
                event.preventDefault();
                button.click();
                break;
            }
        }
    }
});
