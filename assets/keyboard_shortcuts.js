// assets/keyboard_shortcuts.js
document.addEventListener('keydown', function(event) {
    // Définition du mapping : touche -> texte du bouton
    let shortcuts = {
        'g': 'grande',
        'a': 'atrophie',
        'p': 'pigment',
        'n': 'nerf optique'
    };

    let key = event.key.toLowerCase();
    if (key in shortcuts) {
        let desiredText = shortcuts[key];
        // Récupérer tous les boutons de classification
        let buttons = document.getElementsByClassName('classification-button');
        // Parcourir les boutons et simuler le clic sur celui dont le texte correspond
        for (let button of buttons) {
            if (button.innerText.trim().toLowerCase() === desiredText) {
                event.preventDefault();
                button.click();
                break;
            }
        }
    }
});
