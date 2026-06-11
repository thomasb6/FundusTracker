"""Configuration partagée des tests.

Doit s'exécuter AVANT tout import de ``main``/``auth`` :
- FUNDUS_SKIP_REMOTE coupe l'appel réseau GitHub fait à l'import de main.
- FUNDUS_DATA_DIR isole le stockage dans un dossier temporaire jetable.
"""
import os
import tempfile

os.environ.setdefault("FUNDUS_SKIP_REMOTE", "1")
# Tests hors HTTPS : ne pas exiger le flag Secure sur les cookies.
os.environ.setdefault("FUNDUS_SECURE_COOKIES", "0")
# Autoritaire : le compte admin initial est créé avec ce mot de passe, quelle
# que soit la valeur passée par l'environnement CI. Les tests s'y réfèrent.
ADMIN_PASSWORD = "test-admin-pw"
os.environ["FUNDUS_ADMIN_PASSWORD"] = ADMIN_PASSWORD
# Stockage isolé et jetable, toujours neuf (donc base vide → admin recréé).
os.environ["FUNDUS_DATA_DIR"] = tempfile.mkdtemp(prefix="fundus_tests_")
