"""Configuration partagée des tests.

Doit s'exécuter AVANT tout import de ``main``/``auth`` :
- FUNDUS_SKIP_REMOTE coupe l'appel réseau GitHub fait à l'import de main.
- FUNDUS_DATA_DIR isole le stockage dans un dossier temporaire jetable.
"""
import os
import tempfile

os.environ.setdefault("FUNDUS_SKIP_REMOTE", "1")
os.environ.setdefault("FUNDUS_ADMIN_PASSWORD", "test-admin-pw")
if not os.environ.get("FUNDUS_DATA_DIR"):
    os.environ["FUNDUS_DATA_DIR"] = tempfile.mkdtemp(prefix="fundus_tests_")
