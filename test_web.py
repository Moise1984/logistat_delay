"""La version web (web/moteur.js) doit donner exactement les mêmes probabilités et les
mêmes facteurs que l'application Python, avec le modèle livré."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))

from generer_html import donnees_modele  # noqa: E402

from logistat.donnees.import_utilisateur import normaliser  # noqa: E402
from logistat.modele import probabilite_retard, stockage  # noqa: E402
from logistat.modele.bloc3_gb import GROUPES  # noqa: E402
from logistat.modele.explication import facteurs_retard  # noqa: E402


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js absent")
def test_moteur_js_identique_a_python(tmp_path):
    modele = stockage.charger(next(iter(stockage.lister().values())))
    commandes = normaliser(pd.read_csv(RACINE / "exemples" / "commandes_exemple.csv"))
    commandes = modele.completer(commandes).head(40)
    commandes.loc[0, "region_commande"] = "Atlantis"  # valeur inconnue
    commandes.loc[1, "mode_expedition"] = "Drone"
    lignes = json.loads(commandes[modele.variables].to_json(orient="records"))
    (tmp_path / "m.json").write_text(json.dumps(donnees_modele(modele), default=float))
    (tmp_path / "c.json").write_text(json.dumps(lignes))
    code = f"""
      const {{creerMoteur}} = require({json.dumps(str(RACINE / "web" / "moteur.js"))});
      const fs = require("fs");
      const M = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "m.json"))}));
      const C = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "c.json"))}));
      const m = creerMoteur(M, {{}});
      console.log(JSON.stringify({{p: C.map(m.probabilite), f: m.facteurs(C[0]).facteurs}}));
    """
    sortie = json.loads(
        subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout
    )
    np.testing.assert_allclose(sortie["p"], probabilite_retard(modele, commandes), atol=1e-12)
    attendu = facteurs_retard(
        modele, commandes.iloc[[0]][modele.variables], modele.comparaison, GROUPES
    ).set_index("variable")["effet"]
    for f in sortie["f"]:
        assert f["effet"] == pytest.approx(attendu[f["variable"]], abs=1e-12)


def test_version_web_a_jour():
    page = RACINE / "web" / "logistat.html"
    assert page.exists(), "lancer python scripts/generer_html.py"
    assert "__MODELE__" not in page.read_text(encoding="utf-8")
