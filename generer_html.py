"""Construit web/logistat.html : la version web autonome de LogiStat (un seul fichier).

    python scripts/generer_html.py

À relancer après chaque nouvelle livraison du modèle (scripts/livrer_modele_bloc3.py).
Le script lit le modèle livré dans modeles/ et en extrait les paramètres du pipeline du
bloc 3 (catégories de l'encodage one-hot, 299 arbres du LightGBM) : web/moteur.js
parcourt les mêmes arbres en JavaScript. Le test
tests/test_web.py vérifie que les deux donnent les mêmes probabilités.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from logistat.donnees.import_utilisateur import _ALIAS
from logistat.modele import stockage
from logistat.modele.bloc3_gb import CORRESPONDANCE, GROUPES, OBLIGATOIRES

WEB = RACINE / "web"
SORTIE = WEB / "logistat.html"
EXEMPLE = RACINE / "exemples" / "commandes_exemple.csv"


def parametres(modele) -> dict:
    """Paramètres du pipeline du bloc 3 : encodage one-hot puis arbres du LightGBM."""
    encodage = modele.pipeline.named_steps["encodage"]
    lgbm = modele.pipeline.named_steps["lgbm"]
    colonnes = {nom: list(cols) for nom, _, cols in encodage.transformers_ if nom != "remainder"}
    vidage = lgbm.booster_.dump_model()
    if vidage["num_tree_per_iteration"] != 1 or not vidage["objective"].startswith("binary"):
        raise ValueError("Modèle LightGBM inattendu (classification binaire attendue)")
    arbres = []
    for info in vidage["tree_info"]:
        a = {"f": [], "s": [], "g": [], "d": [], "v": []}

        def parcourir(noeud: dict, a: dict = a) -> int:
            """Numérote les nœuds ; une feuille est codée f = -1, valeur dans v."""
            i = len(a["f"])
            for cle in a:
                a[cle].append(0)
            if "leaf_value" in noeud:
                a["f"][i], a["v"][i] = -1, noeud["leaf_value"]
                return i
            if noeud["decision_type"] != "<=" or noeud["missing_type"] != "None":
                raise ValueError("Type de coupure non pris en charge par la version web")
            a["f"][i], a["s"][i] = noeud["split_feature"], noeud["threshold"]
            a["g"][i] = parcourir(noeud["left_child"])
            a["d"][i] = parcourir(noeud["right_child"])
            return i

        parcourir(info["tree_structure"])
        arbres.append(a)
    return {
        "categorielles": colonnes["cat"],
        "categories": [
            [str(c) for c in cats] for cats in encodage.named_transformers_["cat"].categories_
        ],
        "numeriques": colonnes["num"],
        "sigmoide": float(vidage["objective"].split("sigmoid:")[1]),
        "arbres": arbres,
    }


def donnees_modele(modele) -> dict:
    return {
        "pipeline": parametres(modele),
        "nom": modele.nom,
        "seuil": float(modele.seuil),
        "variables": list(modele.variables),
        "correspondance": CORRESPONDANCE,
        "groupes": GROUPES,
        "obligatoires": OBLIGATOIRES,
        "delai_par_mode": modele.delai_par_mode,
        "marche_par_region": modele.marche_par_region,
        "defauts": modele.defauts,
        "fixes": modele.fixes,
        "departement_par_categorie": modele.departement_par_categorie,
        "pays_vers_region": modele.pays_vers_region,
        "modalites": modele.modalites,
        "comparaison": json.loads(modele.comparaison.to_json(orient="records")),
    }


def sans_balise(texte: str) -> str:
    return texte.replace("</script", "<\\/script")


def main() -> None:
    modeles = stockage.lister()
    if not modeles:
        raise SystemExit("Aucun modèle dans modeles/ : lancez scripts/livrer_modele_bloc3.py")
    modele = stockage.charger(next(iter(modeles.values())))
    remplacements = {
        "__PAPAPARSE__": (WEB / "vendor" / "papaparse.min.js").read_text(encoding="utf-8"),
        "__MOTEUR__": (WEB / "moteur.js").read_text(encoding="utf-8"),
        "__MODELE__": json.dumps(donnees_modele(modele), ensure_ascii=False, default=float),
        "__ALIAS__": json.dumps(_ALIAS, ensure_ascii=False),
        "__EXEMPLE__": json.dumps(EXEMPLE.read_text(encoding="utf-8"), ensure_ascii=False),
    }
    page = (WEB / "gabarit.html").read_text(encoding="utf-8")
    for cle, valeur in remplacements.items():
        page = page.replace(cle, sans_balise(valeur))
    SORTIE.write_text(page, encoding="utf-8")
    print(f"Version web : {SORTIE} ({SORTIE.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
