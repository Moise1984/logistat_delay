"""Livre le modèle final du bloc 3 (Moïse Massack) à l'application (dossier modeles/).

Étape préalable, dans le dépôt du bloc 3 : exécuter logistat.ipynb (étape 8), qui
produit modele_retard.joblib et modele_retard_info.json.

Puis ici :
    python scripts/livrer_modele_bloc3.py modele_retard.joblib logistat_propre.csv \
        --info modele_retard_info.json
    python scripts/generer_html.py

Le fichier livré contient le modèle du bloc 3 inchangé, plus ce dont l'application a
besoin sans le jeu de données : listes du formulaire, correspondance pays → région,
valeurs par défaut, 300 commandes servant de point de comparaison pour expliquer les
facteurs. Produit aussi exemples/commandes_exemple.csv (200 commandes de démonstration ;
le découpage entraînement / test du bloc 3 n'étant pas publié, elles peuvent avoir
servi à l'entraînement : elles ne servent pas à mesurer la performance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

import joblib
import pandas as pd

from logistat.config import GRAINE
from logistat.modele import stockage
from logistat.modele.bloc3_gb import CORRESPONDANCE, ModeleBloc3

FICHIER_EXEMPLE = RACINE / "exemples" / "commandes_exemple.csv"
COLONNES_EXEMPLE = [
    "id_commande",
    "date_commande",
    "mode_expedition",
    "region_commande",
    "pays_commande",
    "segment_client",
    "categorie",
    "departement",
    "quantite",
    "prix_unitaire",
    "taux_remise",
    "en_retard",
]


def vers_logistat(df: pd.DataFrame) -> pd.DataFrame:
    """Lignes du jeu propre du bloc 3 → champs LogiStat."""
    date = pd.to_datetime(df["order date (DateOrders)"])
    out = df.rename(columns={v: k for k, v in CORRESPONDANCE.items()})
    out = out.assign(
        id_commande=df["Order Id"],
        date_commande=date,
        heure_commande=date.dt.hour,
        jour_semaine=date.dt.dayofweek,
        mois_commande=date.dt.month,
        pays_commande=df["Order Country"],
        en_retard=df["Late_delivery_risk"].astype(int),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("modele", type=Path, help="modele_retard.joblib du bloc 3")
    parser.add_argument("donnees", type=Path, help="logistat_propre.csv du bloc 3")
    parser.add_argument("--info", type=Path, help="modele_retard_info.json du bloc 3")
    args = parser.parse_args()

    paquet = joblib.load(args.modele)
    sep = ";" if ";" in args.donnees.open(encoding="utf-8-sig").readline() else ","
    propre = pd.read_csv(args.donnees, sep=sep, encoding="utf-8-sig")
    lignes = vers_logistat(propre)

    defauts = {
        "segment_client": "Consumer",
        "categorie": "Cleats",
        "departement": "Apparel",
        "quantite": 1,
        "prix_unitaire": 59.99,
        "taux_remise": 0.05,
        "jour_semaine": int(lignes["jour_semaine"].mode()[0]),
        "mois_commande": int(lignes["mois_commande"].mode()[0]),
    }
    references = {
        "delai_par_mode": propre.groupby("Shipping Mode")["Days for shipment (scheduled)"]
        .first()
        .astype(int)
        .to_dict(),
        "marche_par_region": propre.groupby("Order Region")["Market"]
        .agg(lambda s: s.mode()[0])
        .to_dict(),
        "departement_par_categorie": propre.groupby("Category Name")["Department Name"]
        .agg(lambda s: s.mode()[0])
        .to_dict(),
        "defauts": defauts,
        # Sans effet mesuré par le bloc 3 et non demandées à l'utilisateur
        "fixes": {
            "Type": "DEBIT",
            "Order Item Profit Ratio": float(propre["Order Item Profit Ratio"].median()),
        },
    }
    modele = ModeleBloc3(paquet, references)
    modele.pays_vers_region = (
        lignes.groupby("pays_commande")["region_commande"].agg(lambda s: s.mode()[0]).to_dict()
    )
    modele.modalites = {
        v: sorted(lignes[v].dropna().unique().tolist())
        for v in [
            "mode_expedition",
            "region_commande",
            "segment_client",
            "categorie",
            "departement",
        ]
    }
    modele.comparaison = (
        lignes[modele.variables].sample(300, random_state=GRAINE).reset_index(drop=True)
    )
    info = json.loads(args.info.read_text(encoding="utf-8")) if args.info else {}
    modele.fiche = {
        "source": "DataCo Smart Supply Chain, nettoyé par le bloc 3",
        "seuil": modele.seuil,
        "performance_test": info.get("performance_test", {}),
        "versions": modele.versions,
    }
    for ancien in stockage.lister().values():
        ancien.unlink()
    print(f"Seuil : {modele.seuil} ; modèle livré : {stockage.sauvegarder(modele)}")

    commandes = lignes.groupby("id_commande", sort=False).first().reset_index()
    exemple = commandes.sample(200, random_state=GRAINE).sort_values("date_commande")
    exemple[COLONNES_EXEMPLE].to_csv(FICHIER_EXEMPLE, index=False)
    print(f"Fichier d'exemple : {FICHIER_EXEMPLE}")


if __name__ == "__main__":
    main()
