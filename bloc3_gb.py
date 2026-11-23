"""Modèle final du bloc 3 (Moïse Massack) branché sur l'application.

Le bloc 3 livre ``modele_retard.joblib`` (notebook logistat.ipynb, étape 8) : un
dictionnaire {"pipeline", "seuil", "variables", …}. Le pipeline encode les 7 variables
catégorielles en one-hot puis applique un LightGBM ; le seuil de décision est 0,61
(ratio de coût 1). Cette classe :

- reçoit les champs saisis par l'utilisateur, aux noms français de LogiStat ;
- déduit les champs que l'utilisateur n'a pas à saisir, pour qu'ils restent cohérents :
  délai prévu (fixé par le mode d'expédition), marché (fixé par la région), rayon
  (fixé par la catégorie) ;
- donne une valeur fixe aux deux variables sans effet mesuré par le bloc 3 (perte de
  ROC-AUC ≤ 0,003, étape 7) et non demandées à l'utilisateur : type de paiement et
  ratio de profit de l'article ;
- applique ensuite le modèle du bloc 3 tel quel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Champs saisis (noms LogiStat) → variable du modèle du bloc 3
CORRESPONDANCE: dict[str, str] = {
    "mode_expedition": "Shipping Mode",
    "region_commande": "Order Region",
    "heure_commande": "order_hour",
    "jour_semaine": "order_weekday",
    "mois_commande": "order_month",
    "segment_client": "Customer Segment",
    "categorie": "Category Name",
    "departement": "Department Name",
    "quantite": "Order Item Quantity",
    "prix_unitaire": "Order Item Product Price",
    "taux_remise": "Order Item Discount Rate",
}
NUMERIQUES = [
    "order_hour",
    "order_weekday",
    "order_month",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
]
# Champs indispensables ; les autres prennent une valeur par défaut s'ils manquent.
OBLIGATOIRES = ["mode_expedition", "region_commande", "heure_commande"]
# Facteurs expliqués à l'utilisateur (variables remplacées ensemble)
GROUPES: dict[str, list[str]] = {
    "produit": ["categorie", "departement", "quantite", "prix_unitaire", "taux_remise"],
}


class ModeleBloc3:
    """Respecte le contrat PredicteurRetard (nom, variables, fit, predict_proba)."""

    nom = "Bloc 3 — LightGBM (modèle final)"
    variables = list(CORRESPONDANCE)

    def __init__(self, paquet: dict, references: dict):
        """``paquet`` : contenu de modele_retard.joblib.
        ``references`` : délai par mode, marché par région, rayon par catégorie,
        valeurs par défaut et valeurs fixes (voir scripts/livrer_modele_bloc3.py)."""
        self.pipeline = paquet["pipeline"]
        self.champs: list[str] = list(paquet["variables"])
        self.seuil = float(paquet["seuil"])
        self.versions: dict[str, str] = dict(paquet.get("versions", {}))
        self.delai_par_mode: dict[str, int] = references["delai_par_mode"]
        self.marche_par_region: dict[str, str] = references["marche_par_region"]
        self.departement_par_categorie: dict[str, str] = references["departement_par_categorie"]
        self.defauts: dict[str, object] = references["defauts"]
        self.fixes: dict[str, object] = references["fixes"]

    def completer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les champs facultatifs absents ou vides (valeurs par défaut)."""
        X = X.copy()
        if "categorie" in X.columns and "departement" not in X.columns:
            X["departement"] = X["categorie"].map(self.departement_par_categorie)
        for v in self.variables:
            if v not in X.columns:
                X[v] = self.defauts.get(v, np.nan)
            elif v in self.defauts:
                X[v] = X[v].where(
                    X[v].notna() & (X[v].astype(str).str.strip() != ""), self.defauts[v]
                )
        return X

    def traduire(self, X: pd.DataFrame) -> pd.DataFrame:
        """Champs LogiStat → les 15 variables du modèle du bloc 3."""
        X = self.completer(X)
        out = X[list(CORRESPONDANCE)].rename(columns=CORRESPONDANCE)
        for c in NUMERIQUES:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        out["Days for shipment (scheduled)"] = (
            X["mode_expedition"].map(self.delai_par_mode).astype(float)
        )
        out["Market"] = X["region_commande"].map(self.marche_par_region)
        for c, v in self.fixes.items():
            out[c] = v
        for c in out.columns:
            if c not in NUMERIQUES and not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype(object).where(out[c].notna(), None)
        return out[self.champs]

    def fit(self, X: pd.DataFrame, y) -> ModeleBloc3:
        raise NotImplementedError(
            "Le modèle est entraîné par le bloc 3 (logistat.ipynb), pas par l'application."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self.traduire(X))
