"""Facteurs qui expliquent la prédiction d'UNE commande, classés par importance.

Méthode (valable pour n'importe quel modèle qui respecte le contrat) :
pour chaque variable v, on remplace la valeur saisie par les valeurs que prend v dans
un échantillon de commandes réelles (« commandes de comparaison »), toutes les autres
variables restant celles de la commande. L'effet de v est :

    effet(v) = P(retard | commande) − moyenne des P(retard | commande avec v remplacée)

Un effet de +0,30 signifie : « avec sa valeur actuelle, cette variable augmente la
probabilité de retard de 30 points par rapport à ses valeurs habituelles ».
C'est le principe des explications par permutation conditionnelle (proche des valeurs
de Shapley calculées variable par variable). Limite : les interactions entre variables
ne sont pas séparées.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from logistat.modele.contrat import PredicteurRetard, probabilite_retard

SEUIL_NEGLIGEABLE = 0.01  # moins d'un point de pourcentage


def echantillon_comparaison(commandes: pd.DataFrame, n: int = 300, graine: int = 42):
    """Commandes réelles servant de point de comparaison (tirées de l'apprentissage)."""
    return commandes.sample(min(n, len(commandes)), random_state=graine).reset_index(drop=True)


def facteurs_retard(
    modele: PredicteurRetard,
    commande: pd.DataFrame,
    comparaison: pd.DataFrame,
    groupes: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """``commande`` : DataFrame d'une ligne. Renvoie une ligne par facteur, triée par
    effet décroissant (les facteurs qui poussent le plus vers le retard en premier).

    ``groupes`` réunit des variables liées en un seul facteur, remplacées ensemble
    (ex. {"destination": ["region_commande", "pays_commande", "ville_commande"]}) :
    on évite ainsi de créer des combinaisons impossibles (une ville dans le mauvais pays).
    """
    if len(commande) != 1:
        raise ValueError("Une seule commande à la fois.")
    groupes = {k: [v for v in vs if v in modele.variables] for k, vs in (groupes or {}).items()}
    groupes = {k: vs for k, vs in groupes.items() if vs}
    dans_un_groupe = {v for vs in groupes.values() for v in vs}
    facteurs = dict(groupes)
    facteurs.update({v: [v] for v in modele.variables if v not in dans_un_groupe})

    p_commande = float(probabilite_retard(modele, commande)[0])
    lignes = []
    for nom, variables in facteurs.items():
        variantes = pd.concat([commande] * len(comparaison), ignore_index=True)
        for v in variables:
            variantes[v] = comparaison[v].to_numpy()
        effet = p_commande - float(np.mean(probabilite_retard(modele, variantes)))
        valeur = " · ".join(str(commande[v].iloc[0]) for v in variables)
        lignes.append(
            {
                "variable": nom,
                "valeur": valeur if len(variables) > 1 else commande[variables[0]].iloc[0],
                "effet": effet,
            }
        )
    out = pd.DataFrame(lignes).sort_values("effet", ascending=False).reset_index(drop=True)
    out["sens"] = np.select(
        [out["effet"] >= SEUIL_NEGLIGEABLE, out["effet"] <= -SEUIL_NEGLIGEABLE],
        ["augmente le risque", "réduit le risque"],
        default="effet négligeable",
    )
    out.attrs["probabilite"] = p_commande
    return out
