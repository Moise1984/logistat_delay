"""Contrat entre le bloc 3 (modélisation) et le bloc 1 (application).

Un modèle LogiStat est n'importe quel objet qui respecte ``PredicteurRetard`` :

    nom: str                          # affiché dans l'interface
    variables: list[str]              # colonnes de la table commandes qu'il utilise
    fit(X, y) -> self                 # X = commandes[variables], y = en_retard
    predict_proba(X) -> np.ndarray    # matrice (n, 2) : P(pas de retard), P(retard)

Un ``Pipeline`` scikit-learn convient tel quel si on lui ajoute les attributs
``nom`` et ``variables`` (voir ``envelopper``).

Règle non négociable : ``variables`` ne contient que des colonnes de rôle « facteur »
(connues au moment de la commande). ``verifier_variables`` le contrôle, et un test
automatique fait échouer la CI sinon.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from logistat.donnees.preparation import FACTEURS_COMMANDES, FUITES_COMMANDES, ROLES_COMMANDES


class ErreurModele(ValueError):
    """Le modèle ne respecte pas le contrat."""


@runtime_checkable
class PredicteurRetard(Protocol):
    nom: str
    variables: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PredicteurRetard: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# Variables tirées de la date de commande (connues à la commande), utilisées par le
# modèle du bloc 3 : heure_commande est déjà dans le schéma, mois et jour s'y ajoutent.
DERIVEES_DE_LA_DATE = {"mois_commande": "facteur", "jour_semaine": "facteur"}
# Variables d'un article (ligne de commande), connues à la commande.
FACTEURS_LIGNE = {"prix_unitaire": "facteur", "categorie": "facteur", "departement": "facteur"}


def verifier_variables(variables: list[str]) -> None:
    """Refuse toute variable inconnue, de fuite, d'identifiant ou la cible."""
    roles = {**ROLES_COMMANDES, **DERIVEES_DE_LA_DATE, **FACTEURS_LIGNE}
    autorisees = set(FACTEURS_COMMANDES) | set(DERIVEES_DE_LA_DATE) | set(FACTEURS_LIGNE)
    inconnues = [v for v in variables if v not in roles]
    if inconnues:
        raise ErreurModele("Variables inconnues : " + ", ".join(inconnues))
    interdites = [v for v in variables if v not in autorisees]
    if interdites:
        details = ", ".join(f"{v} ({roles[v]})" for v in interdites)
        raise ErreurModele(
            "Variables interdites dans un modèle (non connues à la commande) : " + details
        )


def verifier_modele(modele: object) -> PredicteurRetard:
    if not isinstance(modele, PredicteurRetard):
        raise ErreurModele(
            "Le modèle doit avoir les attributs nom et variables, et les méthodes "
            "fit et predict_proba."
        )
    verifier_variables(list(modele.variables))
    return modele


def probabilite_retard(modele: PredicteurRetard, commandes: pd.DataFrame) -> np.ndarray:
    """Probabilité de retard pour chaque commande (vecteur de taille n)."""
    proba = np.asarray(modele.predict_proba(commandes[list(modele.variables)]))
    if proba.ndim != 2 or proba.shape != (len(commandes), 2):
        raise ErreurModele(f"predict_proba doit renvoyer (n, 2), reçu {proba.shape}.")
    if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
        raise ErreurModele("predict_proba doit renvoyer des probabilités entre 0 et 1.")
    return proba[:, 1]


def envelopper(estimateur, nom: str, variables: list[str]):
    """Ajoute ``nom`` et ``variables`` à un estimateur scikit-learn (ex. un Pipeline)."""
    verifier_variables(variables)
    estimateur.nom = nom
    estimateur.variables = list(variables)
    return estimateur


__all__ = [
    "FACTEURS_COMMANDES",
    "FUITES_COMMANDES",
    "ErreurModele",
    "PredicteurRetard",
    "envelopper",
    "probabilite_retard",
    "verifier_modele",
    "verifier_variables",
]
