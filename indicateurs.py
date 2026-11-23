"""Indicateurs descriptifs calculés sur les commandes importées par l'utilisateur.

Le fichier de l'utilisateur ne contient pas toujours les mêmes colonnes : chaque
fonction n'utilise que ce qui est présent (les commandes annulées sont écartées
seulement si l'information existe). Le bloc 2 ajoute ici ses analyses en suivant la
même forme : une fonction pure, documentée et testée.
"""

from __future__ import annotations

import pandas as pd


def _livrees(commandes: pd.DataFrame) -> pd.DataFrame:
    d = commandes[commandes["en_retard"].notna()]
    if "annulee" in d.columns:
        d = d[~d["annulee"].astype(bool)]
    return d


def resume_global(commandes: pd.DataFrame) -> dict[str, float]:
    livrees = _livrees(commandes)
    out = {
        "commandes": len(commandes),
        "avec_resultat": len(livrees),
        "taux_retard": float(livrees["en_retard"].astype(float).mean())
        if len(livrees)
        else float("nan"),
    }
    if "annulee" in commandes.columns:
        out["annulees"] = int(commandes["annulee"].sum())
    if "ecart_jours" in livrees.columns and len(livrees):
        out["ecart_moyen_jours"] = float(livrees["ecart_jours"].mean())
    return out


def taux_retard_par(commandes: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Nombre de commandes et proportion en retard, par modalité de ``colonne``."""
    livrees = _livrees(commandes)
    out = livrees.groupby(colonne, observed=True)["en_retard"].agg(
        commandes="size", taux_retard=lambda s: s.astype(float).mean()
    )
    return out.sort_values("taux_retard", ascending=False).reset_index()


def evolution_mensuelle(commandes: pd.DataFrame) -> pd.DataFrame:
    livrees = _livrees(commandes)
    mois = livrees["date_commande"].dt.to_period("M").dt.to_timestamp()
    out = livrees.groupby(mois)["en_retard"].agg(
        commandes="size", taux_retard=lambda s: s.astype(float).mean()
    )
    out.index.name = "mois"
    return out.reset_index()
