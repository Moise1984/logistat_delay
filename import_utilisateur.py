"""Lecture des commandes fournies par l'utilisateur de l'application.

L'application ne contient aucune donnée : l'utilisateur importe un fichier CSV ou
Excel de commandes. Les colonnes peuvent porter :
- les noms d'origine d'un export de type DataCo (« Shipping Mode », « Order City »…) ;
- ou les noms français de LogiStat (« mode_expedition », « ville_commande »…).

Le fichier peut contenir une ligne par commande ou une ligne par article : dans le
second cas, les lignes d'une même commande (même « Order Id ») sont regroupées.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from logistat.donnees import schema
from logistat.donnees.preparation import ErreurContrat

# Noms acceptés en entrée → nom canonique
_ALIAS: dict[str, str] = {c.source.lower(): c.nom for c in schema.COLONNES if c.source}
_ALIAS.update({c.nom.lower(): c.nom for c in schema.COLONNES})
_ALIAS.update(
    {
        "date": "date_commande",
        "heure": "heure_commande",
        "order_hour": "heure_commande",
        "mois": "mois_commande",
        "order_month": "mois_commande",
        "jour_semaine": "jour_semaine",
        "order_weekday": "jour_semaine",
        "pays": "pays_commande",
        "region": "region_commande",
        "région": "region_commande",
    }
)
DERIVEES_DATE = {
    "heure_commande": lambda d: d.dt.hour,
    "jour_semaine": lambda d: d.dt.dayofweek,
    "mois_commande": lambda d: d.dt.month,
}

COLONNE_CIBLE = "en_retard"
STATUT_ANNULE = "Shipping canceled"


def lire_fichier(source: str | Path | IO, nom: str | None = None) -> pd.DataFrame:
    """Lit un CSV (UTF-8 ou latin-1, virgule ou point-virgule) ou un fichier Excel."""
    nom = (nom or str(getattr(source, "name", source))).lower()
    try:
        if nom.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        for encodage in ("utf-8", "latin-1"):
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                return pd.read_csv(source, sep=None, engine="python", encoding=encodage)
            except UnicodeDecodeError:
                continue
    except Exception as exc:  # noqa: BLE001 — message clair pour l'utilisateur
        raise ErreurContrat(f"Lecture du fichier impossible : {exc}") from exc
    raise ErreurContrat("Encodage du fichier non reconnu (utilisez UTF-8).")


def normaliser(brut: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes connues, calcule l'heure à partir de la date, regroupe
    les articles d'une même commande et convertit la cible si elle est présente."""
    df = brut.rename(columns=lambda c: _ALIAS.get(str(c).strip().lower(), str(c).strip()))
    df = df.loc[:, ~df.columns.duplicated()]

    if "date_commande" in df.columns:
        df["date_commande"] = pd.to_datetime(df["date_commande"], errors="coerce")
        for col, extraire in DERIVEES_DATE.items():
            if col not in df.columns:
                df[col] = extraire(df["date_commande"])

    if "id_commande" in df.columns and df["id_commande"].duplicated().any():
        df = df.groupby("id_commande", sort=False).first().reset_index()

    if COLONNE_CIBLE in df.columns:
        df[COLONNE_CIBLE] = pd.to_numeric(df[COLONNE_CIBLE], errors="coerce").astype("boolean")
    if "statut_livraison" in df.columns:
        df["annulee"] = df["statut_livraison"].eq(STATUT_ANNULE)

    for col in DERIVEES_DATE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def completer_destination(df: pd.DataFrame, pays_vers_region: dict[str, str]) -> pd.DataFrame:
    """Région de destination déduite du pays quand le fichier ne donne que le pays."""
    if "pays_commande" not in df.columns:
        return df
    df = df.copy()
    deduite = df["pays_commande"].map(pays_vers_region)
    if "region_commande" in df.columns:
        df["region_commande"] = df["region_commande"].where(df["region_commande"].notna(), deduite)
    else:
        df["region_commande"] = deduite
    return df


def colonnes_manquantes(df: pd.DataFrame, variables: list[str]) -> list[str]:
    return [v for v in variables if v not in df.columns]


def modele_de_fichier(variables: list[str]) -> pd.DataFrame:
    """Gabarit vide à télécharger : les colonnes attendues par le modèle (l'heure, le
    jour et le mois sont tirés de la date de commande)."""
    cols = ["id_commande", "date_commande"] + [v for v in variables if v not in DERIVEES_DATE]
    return pd.DataFrame(columns=cols)


def controler(df: pd.DataFrame, variables: list[str], modalites: dict[str, list]) -> list[str]:
    """Avertissements à montrer à l'utilisateur avant de faire confiance aux prédictions.

    Le modèle ne plante pas sur une valeur inconnue (elle est simplement ignorée) ni sur
    une valeur manquante (elle est remplacée par la valeur la plus fréquente) : il faut
    donc le dire, sinon la prédiction paraît aussi solide que les autres.
    """
    messages = []
    n = len(df)
    for v in variables:
        if v not in df.columns:
            continue
        manquants = int(df[v].isna().sum() + (df[v].astype(str).str.strip() == "").sum())
        if manquants:
            messages.append(
                f"{v} : {manquants} valeur(s) manquante(s) sur {n}, remplacée(s) par la "
                "valeur la plus fréquente."
            )
        if v in modalites:
            connues = set(map(str, modalites[v]))
            inconnues = df[v].dropna().astype(str)
            inconnues = inconnues[~inconnues.isin(connues) & (inconnues.str.strip() != "")]
            if len(inconnues):
                exemples = ", ".join(sorted(inconnues.unique())[:3])
                messages.append(
                    f"{v} : {len(inconnues)} valeur(s) jamais vue(s) à l'entraînement "
                    f"(ex. {exemples}) ; le modèle n'en tient pas compte."
                )
    if "heure_commande" in df.columns:
        h = pd.to_numeric(df["heure_commande"], errors="coerce")
        hors = int(((h < 0) | (h > 23)).sum())
        if hors:
            messages.append(f"heure_commande : {hors} valeur(s) hors de 0 à 23.")
    if "annulee" in df.columns and df["annulee"].any():
        messages.append(
            f"{int(df['annulee'].sum())} commande(s) annulée(s) : elles ne seront jamais "
            "livrées, elles sont exclues de la comparaison avec la réalité."
        )
    return messages
