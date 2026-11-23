"""LogiStat — point d'entrée de l'application.

Lancement :  streamlit run app.py

L'application ne contient aucune donnée : elle charge le modèle livré par le bloc 3
(dossier modeles/) et travaille sur les commandes que l'utilisateur saisit ou importe.

Onglets : Une commande · Mes commandes (fichier) · Analyse des retards.
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent
# Permet de lancer l'application même si le paquet n'a pas été installé (pip install -e .)
sys.path.insert(0, str(RACINE / "src"))

from logistat import __version__
from logistat.analyse.indicateurs import resume_global, taux_retard_par
from logistat.donnees import ErreurContrat
from logistat.donnees.import_utilisateur import (
    completer_destination,
    controler,
    lire_fichier,
    modele_de_fichier,
    normaliser,
)
from logistat.modele import evaluer, probabilite_retard, stockage
from logistat.modele.bloc3_gb import GROUPES, OBLIGATOIRES
from logistat.modele.explication import facteurs_retard
from logistat.ui import graphiques, style

st.set_page_config(page_title="LogiStat", page_icon="📦", layout="wide")
st.markdown(style.CSS, unsafe_allow_html=True)

FICHIER_EXEMPLE = RACINE / "exemples" / "commandes_exemple.csv"
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
LIBELLES = {
    "id_commande": "N° de commande",
    "date_commande": "Date",
    "mode_expedition": "Mode d'expédition",
    "region_commande": "Destination",
    "pays_commande": "Pays",
    "heure_commande": "Heure",
    "jour_semaine": "Jour",
    "mois_commande": "Mois",
    "segment_client": "Segment client",
    "categorie": "Catégorie",
    "departement": "Rayon",
    "quantite": "Quantité",
    "prix_unitaire": "Prix unitaire ($)",
    "taux_remise": "Remise",
    "produit": "Produit",
    "probabilite_retard": "Risque de retard",
    "decision": "Prévision",
    "en_retard": "Retard réel",
}
ANALYSABLES = [
    "mode_expedition",
    "region_commande",
    "pays_commande",
    "heure_commande",
    "jour_semaine",
    "mois_commande",
    "categorie",
    "segment_client",
]


def libelle(v: str) -> str:
    return LIBELLES.get(v, v)


def pct(x: float, d: int = 0) -> str:
    return f"{x:.{d}%}".replace(".", ",")


def nombre(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def valeur_lisible(variable: str, v) -> str:
    if variable == "jour_semaine":
        return JOURS[int(v)]
    if variable == "mois_commande":
        return MOIS[int(v) - 1]
    if variable == "heure_commande":
        return f"{int(v)} h"
    return str(v)


# ------------------------------------------------------------------ cache
@st.cache_resource(show_spinner="Chargement du modèle…")
def modeles_livres() -> dict[str, object]:
    return {nom: stockage.charger(chemin) for nom, chemin in stockage.lister().items()}


@st.cache_data(show_spinner="Lecture du fichier…")
def lire(contenu: bytes, nom: str) -> pd.DataFrame:
    return normaliser(lire_fichier(io.BytesIO(contenu), nom))


def afficher_explication(modele, commande: pd.DataFrame, cle: str) -> None:
    """Verdict et facteurs classés par importance pour UNE commande."""
    explication = facteurs_retard(modele, commande[modele.variables], modele.comparaison, GROUPES)
    p = explication.attrs["probabilite"]
    g, d = st.columns([2, 3])
    with g:
        st.markdown(style.verdict(p, modele.seuil), unsafe_allow_html=True)
    with d:
        st.markdown("#### Ce qui pèse le plus")
        importants = explication[explication["sens"] != "effet négligeable"]
        importants = importants.reindex(
            importants["effet"].abs().sort_values(ascending=False).index
        )
        if importants.empty:
            st.markdown("Rien ne s'écarte d'une commande habituelle.")
        for rang, ligne in enumerate(importants.itertuples(), start=1):
            valeur = (
                valeur_lisible(ligne.variable, ligne.valeur)
                if ligne.variable in modele.variables
                else str(ligne.valeur).split(" · ")[0]
            )
            st.markdown(
                style.facteur(rang, libelle(ligne.variable), valeur, ligne.effet, ligne.sens),
                unsafe_allow_html=True,
            )
    lisible = explication.copy()
    lisible["valeur"] = [
        valeur_lisible(v, x) if v in modele.variables else str(x).split(" · ")[0]
        for v, x in zip(lisible["variable"], lisible["valeur"], strict=True)
    ]
    with st.expander("📈 Tous les facteurs"):
        st.plotly_chart(graphiques.facteurs(lisible, LIBELLES), width="stretch", key=f"fact_{cle}")


# ------------------------------------------------------------------ modèle
modeles = modeles_livres()
if not modeles:
    st.error("Aucun modèle dans modeles/ : lancer scripts/livrer_modele_bloc3.py")
    st.stop()
nom_modele, modele = next(iter(modeles.items()))

with st.sidebar:
    st.markdown("## 📂 Vos commandes")
    televerse = st.file_uploader("Fichier CSV ou Excel", type=["csv", "xlsx", "xls"], key="fichier")
    exemple = st.toggle(
        "Fichier d'exemple",
        value=False,
        disabled=televerse is not None or not FICHIER_EXEMPLE.exists(),
        key="exemple",
    )
    st.download_button(
        "⬇️ Gabarit CSV",
        modele_de_fichier(modele.variables).to_csv(index=False).encode("utf-8"),
        file_name="logistat_gabarit.csv",
    )
    st.caption(f"LogiStat v{__version__}")

donnees: pd.DataFrame | None = None
nom_fichier = ""
try:
    if televerse is not None:
        donnees, nom_fichier = lire(televerse.getvalue(), televerse.name), televerse.name
    elif exemple:
        donnees = lire(FICHIER_EXEMPLE.read_bytes(), FICHIER_EXEMPLE.name)
        nom_fichier = "fichier d'exemple"
except ErreurContrat as err:
    st.sidebar.error(str(err))
if donnees is not None:
    donnees = completer_destination(donnees, modele.pays_vers_region)

st.markdown(
    style.bandeau("LogiStat", "Prévoir les retards de livraison", "Modèle final du bloc 3"),
    unsafe_allow_html=True,
)
onglet_une, onglet_fichier, onglet_analyse = st.tabs(
    ["🔮 Une commande", "📂 Mes commandes", "📊 Analyse des retards"]
)

# ------------------------------------------------------------------ Une commande
with onglet_une:
    regions = modele.modalites["region_commande"]
    pays = sorted(modele.pays_vers_region)
    destinations = [("region", r) for r in regions] + [("pays", p) for p in pays]
    with st.form("formulaire_commande", border=True):
        c1, c2 = st.columns(2)
        choix = c1.selectbox(
            "Destination (région ou pays)",
            range(len(destinations)),
            format_func=lambda i: (
                f"🌍 {destinations[i][1]}"
                if destinations[i][0] == "region"
                else f"{destinations[i][1]}  →  {modele.pays_vers_region[destinations[i][1]]}"
            ),
            index=regions.index("Western Europe") if "Western Europe" in regions else 0,
            key="in_destination",
        )
        mode = c2.selectbox(
            "Mode d'expédition",
            modele.modalites["mode_expedition"],
            index=modele.modalites["mode_expedition"].index("Standard Class"),
            key="in_mode_expedition",
        )
        c3, c4 = st.columns(2)
        jour = c3.date_input("Date de la commande", dt.date(2017, 11, 15), key="in_date")
        heure = c4.number_input("Heure (0-23)", 0, 23, 14, 1, key="in_heure")
        with st.expander("Produit (facultatif)"):
            p1, p2, p3 = st.columns(3)
            categories = modele.modalites["categorie"]
            categorie = p1.selectbox(
                "Catégorie",
                categories,
                index=categories.index(modele.defauts["categorie"]),
                key="in_categorie",
            )
            quantite = p2.number_input("Quantité", 1, 5, 1, key="in_quantite")
            prix = p3.number_input("Prix unitaire ($)", 1.0, 2000.0, 59.99, key="in_prix")
        envoye = st.form_submit_button("🔍 Prédire le retard", type="primary", width="stretch")
    if envoye:
        type_dest, dest = destinations[choix]
        region = dest if type_dest == "region" else modele.pays_vers_region[dest]
        st.session_state["commande_saisie"] = pd.DataFrame(
            [
                {
                    "mode_expedition": mode,
                    "region_commande": region,
                    "heure_commande": int(heure),
                    "jour_semaine": jour.weekday(),
                    "mois_commande": jour.month,
                    "categorie": categorie,
                    "quantite": int(quantite),
                    "prix_unitaire": float(prix),
                }
            ]
        )
    saisie = st.session_state.get("commande_saisie")
    if saisie is not None:
        afficher_explication(modele, modele.completer(saisie), "une")

# ------------------------------------------------------------------ Mes commandes
with onglet_fichier:
    if donnees is None:
        st.info("📂 Importez un fichier à gauche, ou activez le fichier d'exemple.")
        st.markdown(
            "**Colonnes obligatoires :** `mode_expedition`, `region_commande` ou "
            "`pays_commande`, `date_commande`"
        )
    else:
        manquantes = [
            v for v in OBLIGATOIRES if v not in donnees.columns or donnees[v].isna().all()
        ]
        if manquantes:
            st.error("Colonnes manquantes : " + ", ".join(manquantes))
        else:
            avertissements = controler(donnees, OBLIGATOIRES, modele.modalites)
            if avertissements:
                st.warning("\n".join(f"- {m}" for m in avertissements))
            complet = modele.completer(donnees)
            complet["probabilite_retard"] = probabilite_retard(modele, complet)
            complet["decision"] = [
                "en retard" if p >= modele.seuil else "à temps"
                for p in complet["probabilite_retard"]
            ]
            n = len(complet)
            a_risque = int((complet["decision"] == "en retard").sum())
            cartes = [
                ("Commandes", nombre(n), nom_fichier, style.MARINE),
                ("Retards prévus", nombre(a_risque), pct(a_risque / n), style.RETARD),
            ]
            connus = complet[complet["en_retard"].notna()] if "en_retard" in complet else None
            if connus is not None and "annulee" in connus.columns:
                connus = connus[~connus["annulee"].astype(bool)]
            # Le fichier d'exemple a pu servir à l'entraînement : pas de score sur lui.
            if connus is not None and len(connus) and televerse is not None:
                mr = evaluer(
                    connus["en_retard"].astype(bool), connus["probabilite_retard"], modele.seuil
                )
                cartes.append(
                    (
                        "Prévisions justes",
                        pct(mr["exactitude"]),
                        f"sur {nombre(len(connus))} connues",
                        style.A_TEMPS,
                    )
                )
            st.markdown(style.cartes(cartes), unsafe_allow_html=True)

            colonnes = (
                [c for c in ["id_commande", "date_commande", "pays_commande"] if c in complet]
                + ["mode_expedition", "region_commande", "probabilite_retard", "decision"]
                + (["en_retard"] if "en_retard" in complet else [])
            )
            tableau = complet.sort_values("probabilite_retard", ascending=False)[colonnes]
            st.dataframe(
                tableau.rename(columns=libelle),
                hide_index=True,
                width="stretch",
                column_config={
                    libelle("probabilite_retard"): st.column_config.ProgressColumn(
                        format="percent", min_value=0.0, max_value=1.0
                    )
                },
            )
            st.download_button(
                "⬇️ Télécharger (CSV)",
                tableau.to_csv(index=False).encode("utf-8"),
                file_name="logistat_predictions.csv",
            )
            etiquettes = (
                tableau["id_commande"].astype(str)
                if "id_commande" in tableau
                else pd.Series(tableau.index.astype(str), index=tableau.index)
            )
            choisie = st.selectbox(
                "Pourquoi cette commande ?",
                tableau.index,
                format_func=lambda i: (
                    f"{etiquettes[i]} — {pct(tableau.loc[i, 'probabilite_retard'])}"
                ),
                key="commande_choisie",
            )
            afficher_explication(modele, complet.loc[[choisie]], "fichier")

# ------------------------------------------------------------------ Analyse
with onglet_analyse:
    if donnees is None:
        st.info("📂 Importez un fichier à gauche.")
    elif "en_retard" not in donnees.columns:
        st.warning("Il faut une colonne en_retard (1 = en retard, 0 = à temps).")
    else:
        r = resume_global(donnees)
        st.markdown(
            style.cartes(
                [
                    ("Commandes", nombre(r["commandes"]), nom_fichier, style.MARINE),
                    ("Taux de retard", pct(r["taux_retard"], 1), "réel", style.RETARD),
                ]
            ),
            unsafe_allow_html=True,
        )
        disponibles = [v for v in ANALYSABLES if v in donnees.columns]
        c1, c2 = st.columns([1, 3])
        variable = c1.radio("Comparer selon", disponibles, format_func=libelle, key="facteur")
        table = taux_retard_par(donnees, variable)
        if variable in ("jour_semaine", "mois_commande", "heure_commande"):
            table[variable] = [valeur_lisible(variable, v) for v in table[variable]]
        with c2:
            st.plotly_chart(
                graphiques.taux_par_modalite(table, variable, libelle(variable), r["taux_retard"]),
                width="stretch",
            )
