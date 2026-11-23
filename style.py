"""Mise en forme de l'interface : feuille de style et petits composants HTML.

Aucun calcul ici ; uniquement de la présentation.
"""

from __future__ import annotations

from html import escape

CSS = """
<style>
/* ---------- typographie générale ---------- */
html { font-size: 19px; }
html, body, [class*="css"] { font-size: 19px; }
.stMarkdown p, .stMarkdown li { font-size: 1.12rem; }
[data-testid="stWidgetLabel"] p { font-size: 1.1rem; font-weight: 600; }
[data-baseweb="select"] div, .stNumberInput input, .stDateInput input { font-size: 1.1rem; }
[data-testid="stExpander"] summary p { font-size: 1.1rem; font-weight: 600; }
/* Streamlit impose ses tailles : on les force */
[data-testid="stTab"] p, .stTabs [role="tab"] p { font-size: 1.35rem !important;
  font-weight: 700 !important; }
.stFormSubmitButton button p, .stButton button p, .stDownloadButton button p {
  font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stSelectbox"] input, [data-testid="stDateInput"] input, [data-baseweb="select"] div,
.stNumberInput input,
.stDateInput input { font-size: 1.15rem !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 1.15rem !important; }
[data-testid="stDataFrame"] { font-size: 1.05rem; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
h1, h2, h3 { color: #12355B; letter-spacing: -0.01em; }

/* ---------- bandeau d'en-tête ---------- */
.ls-bandeau {
  background: linear-gradient(120deg, #12355B 0%, #1E5288 100%);
  color: #fff; border-radius: 16px; padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap;
}
.ls-bandeau .ls-icone { font-size: 2.8rem; line-height: 1; }
.ls-bandeau h1 { color: #fff; font-size: 2.4rem; margin: 0; padding: 0; }
.ls-bandeau p { margin: .25rem 0 0; font-size: 1.2rem; opacity: .88; }
.ls-pastille {
  margin-left: auto; background: #E8833A; color: #fff; font-weight: 700;
  border-radius: 999px; padding: .35rem .9rem; font-size: .95rem; white-space: nowrap;
}

/* ---------- onglets ---------- */
.stTabs [role="tablist"] { gap: .5rem; }
[data-testid="stTab"] { padding: .6rem 1.2rem !important; }
.stTabs button[role="tab"] {
  padding: .65rem 1.3rem; background: #E9EFF6; border-radius: 12px 12px 0 0;
}
.stTabs button[role="tab"] p { font-size: 1.3rem; font-weight: 700; }
.stTabs button[role="tab"][aria-selected="true"] { background: #fff; }

/* ---------- cartes d'indicateurs ---------- */
.ls-cartes { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: .9rem; margin: .4rem 0 1.2rem; }
.ls-carte { background: #fff; border-radius: 14px; padding: 1rem 1.2rem;
  box-shadow: 0 1px 3px rgba(18,53,91,.10); border-top: 5px solid var(--c, #12355B); }
.ls-carte .ls-libelle { font-size: .9rem; color: #5A6B7D; text-transform: uppercase;
  letter-spacing: .05em; font-weight: 600; }
.ls-carte .ls-valeur { font-size: 2.4rem; font-weight: 700; color: var(--c, #12355B);
  line-height: 1.15; margin-top: .2rem; font-variant-numeric: tabular-nums; }
.ls-carte .ls-note { font-size: .88rem; color: #5A6B7D; margin-top: .15rem; }

/* ---------- verdict de prédiction ---------- */
.ls-verdict { border-radius: 16px; padding: 1.2rem 1.4rem; color: #fff; }
.ls-verdict .ls-titre { font-size: 1.25rem; text-transform: uppercase; letter-spacing: .06em;
  opacity: .92; font-weight: 700; }
.ls-verdict .ls-grand { font-size: 3.6rem; font-weight: 800; line-height: 1.1; margin: .2rem 0; }
.ls-verdict .ls-texte { font-size: 1.2rem; opacity: .95; }
.ls-verdict .ls-decision { margin-top: .7rem; padding-top: .6rem; font-size: 1.2rem;
  border-top: 1px solid rgba(255,255,255,.45); }

/* ---------- liste des facteurs ---------- */
.ls-facteur { display: flex; align-items: center; gap: .8rem; background: #fff;
  border-radius: 12px; padding: .7rem 1rem; margin-bottom: .5rem;
  box-shadow: 0 1px 2px rgba(18,53,91,.08); }
.ls-rang { flex: none; width: 2rem; height: 2rem; border-radius: 50%; color: #fff;
  display: grid; place-items: center; font-weight: 800; }
.ls-facteur .ls-nom { font-weight: 700; color: #12355B; font-size: 1.2rem; }
.ls-facteur .ls-val { color: #5A6B7D; font-size: 1.05rem; }
.ls-facteur .ls-effet { margin-left: auto; font-weight: 800; font-variant-numeric: tabular-nums;
  white-space: nowrap; font-size: 1.3rem; }

/* ---------- bouton principal ---------- */
.stFormSubmitButton button, .stButton button[kind="primary"] {
  font-size: 1.3rem; font-weight: 700; padding: .8rem 1.6rem; border-radius: 12px;
}

/* ---------- barre latérale ---------- */
[data-testid="stSidebar"] h2 { font-size: 1.25rem; }
</style>
"""

RETARD = "#D64550"
A_TEMPS = "#1B998B"
AMBRE = "#E8833A"
MARINE = "#12355B"


def bandeau(titre: str, sous_titre: str, pastille: str) -> str:
    return (
        '<div class="ls-bandeau"><div class="ls-icone">📦</div>'
        f"<div><h1>{escape(titre)}</h1><p>{escape(sous_titre)}</p></div>"
        f'<div class="ls-pastille">{escape(pastille)}</div></div>'
    )


def cartes(elements: list[tuple[str, str, str, str]]) -> str:
    """elements : (libellé, valeur, note, couleur)."""
    html = "".join(
        f'<div class="ls-carte" style="--c:{c}"><div class="ls-libelle">{escape(lib)}</div>'
        f'<div class="ls-valeur">{escape(v)}</div><div class="ls-note">{escape(n)}</div></div>'
        for lib, v, n, c in elements
    )
    return f'<div class="ls-cartes">{html}</div>'


def niveau_risque(p: float, seuil: float = 0.6) -> tuple[str, str, str]:
    """(niveau, couleur, icône) ; le niveau est écrit en toutes lettres.
    « Risque élevé » commence au seuil de décision du modèle : un risque élevé est
    toujours une prévision « en retard »."""
    if p >= seuil:
        return "Risque élevé", RETARD, "🔴"
    if p >= 0.4:
        return "Risque modéré", AMBRE, "🟠"
    return "Risque faible", A_TEMPS, "🟢"


def verdict(p: float, seuil: float = 0.5) -> str:
    niveau, couleur, icone = niveau_risque(p, seuil)
    pct = f"{p:.0%}".replace(".", ",")
    decision = "en retard" if p >= seuil else "à temps"
    return (
        f'<div class="ls-verdict" style="background:{couleur}">'
        f'<div class="ls-titre">{icone} {niveau}</div>'
        f'<div class="ls-grand">{pct}</div>'
        '<div class="ls-texte">de risque de retard</div>'
        f'<div class="ls-decision">Prévision : <b>{decision}</b></div></div>'
    )


def facteur(rang: int, nom: str, valeur: str, effet: float, sens: str) -> str:
    couleur = RETARD if sens == "augmente le risque" else A_TEMPS
    if sens == "effet négligeable":
        couleur = "#8A8984"
    pts = f"{100 * effet:+.1f}".replace(".", ",")
    return (
        f'<div class="ls-facteur"><div class="ls-rang" style="background:{couleur}">{rang}</div>'
        f'<div><div class="ls-nom">{escape(nom)}</div>'
        f'<div class="ls-val">{escape(valeur)}</div></div>'
        f'<div class="ls-effet" style="color:{couleur}">{pts} pts</div></div>'
    )
