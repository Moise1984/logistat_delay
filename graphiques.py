"""Graphiques Plotly de LogiStat.

Palette : bleu marine pour la structure, rouge pour ce qui augmente le retard,
vert sarcelle pour ce qui le réduit, gris pour le neutre. Le sens est toujours écrit
en toutes lettres (légende, étiquette, infobulle), jamais porté par la couleur seule.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

MARINE = "#12355B"
BLEU = "#2A78D6"
AMBRE = "#F29E4C"
RETARD = "#D64550"
A_TEMPS = "#1B998B"
GRIS = "#8A8984"
TEXTE = "#1C2B39"

POLICE = dict(family="Source Sans Pro, sans-serif", size=14, color=TEXTE)

_MISE_EN_PAGE = dict(
    margin=dict(l=10, r=10, t=60, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=POLICE,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    hoverlabel=dict(font_size=14),
)


def _base(titre: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titre, x=0, font=dict(size=19, color=MARINE)), **_MISE_EN_PAGE
    )
    fig.update_xaxes(gridcolor="rgba(18,53,91,0.08)", zeroline=False, tickfont=dict(size=13))
    fig.update_yaxes(gridcolor="rgba(18,53,91,0.08)", zeroline=False)
    return fig


def couleur_risque(p: float) -> str:
    if p >= 0.6:
        return RETARD
    if p >= 0.4:
        return AMBRE
    return A_TEMPS


def taux_par_modalite(
    df: pd.DataFrame, colonne: str, libelle: str, taux_global: float | None = None, top: int = 20
) -> go.Figure:
    """Barres horizontales : rouge au-dessus du taux global, sarcelle en dessous."""
    d = df.head(top).iloc[::-1]
    couleurs = [
        RETARD if taux_global is None or t >= taux_global else A_TEMPS for t in d["taux_retard"]
    ]
    fig = _base(f"Taux de retard selon : {libelle}")
    fig.add_bar(
        x=d["taux_retard"],
        y=d[colonne].astype(str),
        orientation="h",
        marker=dict(color=couleurs, line=dict(width=0), cornerradius=4),
        customdata=d["commandes"],
        hovertemplate="<b>%{y}</b><br>Retard : %{x:.1%}<br>%{customdata} commandes<extra></extra>",
        text=d["taux_retard"].map(lambda v: f"{v:.0%}"),
        textposition="outside",
        textfont=dict(size=14, color=TEXTE),
        cliponaxis=False,
    )
    if taux_global is not None:
        fig.add_vline(
            x=taux_global,
            line=dict(color=MARINE, width=2, dash="dash"),
            annotation_text=f"Moyenne {taux_global:.0%}",
            annotation_position="top",
            annotation_font=dict(size=13, color=MARINE),
        )
    fig.update_xaxes(title_text="Part des commandes en retard", tickformat=".0%", range=[0, 1.1])
    fig.update_yaxes(automargin=True, tickfont=dict(size=14))
    fig.update_layout(height=max(340, 30 * len(d) + 130), showlegend=False, bargap=0.3)
    return fig


def jauge_probabilite(p: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=100 * p,
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix=" %", tickfont=dict(size=13)),
                bar=dict(color=couleur_risque(p), thickness=0.3),
                steps=[
                    dict(range=[0, 40], color="rgba(27,153,139,0.15)"),
                    dict(range=[40, 60], color="rgba(242,158,76,0.18)"),
                    dict(range=[60, 100], color="rgba(214,69,80,0.15)"),
                ],
                borderwidth=0,
            ),
        )
    )
    fig.update_layout(
        height=200, margin=dict(l=30, r=30, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=POLICE
    )
    return fig


def facteurs(df: pd.DataFrame, libelles: dict[str, str]) -> go.Figure:
    """Effet de chaque variable sur la probabilité de retard, en points de pourcentage."""
    d = df.iloc[::-1]
    couleurs = [
        RETARD if s == "augmente le risque" else A_TEMPS if s == "réduit le risque" else GRIS
        for s in d["sens"]
    ]
    noms = [
        f"{libelles.get(v, v)} : {val}" for v, val in zip(d["variable"], d["valeur"], strict=True)
    ]
    fig = _base("Ce qui pousse cette commande vers le retard")
    fig.add_bar(
        x=100 * d["effet"],
        y=noms,
        orientation="h",
        marker=dict(color=couleurs, line=dict(width=0), cornerradius=4),
        customdata=d["sens"],
        hovertemplate="<b>%{y}</b><br>%{customdata} : %{text}<extra></extra>",
        text=[f"{100 * e:+.1f} pts".replace(".", ",") for e in d["effet"]],
        textposition="outside",
        textfont=dict(size=14, color=TEXTE),
        cliponaxis=False,
    )
    fig.add_vline(x=0, line=dict(color=MARINE, width=1.5))
    borne = max(5.0, 100 * float(d["effet"].abs().max()) * 1.35)
    fig.update_xaxes(
        title_text="Effet sur la probabilité de retard (points de %)", range=[-borne, borne]
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=14))
    fig.update_layout(height=max(320, 40 * len(d) + 120), showlegend=False, bargap=0.35)
    return fig
