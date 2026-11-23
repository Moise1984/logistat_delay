"""Tests de fumée de l'application : elle démarre sans aucun jeu de données, avec le
seul modèle livré, et chaque parcours (saisie, fichier, analyse) fonctionne."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parents[1]
APP = str(RACINE / "app.py")


def lancer() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_application_ne_lit_pas_le_jeu_de_donnees():
    source = (RACINE / "app.py").read_text(encoding="utf-8")
    assert "charger_commandes" not in source
    assert "parquet" not in source


@pytest.mark.slow
def test_app_demarre():
    at = lancer()
    assert [t.label for t in at.tabs] == [
        "🔮 Une commande",
        "📂 Mes commandes",
        "📊 Analyse des retards",
    ]
    libelles = " ".join(w.label for w in at.selectbox)
    assert "paiement" not in libelles  # type de paiement retiré de l'interface


@pytest.mark.slow
def test_bouton_prediction_affiche_facteurs():
    at = lancer()
    at.selectbox(key="in_mode_expedition").set_value("First Class")
    at.button[0].click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Risque élevé" in html
    cartes = [m.value for m in at.markdown if 'class="ls-facteur"' in m.value]
    assert "expédition" in cartes[0]


@pytest.mark.slow
def test_facteurs_classes_par_importance_absolue():
    """Standard Class : l'effet le plus fort est négatif (il réduit le risque) et doit
    quand même apparaître en premier dans la liste."""
    at = lancer()
    at.selectbox(key="in_mode_expedition").set_value("Standard Class")
    at.button[0].click().run()
    assert not at.exception, at.exception
    cartes = [m.value for m in at.markdown if 'class="ls-facteur"' in m.value]
    assert "expédition" in cartes[0]
    assert "#1B998B" in cartes[0]  # couleur « réduit le risque »


@pytest.mark.slow
def test_fichier_exemple_predit_et_analyse():
    at = lancer()
    at.toggle(key="exemple").set_value(True).run()
    assert not at.exception, at.exception
    assert at.dataframe[0].value.shape[0] == 200
    html = " ".join(m.value for m in at.markdown)
    assert "Prévisions justes" not in html  # pas de score sur le fichier d'exemple
    for variable in ("region_commande", "pays_commande", "jour_semaine"):
        at.radio(key="facteur").set_value(variable).run()
        assert not at.exception, at.exception


@pytest.mark.slow
def test_destination_par_pays():
    at = lancer()
    sb = at.selectbox(key="in_destination")
    francia = next(k for k, o in enumerate(sb.options) if o.startswith("Francia"))
    sb.set_value(francia)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("ls-verdict" in m.value for m in at.markdown)
