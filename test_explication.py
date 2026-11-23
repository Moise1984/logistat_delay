import numpy as np
import pandas as pd
import pytest

from logistat.modele.explication import facteurs_retard


class ModeleJouet:
    """P(retard) = 0,9 en First Class, 0,3 sinon ; le marché n'a aucun effet."""

    nom = "jouet"
    variables = ["mode_expedition", "marche"]

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.where(X["mode_expedition"] == "First Class", 0.9, 0.3)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def comparaison():
    return pd.DataFrame(
        {
            "mode_expedition": ["First Class"] * 1 + ["Standard Class"] * 3,
            "marche": ["Europe", "LATAM", "USCA", "Africa"],
        }
    )


def test_effet_exact_sur_modele_connu(comparaison):
    cmd = pd.DataFrame([{"mode_expedition": "First Class", "marche": "Europe"}])
    f = facteurs_retard(ModeleJouet(), cmd, comparaison).set_index("variable")
    # 0,9 − moyenne(0,9 ; 0,3 ; 0,3 ; 0,3) = 0,9 − 0,45 = 0,45
    assert f.loc["mode_expedition", "effet"] == pytest.approx(0.45)
    assert f.loc["marche", "effet"] == pytest.approx(0.0)
    assert f.loc["mode_expedition", "sens"] == "augmente le risque"
    assert f.loc["marche", "sens"] == "effet négligeable"


def test_tri_et_sens_negatif(comparaison):
    cmd = pd.DataFrame([{"mode_expedition": "Standard Class", "marche": "Europe"}])
    f = facteurs_retard(ModeleJouet(), cmd, comparaison)
    assert f["effet"].is_monotonic_decreasing
    assert f.iloc[-1]["variable"] == "mode_expedition"
    assert f.iloc[-1]["sens"] == "réduit le risque"
    assert f.attrs["probabilite"] == pytest.approx(0.3)


def test_une_seule_commande(comparaison):
    cmd = pd.DataFrame([{"mode_expedition": "First Class", "marche": "Europe"}] * 2)
    with pytest.raises(ValueError):
        facteurs_retard(ModeleJouet(), cmd, comparaison)
