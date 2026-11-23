"""Le modèle final du bloc 3 (LightGBM) livré à l'application : mêmes prédictions que
predire.py du bloc 3, champs déduits cohérents, type de paiement absent de l'interface."""

import numpy as np
import pandas as pd
import pytest

from logistat.modele import probabilite_retard, stockage, verifier_modele
from logistat.modele.bloc3_gb import GROUPES, OBLIGATOIRES
from logistat.modele.explication import facteurs_retard


@pytest.fixture(scope="module")
def modele():
    modeles = stockage.lister()
    assert modeles, "le modèle du bloc 3 doit être livré dans modeles/"
    return stockage.charger(next(iter(modeles.values())))


COMMANDE = {
    "mode_expedition": "Same Day",
    "region_commande": "Central America",
    "heure_commande": 15,
    "jour_semaine": 2,
    "mois_commande": 11,
    "segment_client": "Consumer",
    "categorie": "Cleats",
    "departement": "Apparel",
    "quantite": 2,
    "prix_unitaire": 59.99,
    "taux_remise": 0.05,
}


def test_contrat_et_type_de_paiement_absent(modele):
    verifier_modele(modele)
    assert modele.seuil == pytest.approx(0.61)
    assert set(modele.fixes) == {"Type", "Order Item Profit Ratio"}
    assert "type_paiement" not in modele.variables
    assert set(OBLIGATOIRES) <= set(modele.variables)


def test_meme_resultat_que_predire_py_du_bloc_3(modele):
    """Exemple du README du bloc 3 : predire.py donne 0,996 (ratio de profit 0,25)."""
    fixes = dict(modele.fixes)
    modele.fixes["Order Item Profit Ratio"] = 0.25
    try:
        p = probabilite_retard(modele, pd.DataFrame([COMMANDE]))[0]
    finally:
        modele.fixes = fixes
    assert round(p, 3) == 0.996
    assert p >= modele.seuil


def test_champs_deduits(modele):
    x = modele.traduire(pd.DataFrame([{**COMMANDE, "mode_expedition": "First Class"}]))
    assert list(x.columns) == modele.champs
    assert x.loc[0, "Days for shipment (scheduled)"] == 1
    assert x.loc[0, "Market"] == "LATAM"
    assert x.loc[0, "Type"] == "DEBIT"
    assert x.loc[0, "order_hour"] == 15


def test_champs_facultatifs_remplis(modele):
    minimal = pd.DataFrame(
        [{"mode_expedition": "First Class", "region_commande": "Oceania", "heure_commande": 9}]
    )
    complet = modele.completer(minimal)
    assert complet.loc[0, "categorie"] == modele.defauts["categorie"]
    assert probabilite_retard(modele, complet)[0] > 0.9


def test_valeur_inconnue_ne_plante_pas(modele):
    p = probabilite_retard(modele, pd.DataFrame([{**COMMANDE, "region_commande": "Atlantis"}]))
    assert 0 <= p[0] <= 1


def test_facteurs_groupes(modele):
    cmd = modele.completer(pd.DataFrame([{**COMMANDE, "mode_expedition": "First Class"}]))
    f = facteurs_retard(modele, cmd, modele.comparaison, GROUPES)
    assert "produit" in set(f["variable"])
    assert "categorie" not in set(f["variable"])
    assert f.iloc[0]["variable"] == "mode_expedition"
    assert np.isclose(f.attrs["probabilite"], probabilite_retard(modele, cmd)[0])
