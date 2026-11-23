import io

import pandas as pd
import pytest

from logistat.donnees import ErreurContrat
from logistat.donnees.import_utilisateur import (
    colonnes_manquantes,
    lire_fichier,
    modele_de_fichier,
    normaliser,
)
from logistat.modele.bloc3_gb import CORRESPONDANCE


def test_noms_dataco_et_lignes_articles_regroupees():
    brut = pd.DataFrame(
        {
            "Order Id": [1, 1, 2],
            "order date (DateOrders)": ["1/31/2018 22:56", "1/31/2018 22:56", "2/1/2018 08:05"],
            "Shipping Mode": ["First Class", "First Class", "Standard Class"],
            "Late_delivery_risk": [1, 1, 0],
            "Delivery Status": ["Late delivery", "Late delivery", "Shipping canceled"],
        }
    )
    df = normaliser(brut)
    assert len(df) == 2
    assert df["heure_commande"].tolist() == [22, 8]
    assert df["mode_expedition"].tolist() == ["First Class", "Standard Class"]
    assert df["en_retard"].tolist() == [True, False]
    assert df["annulee"].tolist() == [False, True]


def test_noms_francais_et_point_virgule():
    texte = "id_commande;mode_expedition;heure_commande\n7;Same Day;9\n"
    df = normaliser(lire_fichier(io.StringIO(texte), "a.csv"))
    assert df.loc[0, "mode_expedition"] == "Same Day"
    assert df.loc[0, "heure_commande"] == 9
    assert "en_retard" not in df.columns


def test_colonnes_manquantes_et_gabarit():
    variables = list(CORRESPONDANCE)
    gabarit = modele_de_fichier(variables)
    assert "date_commande" in gabarit.columns and "heure_commande" not in gabarit.columns
    assert "type_paiement" not in gabarit.columns
    df = normaliser(gabarit.assign(date_commande=pd.Series(dtype="object")))
    assert colonnes_manquantes(df, variables) == []
    assert colonnes_manquantes(df.drop(columns="mode_expedition"), variables) == ["mode_expedition"]


def test_jour_mois_heure_et_pays_vers_region():
    from logistat.donnees.import_utilisateur import completer_destination

    df = normaliser(pd.DataFrame({"date_commande": ["2017-11-15 14:05"], "pays": ["Francia"]}))
    assert (df.loc[0, "heure_commande"], df.loc[0, "jour_semaine"], df.loc[0, "mois_commande"]) == (
        14,
        2,  # mercredi
        11,
    )
    df = completer_destination(df, {"Francia": "Western Europe"})
    assert df.loc[0, "region_commande"] == "Western Europe"


def test_fichier_illisible():
    with pytest.raises(ErreurContrat):
        lire_fichier(io.BytesIO(b"\x00\x01"), "a.xlsx")


def test_controle_valeurs_inconnues_manquantes_et_annulees():
    from logistat.donnees.import_utilisateur import controler

    df = pd.DataFrame(
        {
            "mode_expedition": ["First Class", "Fusée", None],
            "heure_commande": [3, 25, 4],
            "annulee": [False, True, False],
        }
    )
    messages = controler(
        df, ["mode_expedition", "heure_commande"], {"mode_expedition": ["First Class"]}
    )
    texte = " ".join(messages)
    assert "1 valeur(s) manquante(s)" in texte
    assert "jamais vue(s)" in texte and "Fusée" in texte
    assert "hors de 0 à 23" in texte
    assert "1 commande(s) annulée(s)" in texte
    assert controler(df.iloc[[0]], ["mode_expedition"], {"mode_expedition": ["First Class"]}) == []
