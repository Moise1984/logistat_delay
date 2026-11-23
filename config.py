"""Chemins et paramètres partagés par les trois blocs.

Toute valeur utilisée à plusieurs endroits (chemin, graine, date de coupure) vit ici,
pour être modifiée en un seul endroit et citée dans le rapport.
"""

from __future__ import annotations

from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]

# --- Données ---------------------------------------------------------------
DOSSIER_BRUT = RACINE / "data" / "brut"  # CSV DataCo d'origine (non versionné)
FICHIER_BRUT = DOSSIER_BRUT / "DataCoSupplyChainDataset.csv"
ENCODAGE_BRUT = "latin-1"  # le CSV DataCo n'est pas en UTF-8

FICHIER_LIGNES = RACINE / "data" / "prepare" / "lignes.parquet"  # 1 ligne = 1 article commandé
FICHIER_COMMANDES = RACINE / "data" / "prepare" / "commandes.parquet"  # 1 ligne = 1 commande
FICHIER_ECHANTILLON = RACINE / "data" / "echantillon" / "commandes_echantillon.csv"

# --- Modèles ---------------------------------------------------------------
DOSSIER_MODELES = RACINE / "modeles"  # modèles entraînés (.joblib) livrés par le bloc 3

# --- Reproductibilité ------------------------------------------------------
GRAINE = 42

# Découpage chronologique : apprentissage avant cette date, test à partir d'elle.
# Aligné sur le notebook V1 du bloc 3 (70 % / 15 % / 15 % chronologique) : le test
# correspond aux 15 % de commandes les plus récentes, à partir de cette date.
DATE_COUPURE = "2017-09-05 12:59"
# Fin de l'apprentissage (70 %) et début de la validation (15 %) du bloc 3.
DATE_DEBUT_VALIDATION = "2017-03-16 19:43"

# Les commandes annulées n'ont jamais été livrées : elles sont exclues de la population
# de modélisation par défaut. Décision à valider par les blocs 2 et 3.
EXCLURE_ANNULEES = True
