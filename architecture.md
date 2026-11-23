# Architecture de LogiStat

## Vue en couches et responsabilités

```
                 ┌──────────────────── app.py (Streamlit) ── bloc 1 ─────────────────────┐
                 │ filtres · Vue d'ensemble · Facteurs · Prédiction · Performance · Données │
                 └───────┬───────────────────────┬───────────────────────┬──────────────┘
                         │                       │                       │
   ┌─────────────────────▼─────┐   ┌─────────────▼────────────┐   ┌──────▼──────────────┐
   │ donnees/   (bloc 2 + 1)    │   │ analyse/   (bloc 2)       │   │ modele/  (bloc 3 + 1)│
   │ schema.py  ← CONTRAT       │   │ indicateurs, tests,       │   │ contrat.py ← CONTRAT │
   │ preparation.py             │──►│ distributions…            │   │ decoupage, evaluation│
   │ chargement.py              │   └───────────────────────────┘   │ stockage → modeles/  │
   └────────────────────────────┘                                   └──────────────────────┘
          ▲                                                                   ▲
   scripts/preparer_donnees.py                                   notebooks / scripts du bloc 3
```

Règle : `app.py` n'effectue aucun calcul. Il appelle des fonctions de `analyse/` et
`modele/`, testées séparément.

## Contrat 1 — données (bloc 2 → blocs 1 et 3)

- Point d'entrée unique : `charger_commandes()` renvoie la table `commandes`
  (une ligne par commande), validée par `valider()`.
- Toute colonne a un **rôle** (`docs/donnees.md`). Seules les colonnes de rôle
  `facteur` peuvent entrer dans un modèle.
- Le bloc 2 peut ajouter des règles de nettoyage dans `preparation.py`, sans changer
  les noms ni les rôles existants sans accord de l'équipe (pull request relue).

## Contrat 2 — modèle (bloc 3 → bloc 1)

Un modèle est un objet avec :

| Élément | Exigence |
|---|---|
| `nom` | texte affiché dans l'interface |
| `variables` | liste de colonnes de `commandes`, toutes de rôle `facteur` |
| `fit(X, y)` | `X = commandes[variables]`, `y = commandes["en_retard"]` |
| `predict_proba(X)` | tableau `(n, 2)` ; colonne 1 = probabilité de retard |

Un `Pipeline` scikit-learn convient : `envelopper(pipeline, nom, variables)`. Le bloc 3
appelle `stockage.sauvegarder(modele)` ; le fichier `.joblib` apparaît dans `modeles/`
et l'application l'affiche dans « Prédiction » et « Performance », sans modification de code.

Exemple minimal (bloc 3) :

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from logistat.donnees import charger_commandes, population_modelisation
from logistat.modele import decoupage_chronologique, envelopper, evaluer, probabilite_retard
from logistat.modele import stockage

pop = population_modelisation(charger_commandes().commandes)
d = decoupage_chronologique(pop)

variables = ["mode_expedition", "marche", "type_paiement", "nb_articles"]
prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), variables[:3])], remainder="passthrough"
)
modele = envelopper(
    make_pipeline(prep, LogisticRegression(max_iter=1000)), "Régression logistique", variables
)
modele.fit(d.apprentissage[variables], d.apprentissage["en_retard"])
print(evaluer(d.test["en_retard"], probabilite_retard(modele, d.test)))
stockage.sauvegarder(modele)
```

## Choix communs (à valider par l'équipe)

| Choix | Valeur proposée | Où le changer | Qui valide |
|---|---|---|---|
| Unité de modélisation | la ligne d'article pour le modèle du bloc 3 ; la commande dans l'application (une prédiction par commande) | `bloc3_gb.py`, `import_utilisateur.normaliser` | blocs 1 et 3 |
| Commandes annulées | exclues de la modélisation | `config.EXCLURE_ANNULEES` | blocs 2 et 3 |
| Découpage | 85 / 15 stratifié (graine 42) ; validation croisée regroupée par commande et validation chronologique sur train (bloc 3) | `logistat.ipynb` | bloc 3 (fait) |
| Seuil de décision | 0,61 (ratio de coût 1, bloc 3 étape 8 ; attribut `seuil` du modèle) | `modele.seuil` | bloc 3 (fait) |

## Faits établis lors de la préparation (à interpréter par les blocs 2 et 3)

- 65 752 commandes, 2 855 annulées ; 57,3 % des commandes livrées sont en retard.
- Le délai prévu est entièrement déterminé par le mode d'expédition
  (Same Day 0 j, First Class 1 j, Second Class 2 j, Standard Class 4 j).
- Taux de retard des commandes livrées par mode : First Class 100 %, Second Class 80 %,
  Same Day 48 %, Standard Class 40 %.
- Rupture en octobre 2017 : les commandes ne contiennent plus qu'un article en moyenne
  (1,0 contre 3,0 de janvier 2015 à septembre 2017), et le nombre de commandes par mois
  passe d'environ 1 750 à 2 100. Avec la coupure au 2017-07-01, l'ensemble de test
  contient 4 mois de ce nouveau régime : à prendre en compte dans le choix de la date de
  coupure et des variables (`nb_articles`, `quantite`, montants).
- Les données s'arrêtent le 31 janvier 2018.

## Modèle utilisé par l'application : modèle final du bloc 3 (LightGBM)

Le bloc 3 (Moïse Massack) livre `modele_retard.joblib`, produit par l'étape 8 de
`logistat.ipynb` : encodage one-hot des 7 variables catégorielles puis LightGBM
(299 arbres), sur 15 variables d'une ligne de commande. Seuil de décision : 0,61
(ratio de coût 1). Performance annoncée par le bloc 3 : ROC-AUC d'environ 0,77,
précision d'environ 0,89 et rappel d'environ 0,58 au seuil 0,61.

`src/logistat/modele/bloc3_gb.py` l'enveloppe sans le modifier :

| Variable du bloc 3 | Dans l'application |
|---|---|
| Shipping Mode, Order Region | saisis (la destination peut être choisie par pays : région majoritaire du pays) |
| order_hour, order_weekday, order_month | tirés de la date et de l'heure de commande |
| Category Name, Order Item Quantity, Order Item Product Price | facultatifs (bloc « Produit ») |
| Days for shipment (scheduled) | déduit du mode d'expédition (identique sur 100 % des lignes) |
| Market | déduit de la région (100 %) |
| Department Name | déduit de la catégorie (99,5 %) |
| Customer Segment, Order Item Discount Rate | valeur par défaut |
| Type (paiement), Order Item Profit Ratio | valeurs fixes (DEBIT, médiane 0,27), non demandées : perte de ROC-AUC ≤ 0,003 selon le bloc 3 |

Effet des valeurs fixes, mesuré sur 5 000 lignes réelles : même décision que le modèle
avec les 15 vraies valeurs dans 99,5 % des cas (écart moyen de probabilité 0,036).
L'exemple du README du bloc 3 donne la même probabilité (0,996) : test
`test_bloc3_gb.py`.

## Processus de bout en bout (bloc 1)

| Étape | Qui | Commande | Fichiers lus → produits |
|---|---|---|---|
| 1. Nettoyer et entraîner | bloc 3 | `logistat.ipynb` (étapes 1 à 8) | `train.csv`, `test.csv` → `modele_retard.joblib`, `modele_retard_info.json` |
| 2. Livrer le modèle à l'application | bloc 1 | `python scripts/livrer_modele_bloc3.py modele_retard.joblib logistat_propre.csv --info modele_retard_info.json` | → `modeles/bloc_3_lightgbm_modele_final.joblib` (modèle + listes du formulaire + pays → région + 300 commandes de comparaison) et `exemples/commandes_exemple.csv` |
| 3. Construire la version web | bloc 1 | `python scripts/generer_html.py` | modèle livré + `web/gabarit.html` + `web/moteur.js` → `web/logistat.html` |
| 4. Utiliser l'application | utilisateur | `streamlit run app.py` ou double-clic sur `web/logistat.html` | commandes saisies ou importées → risque de retard, facteurs, export CSV |

Explication des facteurs (`modele/explication.py`) : pour chaque facteur, on remplace sa
valeur par celles de 300 commandes d'entraînement et on mesure l'écart moyen de
probabilité. La version web refait les mêmes calculs (même encodage one-hot, mêmes 299 arbres
LightGBM) ; `tests/test_web.py` vérifie l'égalité avec Python.

Contrôles du fichier importé (`donnees/import_utilisateur.controler`) : colonnes
obligatoires, valeurs jamais vues à l'entraînement, valeurs manquantes, heure hors de
0 à 23, commandes annulées (exclues de la comparaison avec la réalité).
