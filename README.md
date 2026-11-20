# Logistat : prédiction du retard de livraison

## Objectif

Prédire, au moment où une commande est passée, si elle sera livrée en retard.

- **Cible :** `Late_delivery_risk` (1 = retard, 0 = à temps)
- **Type de problème :** classification binaire
- **Répartition :** 57,3 % de retards, 42,7 % de livraisons à temps

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `logistat_propre.csv` | Données complètes nettoyées : 172 765 lignes, 42 colonnes, séparateur `;` |
| `train.csv` | Jeu d'entraînement (85 %, 146 850 lignes), toutes les colonnes |
| `test.csv` | Jeu de test (15 %, 25 915 lignes), toutes les colonnes. **Ne pas modifier.** |
| `logistat.ipynb` | Notebook principal : préparation (étape 1) et modèle de référence (étape 2) |
| `selection_variables.docx` | Explication du passage de 42 colonnes à 15 variables |
| `train_prepare.csv`, `test_prepare.csv` | Jeux encodés (108 colonnes + cible), créés quand on exécute le notebook |
| `train.csv.gz` | Copie compressée de `train.csv` laissée par le transfert. Elle peut être supprimée. |

## Séparation entraînement / test

- 85 % entraînement, 15 % test
- Mélange aléatoire avec une graine fixe (`random_state = 42`), reproductible
- **Stratifiée** sur `Late_delivery_risk` : 57,3 % de retards dans chaque jeu

## Règle : ne pas toucher aux données de test

- Toute l'exploration, les choix et les ajustements (encodeur, mise à l'échelle, modèles) se font **sur `train.csv` seulement**.
- Les modèles sont comparés par **validation croisée sur l'entraînement**.
- Le test est seulement *transformé* avec les règles apprises sur train. Il ne sert qu'**une seule fois, à la fin**, pour évaluer le modèle retenu.
- `test.csv` n'est jamais modifié.

## Sélection des variables : de 42 colonnes à 15 variables

| Étape | Colonnes retirées | Nombre |
|---|---|---|
| Cible | `Late_delivery_risk` | 1 |
| Fuites de la réponse (connues seulement après la livraison) | `delay_days`, `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)` | 4 |
| Identifiants et codes | `Order Id`, `Order Item Id`, `Customer Id`, `Product Card Id`, `Category Id`, `Department Id` | 6 |
| Date brute (déjà découpée en mois, jour, heure) | `order date (DateOrders)` | 1 |
| Redondances | `Sales`, `Order Item Total`, `Order Item Discount`, `Order Profit Per Order`, `Customer City/State/Country`, `Order City/State/Country`, `Latitude`, `Longitude`, `Product Name`, `Order Status`, `order_year` | 15 |

**Les 15 variables retenues :**

- **Catégorielles (7)**, encodées en one-hot : `Type`, `Shipping Mode`, `Customer Segment`, `Category Name`, `Department Name`, `Market`, `Order Region`
- **Numériques (8)** : `Days for shipment (scheduled)`, `Order Item Product Price`, `Order Item Quantity`, `Order Item Discount Rate`, `Order Item Profit Ratio`, `order_month`, `order_weekday`, `order_hour`

Le détail est dans `selection_variables.docx`.

## Étapes réalisées

### Étape 1 : préparation

- Les 220 lignes sans `Latitude` / `Longitude` sont **conservées**. Ces deux colonnes sont retirées, car la géographie est déjà couverte par `Market` et `Order Region`. Il ne reste aucune valeur manquante.
- Encodage **one-hot** des 7 variables catégorielles (nominales, sans ordre naturel), ajusté sur train seulement. On obtient 108 colonnes.

### Étape 2 : modèle de référence, régression logistique

- Pipeline : standardisation des variables numériques puis régression logistique
- Validation croisée en 5 plis **regroupée par commande** (`StratifiedGroupKFold` sur `Order Id`), sur train seulement

| Modèle | ROC-AUC | Exactitude | F1 | Rappel retards | Précision retards |
|---|---|---|---|---|---|
| Naïf (toujours « retard ») | 0,500 | 0,573 | 0,728 | 1,000 | 0,573 |
| **Régression logistique** | **0,744** | **0,713** | 0,694 | 0,569 | 0,889 |

**Score de référence à battre : ROC-AUC 0,744.** Le mode d'expédition est le facteur dominant : *First Class* augmente fortement le risque de retard, *Same Day* et *Standard Class* le réduisent.

## Exécuter le notebook

1. Ouvrir un terminal dans ce dossier et installer les dépendances :
   ```
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
2. Lancer Jupyter et ouvrir `logistat.ipynb` :
   ```
   jupyter notebook logistat.ipynb
   ```
3. Exécuter toutes les cellules (*Run All*). Il faut compter environ une minute.

Le notebook lit `train.csv` et `test.csv` dans le même dossier.

## Prochaines étapes

1. **Modèles plus puissants** (forêt aléatoire, gradient boosting), évalués avec la même validation croisée
2. **Ajustement du seuil de décision** pour améliorer le rappel des retards
3. **Réglage des hyperparamètres** du meilleur modèle, toujours sur train
4. **Évaluation finale, une seule fois, sur le jeu de test**
5. Optionnel : validation chronologique (les commandes vont de janvier 2015 à janvier 2018) pour vérifier la tenue du modèle dans le temps
