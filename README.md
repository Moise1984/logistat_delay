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
| `logistat.ipynb` | Notebook principal : préparation (étape 1), modèle de référence (étape 2), modèles plus forts (étape 3), validation chronologique (étape 4), évaluation détaillée (étape 5), réglages (étape 6) |
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

### Étape 3 : modèles plus forts

Même protocole (5 plis regroupés par commande, sur train seulement) :

| Modèle | ROC-AUC | Exactitude | F1 | Rappel retards | Précision retards |
|---|---|---|---|---|---|
| Forêt aléatoire (300 arbres) | **0,771** | **0,721** | 0,708 | 0,591 | 0,884 |
| LightGBM (500 arbres) | 0,770 | 0,714 | **0,709** | **0,608** | 0,849 |
| Régression logistique (référence) | 0,744 | 0,713 | 0,694 | 0,569 | 0,889 |
| Naïf | 0,500 | 0,573 | 0,728 | 1,000 | 0,573 |

- Les deux modèles à arbres battent la référence (ROC-AUC +0,027) et sont à égalité entre eux.
- Variables les plus importantes : délai promis, heure de commande, mode d'expédition.
- Le rappel reste autour de 60 % : c'est le point à améliorer.

### Étape 4 : valider sans tricher

Deux validations, toutes deux sur train seulement :
1. **Validation croisée regroupée par commande** (étapes 2 et 3) : les articles d'une même commande ne sont jamais séparés.
2. **Validation chronologique** : apprentissage sur les commandes jusqu'au 30 juin 2017, validation sur juillet 2017 à janvier 2018 (13,5 % de train). On simule ainsi la prédiction de commandes futures.

| Modèle (ROC-AUC) | CV regroupée | Juil. 2017 - janv. 2018 | Juil. - sept. 2017 | Oct. 2017 - janv. 2018 |
|---|---|---|---|---|
| Forêt aléatoire | 0,771 | **0,782** | 0,780 | 0,786 |
| LightGBM | 0,770 | **0,782** | 0,781 | 0,784 |
| Régression logistique | 0,744 | 0,755 | 0,757 | 0,755 |

- Les scores ne baissent pas sur les mois récents : **pas de fuite**, et les modèles **tiennent dans le temps**.
- Le constat vaut aussi pour la période d'octobre 2017 à janvier 2018, où le volume est réduit (environ 1 800 lignes par mois contre 4 300) et le catalogue passe à 8 à 20 produits.
- Le taux de retard reste stable (environ 57 %) sur toute la période.

### Étape 5 : évaluer avec les bonnes métriques

Avec 57 % de retards, un modèle qui répond toujours « retard » obtient 57 % d'exactitude et un rappel de 100 %, sans rien apprendre. On évalue donc avec le **ROC-AUC**, le **rappel** et la **précision sur les retards**, et le **F1**, à partir de prédictions honnêtes : hors-pli en validation regroupée, et sur les mois récents en validation chronologique. Le notebook contient aussi les matrices de confusion et les courbes ROC et précision-rappel.

| Modèle (CV regroupée, seuil 0,5) | Exactitude | Précision retards | Rappel retards | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Forêt aléatoire | 0,721 | 0,884 | 0,591 | 0,708 | **0,771** |
| LightGBM | 0,714 | 0,849 | **0,608** | **0,709** | 0,770 |
| Régression logistique | 0,713 | 0,889 | 0,569 | 0,694 | 0,743 |
| Naïf | 0,573 | 0,573 | 1,000 | 0,728 | 0,500 |

- Les alertes sont **fiables** (précision de 85 à 89 %), mais environ **40 % des retards sont manqués** au seuil 0,5.
- Jusqu'à environ 60 % de rappel, la précision reste proche de 90 %. Au-delà, elle chute vite.
- Validation chronologique : même conclusion (ROC-AUC 0,782 pour la forêt et LightGBM).

### Étape 6 : réglage des hyperparamètres et du seuil de décision

**Hyperparamètres** (validation regroupée sur train) : une grille pour la forêt et une recherche aléatoire pour LightGBM.

| Modèle | ROC-AUC avant | ROC-AUC après réglage |
|---|---|---|
| Forêt aléatoire | 0,771 | 0,771 |
| LightGBM | 0,770 | 0,770 |

Le réglage ne gagne rien : la limite vient de l'information contenue dans les 15 variables, pas du modèle. **Modèle retenu : LightGBM réglé.**

**Seuil de décision.** Coût = `ratio_cout` × retards manqués + fausses alertes. Le seuil est choisi sur les prédictions hors-pli de train.

| Ratio (coût d'un retard manqué / coût d'une fausse alerte) | Seuil optimal | Rappel retards | Précision retards |
|---|---|---|---|
| 1 | 0,61 | 0,59 | 0,89 |
| 1,5 | 0,59 | 0,59 | 0,89 |
| 2 | 0,19 | 1,00 | 0,59 |
| 3 (hypothèse par défaut) | 0,14 | 1,00 | 0,59 |
| 5 | 0,14 | 1,00 | 0,59 |

- **Bascule vers un ratio de 1,6 :** au-delà, le seuil optimal revient à alerter presque toutes les commandes. Environ 40 % des retards sont indiscernables des commandes à temps avec ces variables.
- Le ratio est une **hypothèse à remplacer** par les coûts réels (variable `ratio_cout` dans le notebook).
- Si un retard manqué coûte cher, une stratégie à deux niveaux est préférable : une action forte au-dessus de 0,6 (alertes justes à environ 89 %) et une action légère sur le reste.

## Exécuter le notebook

1. Ouvrir un terminal dans ce dossier et installer les dépendances :
   ```
   pip install pandas numpy scikit-learn lightgbm matplotlib jupyter
   ```
2. Lancer Jupyter et ouvrir `logistat.ipynb` :
   ```
   jupyter notebook logistat.ipynb
   ```
3. Exécuter toutes les cellules (*Run All*). Il faut compter 10 à 20 minutes selon l'ordinateur, dont la moitié pour la recherche d'hyperparamètres. Pour aller plus vite, mettre `FAIRE_RECHERCHE = False` à l'étape 6 : le notebook reprend alors les meilleurs paramètres déjà trouvés (environ 5 minutes).

Le notebook lit `train.csv` et `test.csv` dans le même dossier.

## Prochaines étapes

1. ~~Modèles plus puissants (forêt aléatoire, gradient boosting)~~ : fait (étape 3)
2. ~~Ajustement du seuil de décision~~ : fait (étape 6)
3. ~~Réglage des hyperparamètres~~ : fait (étape 6)
4. ~~Validation sans tricher (regroupée + chronologique)~~ : fait (étape 4)
5. ~~Évaluation avec F1, rappel, ROC-AUC~~ : fait (étape 5)
6. **Évaluation finale, une seule fois, sur le jeu de test** (LightGBM réglé et seuil choisi)
7. Piste d'amélioration : ajouter des variables connues à la commande (distance, charge de l'entrepôt…)
