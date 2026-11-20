"""Prédiction du risque de retard de livraison.

Utilisation :
    from predire import ModeleRetard
    modele = ModeleRetard('modele_retard.joblib')
    modele.predire({'Type': 'DEBIT', 'Shipping Mode': 'Same Day', ...})   # une commande (dict)
    modele.predire(dataframe)                                              # plusieurs commandes

Dépendances : pandas, scikit-learn, lightgbm, joblib (mêmes versions que l'entraînement,
voir modele.versions ou modele_retard_info.json).
"""
import joblib
import pandas as pd


class ModeleRetard:
    def __init__(self, chemin='modele_retard.joblib'):
        paquet = joblib.load(chemin)
        self.pipeline = paquet['pipeline']
        self.seuil = paquet['seuil']
        self.variables = paquet['variables']
        self.versions = paquet['versions']

    def predire(self, commandes):
        """Retourne un DataFrame avec probabilite_retard (0 à 1) et retard_prevu (0/1)."""
        if isinstance(commandes, dict):
            commandes = pd.DataFrame([commandes])
        manquantes = [v for v in self.variables if v not in commandes.columns]
        if manquantes:
            raise ValueError(f'Variables manquantes : {manquantes}')
        proba = self.pipeline.predict_proba(commandes[self.variables])[:, 1]
        return pd.DataFrame({'probabilite_retard': proba.round(4),
                             'retard_prevu': (proba >= self.seuil).astype(int)},
                            index=commandes.index)


if __name__ == '__main__':
    modele = ModeleRetard()
    exemple = {
        'Type': 'DEBIT', 'Shipping Mode': 'Same Day', 'Customer Segment': 'Consumer',
        'Category Name': 'Cleats', 'Department Name': 'Apparel', 'Market': 'LATAM',
        'Order Region': 'Central America', 'Days for shipment (scheduled)': 0,
        'Order Item Product Price': 59.99, 'Order Item Quantity': 2,
        'Order Item Discount Rate': 0.05, 'Order Item Profit Ratio': 0.25,
        'order_month': 11, 'order_weekday': 2, 'order_hour': 15,
    }
    print(f'Seuil : {modele.seuil}')
    print(modele.predire(exemple))
