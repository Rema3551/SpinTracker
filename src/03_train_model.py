"""
Étape 3 : Entraînement et évaluation du modèle de classification.

Objectif :
- Charger data/processed/dataset.csv (généré à l'étape 2)
- Séparer les variables explicatives (features X) de la cible (label y)
- Découper en jeu d'entraînement et jeu de test (train_test_split)
- Entraîner un classifieur Random Forest
- Évaluer les prédictions (rapport de classification et matrice de confusion)
- Sauvegarder le modèle entraîné dans data/processed/model.joblib
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_classifier(dataset_csv: str = "data/processed/dataset.csv", model_path: str = "data/processed/model.joblib"):
    """
    Entraîne le modèle Random Forest et sauvegarde les poids.
    
    À faire :
    1. Charger le dataset avec pd.read_csv()
    2. Définir la liste des features :
       ["wrist_delta_y", "wrist_delta_x", "wrist_max_velocity",
        "elbow_angle_min", "elbow_angle_max", "duration_frames"]
    3. Extraire la matrice X et la cible y (colonne 'label')
    4. Effectuer un train_test_split avec stratification (stratify=y)
    5. Instancier RandomForestClassifier et appeler fit() sur les données d'entraînement
    6. Prédire sur le jeu de test avec predict()
    7. Afficher classification_report et confusion_matrix
    8. Sauvegarder le modèle avec joblib.dump(clf, model_path)
    """
    # TODO: À toi de coder l'entraînement et l'évaluation du modèle !
    pass

if __name__ == "__main__":
    train_classifier()
