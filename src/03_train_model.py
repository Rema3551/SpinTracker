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
    """
    if not os.path.exists(dataset_csv):
        print(f"[03] Erreur : fichier introuvable {dataset_csv}")
        return

    # 1. Charger le dataset avec pd.read_csv()
    df = pd.read_csv(dataset_csv)
    if df.empty:
        print("[03] Erreur : le dataset est vide.")
        return

    # 2. Définir la liste des features :
    features = [
        "wrist_delta_y", "wrist_delta_x", "wrist_max_velocity",
        "elbow_angle_min", "elbow_angle_max", "duration_frames"
    ]

    # 3. Extraire la matrice X et la cible y (colonne 'label')
    X = df[features]
    y = df['label']

    # 4. Effectuer un train_test_split avec stratification (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5. Instancier RandomForestClassifier et appeler fit() sur les données d'entraînement
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Prédire sur le jeu de test avec predict()
    y_pred = clf.predict(X_test)

    # 7. Afficher classification_report et confusion_matrix
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--- Matrice de Confusion ---")
    print(confusion_matrix(y_test, y_pred))

    # 8. Sauvegarder le modèle avec joblib.dump(clf, model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\n[03] Modèle sauvegardé avec succès -> {model_path}")

if __name__ == "__main__":
    train_classifier()
