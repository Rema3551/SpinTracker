"""
Étape 2 : Ingénierie des caractéristiques biomécaniques (Feature Engineering).

Objectif :
- Charger raw_landmarks.csv (généré à l'étape 1)
- Transformer la séquence temporelle de frames en 1 seul vecteur de variables par vidéo :
    * wrist_delta_y : dénivelé vertical du poignet (y_start - y_end)
    * wrist_delta_x : déplacement horizontal du poignet (x_end - x_start)
    * wrist_max_velocity : vitesse maximale atteinte par le poignet entre deux frames
    * elbow_angle_min / elbow_angle_max : angle min et max formé par l'épaule, le coude et le poignet
    * duration_frames : durée totale du geste (nombre de frames)
- Sauvegarder le résultat dans data/processed/dataset.csv
"""

import os
import numpy as np
import pandas as pd

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calcule l'angle (en degrés) au sommet b formé par les points 2D a, b et c.
    
    Formule mathématique :
    - Vecteur ba = a - b
    - Vecteur bc = c - b
    - cos(angle) = (ba . bc) / (||ba|| * ||bc||)
    - angle = arccos(cos(angle)) converti en degrés
    """
    # TODO: À toi de coder le calcul vectoriel de l'angle !
    return 0.0

def process_features(input_csv: str = "data/processed/raw_landmarks.csv", output_csv: str = "data/processed/dataset.csv"):
    """
    Lit le fichier raw_landmarks.csv, regroupe par vidéo et extrait les caractéristiques clés.
    
    À faire :
    1. Charger input_csv avec pandas
    2. Grouper par 'video_id' (df.groupby) et trier chaque groupe par 'frame_idx'
    3. Pour chaque geste/vidéo :
       - Récupérer le label
       - Calculer le déplacement du poignet (delta_y, delta_x)
       - Calculer la vitesse euclidienne max du poignet entre frames consécutives
       - Calculer l'angle du coude pour chaque frame et trouver l'angle min et max
       - Enregistrer la durée en nombre de frames
    4. Créer un pd.DataFrame avec 1 ligne par vidéo
    5. Sauvegarder dans output_csv
    """
    # TODO: À toi de coder l'agrégation et le calcul des features par vidéo !
    pass

if __name__ == "__main__":
    process_features()
