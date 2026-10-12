"""
Étape 2 : Ingénierie des caractéristiques biomécaniques (Feature Engineering).

Objectif :
- Charger raw_landmarks.csv (généré à l'étape 1)
- Transformer la séquence temporelle de frames en 1 seul vecteur de variables par vidéo :
    * wrist_delta_y : dénivelé vertical du poignet (y_start - y_end, positif si le bras monte)
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
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.dot(ba, bc) / norm_product
    angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(angle)))

def process_features(input_csv: str = "data/processed/raw_landmarks.csv", output_csv: str = "data/processed/dataset.csv"):
    """
    Lit le fichier raw_landmarks.csv, regroupe par vidéo et extrait les caractéristiques clés.
    """
    if not os.path.exists(input_csv):
        print(f"[02] Erreur : fichier introuvable {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if df.empty:
        print("[02] Erreur : le fichier de repères est vide.")
        return

    grouped = df.groupby("video_id")
    results = []

    for video_id, group in grouped:
        group = group.sort_values("frame_idx")
        label = group["label"].iloc[0]

        # Déplacement du poignet
        wrist_x = group["R_WRIST_x"].values
        wrist_y = group["R_WRIST_y"].values

        wrist_delta_y = float(wrist_y[0] - wrist_y[-1])  # Positif si le bras monte
        wrist_delta_x = float(wrist_x[-1] - wrist_x[0])

        # Vitesse euclidienne maximale entre frames consécutives
        velocities = np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2)
        wrist_max_velocity = float(np.max(velocities)) if len(velocities) > 0 else 0.0

        # Angles du coude au fil du geste
        angles = []
        for _, row in group.iterrows():
            shoulder = np.array([row["R_SHOULDER_x"], row["R_SHOULDER_y"]])
            elbow = np.array([row["R_ELBOW_x"], row["R_ELBOW_y"]])
            wrist = np.array([row["R_WRIST_x"], row["R_WRIST_y"]])
            angles.append(calculate_angle(shoulder, elbow, wrist))

        elbow_angle_min = float(np.min(angles)) if angles else 0.0
        elbow_angle_max = float(np.max(angles)) if angles else 0.0
        duration_frames = int(len(group))

        results.append({
            "video_id": video_id,
            "label": label,
            "wrist_delta_y": wrist_delta_y,
            "wrist_delta_x": wrist_delta_x,
            "wrist_max_velocity": wrist_max_velocity,
            "elbow_angle_min": elbow_angle_min,
            "elbow_angle_max": elbow_angle_max,
            "duration_frames": duration_frames
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"[02] Features générées : {len(results_df)} gestes traités -> {output_csv}")

if __name__ == "__main__":
    process_features()
