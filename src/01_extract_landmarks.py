"""
Étape 1 : Extraction des repères anatomiques (Pose Landmarks) avec MediaPipe.

Objectif :
- Parcourir chaque vidéo dans data/raw_videos/<label>/
- Extraire frame par frame les coordonnées (x, y, z) des articulations clés du bras droit et du buste
- Sauvegarder les résultats dans data/processed/raw_landmarks.csv
"""

import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import mediapipe as mp

mp_pose = mp.solutions.pose

# Articulations utiles pour analyser un coup de tennis de table
LANDMARK_IDS = {
    "R_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "R_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "R_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "L_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "R_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "L_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
}

def extract_from_video(video_path: str, label: str, video_id: str) -> list[dict]:
    """
    Lit une vidéo et extrait les coordonnées des repères pour chaque frame.
    
    À faire :
    1. Ouvrir le flux vidéo avec cv2.VideoCapture(video_path)
    2. Initialiser MediaPipe Pose dans un contexte 'with'
    3. Boucler sur les frames tant que la vidéo est ouverte (cap.isOpened())
    4. Convertir chaque frame en RGB
    5. Appliquer pose.process() et récupérer results.pose_landmarks
    6. Pour chaque frame détectée, construire un dictionnaire avec :
       - video_id, label, frame_idx
       - {name}_x, {name}_y, {name}_z pour chaque point dans LANDMARK_IDS
    7. Ne pas oublier de libérer la vidéo avec cap.release()
    8. Retourner la liste des dictionnaires (une entrée par frame)
    """
    records = []
    # TODO: À toi de coder l'extraction ici !
    return records

def run_extraction(raw_dir: str = "data/raw_videos", output_csv: str = "data/processed/raw_landmarks.csv"):
    """
    Parcourt l'ensemble des sous-dossiers de raw_dir, extrait les landmarks
    et enregistre le DataFrame final dans output_csv.
    
    À faire :
    1. Créer le dossier parent de output_csv s'il n'existe pas (os.makedirs)
    2. Parcourir les labels (dossiers dans raw_dir)
    3. Pour chaque vidéo (.mp4, .mov, etc.), appeler extract_from_video(...)
    4. Créer un pd.DataFrame à partir de toutes les données extraites
    5. Sauvegarder en CSV sans index (to_csv)
    """
    # TODO: À toi de coder le parcours des dossiers et la sauvegarde CSV !
    pass

if __name__ == "__main__":
    run_extraction()
