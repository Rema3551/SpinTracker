"""
Étape 1 : Extraction des repères anatomiques (Pose Landmarks) avec MediaPipe.

Objectif :
- Parcourir chaque vidéo dans data/raw_videos/<label>/
- Extraire frame par frame les coordonnées (x, y, z) des articulations clés du bras droit et du buste
- Sauvegarder les résultats dans data/processed/raw_landmarks.csv
"""

import os
import cv2
import pandas as pd
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
    Retourne 1 dictionnaire par frame contenant les coordonnées des repères clés.
    """
    records = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return records

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = {
                "video_id": video_id,
                "label": label,
                "frame_idx": frame_idx
            }
            for name, idx in LANDMARK_IDS.items():
                row[f"{name}_x"] = landmarks[idx].x
                row[f"{name}_y"] = landmarks[idx].y
                row[f"{name}_z"] = landmarks[idx].z
            records.append(row)

        frame_idx += 1

    video.release()
    pose.close()
    return records

def run_extraction(raw_dir: str = "data/raw_videos", output_csv: str = "data/processed/raw_landmarks.csv"):
    """
    Parcourt l'ensemble des sous-dossiers de raw_dir, extrait les landmarks
    et enregistre le DataFrame final dans output_csv.
    """
    parent_dir = os.path.dirname(output_csv)
    os.makedirs(parent_dir, exist_ok=True)
    all_records = []

    for label in os.listdir(raw_dir):
        label_dir = os.path.join(raw_dir, label)
        if os.path.isdir(label_dir):
            for video_id in os.listdir(label_dir):
                if video_id.lower().endswith((".mp4", ".mov", ".avi")):
                    video_path = os.path.join(label_dir, video_id)
                    if os.path.isfile(video_path):
                        print(f"[01] Traitement de : {video_path} (classe: {label})")
                        records = extract_from_video(video_path, label, video_id)
                        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"[01] Extraction terminée : {len(df)} frames extraites -> {output_csv}")

if __name__ == "__main__":
    run_extraction()
