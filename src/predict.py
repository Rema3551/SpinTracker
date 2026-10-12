"""
Script d'inférence : Prédire le coup technique d'une vidéo arbitraire.

Usage :
    python src/predict.py data/raw_videos/topspin/topspin_01.mp4
    python src/predict.py mon_coup.mp4
"""

import os
import sys
import joblib
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose

LANDMARK_IDS = {
    "R_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "R_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "R_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "L_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "R_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "L_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
}

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.dot(ba, bc) / norm_product
    angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(angle)))

def predict_video(video_path: str, model_path: str = "data/processed/model.joblib", return_results: bool = False):
    if not os.path.exists(video_path):
        print(f"Erreur : Vidéo introuvable -> {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Erreur : Modèle introuvable -> {model_path}. Exécute d'abord pipeline.py !")
        return

    # 1. Charger le modèle
    clf = joblib.load(model_path)

    # 2. Extraire les repères avec MediaPipe
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo -> {video_path}")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            row = {}
            for name, idx in LANDMARK_IDS.items():
                row[f"{name}_x"] = lm[idx].x
                row[f"{name}_y"] = lm[idx].y
                row[f"{name}_z"] = lm[idx].z
            frames_data.append(row)

    cap.release()
    pose.close()

    if len(frames_data) < 3:
        print("Erreur : Trop peu de repères détectés dans la vidéo pour analyser le geste.")
        return

    # 3. Calculer les caractéristiques cinématiques
    df = pd.DataFrame(frames_data)
    wrist_x = df["R_WRIST_x"].values
    wrist_y = df["R_WRIST_y"].values

    wrist_delta_y = float(wrist_y[0] - wrist_y[-1])
    wrist_delta_x = float(wrist_x[-1] - wrist_x[0])

    velocities = np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2)
    wrist_max_velocity = float(np.max(velocities)) if len(velocities) > 0 else 0.0

    angles = []
    for _, row in df.iterrows():
        shoulder = np.array([row["R_SHOULDER_x"], row["R_SHOULDER_y"]])
        elbow = np.array([row["R_ELBOW_x"], row["R_ELBOW_y"]])
        wrist = np.array([row["R_WRIST_x"], row["R_WRIST_y"]])
        angles.append(calculate_angle(shoulder, elbow, wrist))

    elbow_angle_min = float(np.min(angles))
    elbow_angle_max = float(np.max(angles))
    duration_frames = int(len(df))

    features = [
        "wrist_delta_y", "wrist_delta_x", "wrist_max_velocity",
        "elbow_angle_min", "elbow_angle_max", "duration_frames"
    ]
    X_sample = pd.DataFrame([{
        "wrist_delta_y": wrist_delta_y,
        "wrist_delta_x": wrist_delta_x,
        "wrist_max_velocity": wrist_max_velocity,
        "elbow_angle_min": elbow_angle_min,
        "elbow_angle_max": elbow_angle_max,
        "duration_frames": duration_frames
    }])[features]

    # 4. Prédiction
    pred = clf.predict(X_sample)[0]
    probabilities = clf.predict_proba(X_sample)[0]
    classes = clf.classes_

    # 5. Affichage clair des résultats
    print("\n" + "="*45)
    print(f"Analyse de la video : {os.path.basename(video_path)}")
    print("="*45)
    print("Metriques physiques calculees :")
    print(f"  - Delta vertical poignet (delta_y) : {wrist_delta_y:+.3f} ({'monte' if wrist_delta_y > 0 else 'descend'})")
    print(f"  - Delta horizontal poignet (delta_x) : {wrist_delta_x:+.3f}")
    print(f"  - Vitesse maximale poignet          : {wrist_max_velocity:.3f}")
    print(f"  - Angle coude min / max             : {elbow_angle_min:.1f} deg / {elbow_angle_max:.1f} deg")
    print(f"  - Duree utile                       : {duration_frames} frames")
    print("-"*45)
    print(f"PREDICTION FINALE : >>> {pred.upper()} <<<")
    print("Probabilites :")
    for cls, prob in zip(classes, probabilities):
        bar = "#" * int(prob * 20)
        print(f"  - {cls:<10} : {prob*100:5.1f}% | {bar}")
    print("="*45 + "\n")
    if return_results:
        return {
            "prediction": pred,
            "probabilities": dict(zip(classes, probabilities.tolist())),
            "metrics": {
                "wrist_delta_y": wrist_delta_y,
                "wrist_delta_x": wrist_delta_x,
                "wrist_max_velocity": wrist_max_velocity,
                "elbow_angle_min": elbow_angle_min,
                "elbow_angle_max": elbow_angle_max,
                "duration_frames": duration_frames
            }
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "data/raw_videos/topspin/topspin_01.mp4"
    predict_video(target)
