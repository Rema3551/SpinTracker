"""
Visualisation biomécanique en vidéo avec superposition du squelette et de la trajectoire.

Fonctionnalités :
- Détection MediaPipe Pose et tracé du squelette (épaules, coudes, poignets, hanches)
- Tracé de la traînée de trajectoire du poignet (effet lumineux)
- Affichage de l'angle du coude en temps réel
- Bandeau HUD semi-transparent avec la prédiction IA et le pourcentage de confiance
- Export de la vidéo annotée en .mp4 dans data/processed/

Usage :
    python src/visualize.py data/raw_videos/topspin/topspin_01.mp4
    python src/visualize.py data/raw_videos/poussette/poussette_01.mp4
    python src/visualize.py data/raw_videos/bloc/bloc_01.mp4
"""

import os
import sys
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LANDMARK_IDS = {
    "R_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "R_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "R_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "L_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "R_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "L_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
}

CLASS_COLORS = {
    "topspin": (46, 204, 113),     # Vert émeraude
    "poussette": (241, 196, 15),   # Jaune / Ambre
    "bloc": (52, 152, 219),        # Bleu ciel
    "inconnu": (189, 195, 199)     # Gris clair
}

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.dot(ba, bc) / norm_product
    angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(angle)))

def compute_overall_features(frames_landmarks: list[dict]) -> tuple[str, float]:
    """Charge le modèle et prédit la classe et la confiance de la vidéo complète."""
    model_path = "data/processed/model.joblib"
    if not os.path.exists(model_path) or len(frames_landmarks) < 3:
        return "inconnu", 0.0

    df = pd.DataFrame(frames_landmarks)
    wrist_x = df["R_WRIST_x"].values
    wrist_y = df["R_WRIST_y"].values

    wrist_delta_y = float(wrist_y[0] - wrist_y[-1])
    wrist_delta_x = float(wrist_x[-1] - wrist_x[0])

    velocities = np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2)
    wrist_max_velocity = float(np.max(velocities)) if len(velocities) > 0 else 0.0

    angles = []
    for _, row in df.iterrows():
        s = np.array([row["R_SHOULDER_x"], row["R_SHOULDER_y"]])
        e = np.array([row["R_ELBOW_x"], row["R_ELBOW_y"]])
        w = np.array([row["R_WRIST_x"], row["R_WRIST_y"]])
        angles.append(calculate_angle(s, e, w))

    X_sample = pd.DataFrame([{
        "wrist_delta_y": wrist_delta_y,
        "wrist_delta_x": wrist_delta_x,
        "wrist_max_velocity": wrist_max_velocity,
        "elbow_angle_min": float(np.min(angles)),
        "elbow_angle_max": float(np.max(angles)),
        "duration_frames": int(len(df))
    }])

    clf = joblib.load(model_path)
    pred = clf.predict(X_sample)[0]
    prob = float(np.max(clf.predict_proba(X_sample)[0]))
    return pred, prob

def generate_annotated_video(input_video_path: str, output_video_path: str = None, roi: tuple = (0.0, 0.0, 1.0, 1.0)) -> str:
    if not os.path.exists(input_video_path):
        print(f"Erreur : Vidéo source introuvable -> {input_video_path}")
        return None

    if output_video_path is None:
        os.makedirs("data/processed", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = f"data/processed/annotated_{base_name}.mp4"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Erreur d'ouverture vidéo : {input_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rx1, ry1, rx2, ry2 = roi
    x_min_px = int(rx1 * width)
    y_min_px = int(ry1 * height)
    x_max_px = int(rx2 * width)
    y_max_px = int(ry2 * height)

    # Passe 1 : Récupérer tous les landmarks pour prédire le geste global
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    all_frames_data = []

    print(f"Analyse de '{os.path.basename(input_video_path)}' ({total_frames} frames)...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if roi != (0.0, 0.0, 1.0, 1.0):
            proc_frame = frame[y_min_px:y_max_px, x_min_px:x_max_px]
        else:
            proc_frame = frame

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        row = {}
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            for name, idx in LANDMARK_IDS.items():
                lx = lm[idx].x
                ly = lm[idx].y
                if roi != (0.0, 0.0, 1.0, 1.0):
                    row[f"{name}_x"] = rx1 + lx * (rx2 - rx1)
                    row[f"{name}_y"] = ry1 + ly * (ry2 - ry1)
                else:
                    row[f"{name}_x"] = lx
                    row[f"{name}_y"] = ly
                row[f"{name}_z"] = lm[idx].z

            if roi != (0.0, 0.0, 1.0, 1.0):
                for l in res.pose_landmarks.landmark:
                    l.x = rx1 + l.x * (rx2 - rx1)
                    l.y = ry1 + l.y * (ry2 - ry1)

            row["pose_landmarks"] = res.pose_landmarks
        else:
            row["pose_landmarks"] = None
        all_frames_data.append(row)
        frame_count += 1
        if frame_count % 300 == 0:
            print(f"  -> Analyse : {frame_count}/{total_frames} frames...")

    cap.release()
    pose.close()

    # Déterminer la prédiction globale
    valid_landmarks = [f for f in all_frames_data if "R_WRIST_x" in f]
    pred_label, pred_confidence = compute_overall_features(valid_landmarks)
    badge_color = CLASS_COLORS.get(pred_label, (200, 200, 200))

    # Passe 2 : Rendu vidéo avec overlay en streaming (sans charger toutes les images en RAM)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_out_path = output_video_path.replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    cap2 = cv2.VideoCapture(input_video_path)
    wrist_trail = []
    prev_wrist = None

    print(f"Rendu de la vidéo annotée ({len(all_frames_data)} frames)...")
    for idx, data in enumerate(all_frames_data):
        ret, frame = cap2.read()
        if not ret:
            break
        annotated = frame
        
        # 1. Tracé du squelette MediaPipe
        landmarks = data.get("pose_landmarks")
        elbow_angle_str = "--"
        wrist_speed_str = "--"

        if landmarks is not None:
            # Dessiner les connexions principales
            mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Calculer les coordonnées en pixels
            lm = landmarks.landmark
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            rw_px = (int(rw.x * width), int(rw.y * height))
            re_px = (int(re.x * width), int(re.y * height))
            rs_px = (int(rs.x * width), int(rs.y * height))

            # Angle du coude
            angle = calculate_angle(
                np.array([rs.x, rs.y]),
                np.array([re.x, re.y]),
                np.array([rw.x, rw.y])
            )
            elbow_angle_str = f"{angle:.0f}°"

            # Vitesse instantanée
            if prev_wrist is not None:
                speed = np.sqrt((rw.x - prev_wrist[0])**2 + (rw.y - prev_wrist[1])**2)
                wrist_speed_str = f"{speed:.3f}"
            prev_wrist = (rw.x, rw.y)

            # Ajouter à la traînée du poignet
            wrist_trail.append(rw_px)
            if len(wrist_trail) > 25:
                wrist_trail.pop(0)

            # Afficher l'angle près du coude
            cv2.putText(annotated, elbow_angle_str, (re_px[0] + 10, re_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 2. Dessiner la traînée lumineuse du poignet
        for i in range(1, len(wrist_trail)):
            thickness = int(np.interp(i, [1, len(wrist_trail)], [1, 6]))
            alpha_ratio = i / len(wrist_trail)
            trail_color = (
                int(badge_color[0] * alpha_ratio),
                int(badge_color[1] * alpha_ratio),
                int(badge_color[2] * alpha_ratio)
            )
            cv2.line(annotated, wrist_trail[i - 1], wrist_trail[i], trail_color, thickness)
            cv2.circle(annotated, wrist_trail[i], max(1, thickness - 1), (255, 255, 255), -1)

        # 3. HUD semi-transparent en haut de l'image
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (10, 10), (width - 10, 95), (25, 25, 25), -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)

        # Bordure du HUD aux couleurs de la classe
        cv2.rectangle(annotated, (10, 10), (width - 10, 95), badge_color, 2)

        # Textes du HUD
        title_text = f"GESTURE : {pred_label.upper()} ({pred_confidence * 100:.1f}%)"
        cv2.putText(annotated, title_text, (25, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, badge_color, 2)

        metrics_text = f"Coude: {elbow_angle_str}  |  Vitesse poignet: {wrist_speed_str}  |  Frame: {idx + 1}/{len(all_frames_data)}"
        cv2.putText(annotated, metrics_text, (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # 4. Barre de progression en bas
        progress_w = int((idx + 1) / len(all_frames_data) * (width - 20))
        cv2.rectangle(annotated, (10, height - 12), (width - 10, height - 6), (50, 50, 50), -1)
        cv2.rectangle(annotated, (10, height - 12), (10 + progress_w, height - 6), badge_color, -1)

        out.write(annotated)

    cap2.release()
    out.release()

    # Re-compression propre en H.264 via ffmpeg pour compatibilité universelle
    cmd = f'ffmpeg -y -i "{temp_out_path}" -c:v libx264 -pix_fmt yuv420p "{output_video_path}"'
    ret_code = os.system(cmd)
    if os.path.exists(temp_out_path):
        try:
            os.remove(temp_out_path)
        except Exception:
            pass

    final_path = output_video_path if ret_code == 0 and os.path.exists(output_video_path) else temp_out_path
    print(f"\n[OK] Video annotee generee avec succes -> {final_path}")
    return final_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualisation biomecanique avec squelette et telemetrie.")
    parser.add_argument("video", nargs="?", default="data/raw_videos/topspin/topspin_01.mp4", help="Video a annoter")
    parser.add_argument("--roi", type=str, default="0.0,0.0,1.0,1.0", help="Zone d'interet xmin,ymin,xmax,ymax")
    args = parser.parse_args()

    roi_parts = [float(x.strip()) for x in args.roi.split(",")]
    roi_tuple = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])

    generate_annotated_video(args.video, roi=roi_tuple)
