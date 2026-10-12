"""
Découpeur automatique de vidéos longues (Match / Entraînement).

Principe :
1. Analyse le flux vidéo continu avec MediaPipe Pose.
2. Calcule la courbe de vitesse instantanée du poignet au cours du temps.
3. Détecte les pics d'accélération (impacts de balle / gestes actifs).
4. Découpe automatiquement chaque geste en clip court (armé + impact + accompagnement).
5. (Optionnel) Classe automatiquement chaque geste avec le modèle IA entraîné.
6. Génère un graphique du profil de vitesse avec les pics détectés.

Usage :
    python src/auto_cutter.py data/sample_long_match.mp4
    python src/auto_cutter.py match_10min.mp4 --output-dir data/auto_clips
"""

import os
import sys

# Compatibilité encodage console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

def predict_clip_features(landmarks_list: list[dict], model_path: str = "data/processed/model.joblib") -> tuple[str, float]:
    """Prédit la classe et la probabilité d'un clip extrait."""
    if not os.path.exists(model_path) or len(landmarks_list) < 3:
        return "inconnu", 0.0

    df = pd.DataFrame(landmarks_list)
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

    try:
        clf = joblib.load(model_path)
        pred = clf.predict(X_sample)[0]
        prob = float(np.max(clf.predict_proba(X_sample)[0]))
        return pred, prob
    except Exception:
        return "inconnu", 0.0

def process_long_video(
    video_path: str,
    output_dir: str = "data/auto_clips",
    min_speed: float = 0.06,
    min_distance_sec: float = 1.0,
    pre_impact_sec: float = 0.6,
    post_impact_sec: float = 0.7,
    roi: tuple = (0.0, 0.0, 1.0, 1.0),
    save_plot: bool = True
):
    if not os.path.exists(video_path):
        print(f"Erreur : Video introuvable -> {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la video -> {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rx1, ry1, rx2, ry2 = roi
    x_min_px = int(rx1 * frame_w)
    y_min_px = int(ry1 * frame_h)
    x_max_px = int(rx2 * frame_w)
    y_max_px = int(ry2 * frame_h)

    print("\n" + "="*50)
    print(f"[DECOUPAGE AUTO] Analyse de : {os.path.basename(video_path)}")
    print(f"  Duree : {duration_sec:.1f}s | FPS : {fps:.1f} | Frames : {total_frames}")
    if roi != (0.0, 0.0, 1.0, 1.0):
        print(f"  Zone d'interet (ROI) : x=[{rx1:.2f}, {rx2:.2f}], y=[{ry1:.2f}, {ry2:.2f}]")
    print("="*50)

    # 1. Extraction des coordonnées et calcul du signal de mouvement de table
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    
    wrist_coords = []
    landmarks_per_frame = []
    table_motion = []
    prev_table_gray = None

    print("[1/3] Analyse MediaPipe et suivi des echanges sur la table...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mesure d'activite sur la table de ping-pong (detection d'echanges)
        table_crop = frame[int(frame_h * 0.35):int(frame_h * 0.65), int(frame_w * 0.05):int(frame_w * 0.55)]
        t_gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        if prev_table_gray is not None:
            t_diff = cv2.absdiff(t_gray, prev_table_gray)
            _, t_thresh = cv2.threshold(t_diff, 20, 255, cv2.THRESH_BINARY)
            table_motion.append(np.sum(t_thresh > 0))
        else:
            table_motion.append(0)
        prev_table_gray = t_gray

        # Appliquer le ROI si défini
        if roi != (0.0, 0.0, 1.0, 1.0):
            proc_frame = frame[y_min_px:y_max_px, x_min_px:x_max_px]
        else:
            proc_frame = frame

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            # Remapper les coordonnées si ROI
            if roi != (0.0, 0.0, 1.0, 1.0):
                norm_x = rx1 + rw.x * (rx2 - rx1)
                norm_y = ry1 + rw.y * (ry2 - ry1)
            else:
                norm_x = rw.x
                norm_y = rw.y

            wrist_coords.append((frame_idx, norm_x, norm_y))

            frame_dict = {}
            for name, idx in LANDMARK_IDS.items():
                lx = lm[idx].x
                ly = lm[idx].y
                if roi != (0.0, 0.0, 1.0, 1.0):
                    frame_dict[f"{name}_x"] = rx1 + lx * (rx2 - rx1)
                    frame_dict[f"{name}_y"] = ry1 + ly * (ry2 - ry1)
                else:
                    frame_dict[f"{name}_x"] = lx
                    frame_dict[f"{name}_y"] = ly
                frame_dict[f"{name}_z"] = lm[idx].z
            landmarks_per_frame.append(frame_dict)
        else:
            wrist_coords.append((frame_idx, np.nan, np.nan))
            landmarks_per_frame.append({})

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  -> Analyse : {frame_idx}/{total_frames} frames ({frame_idx/fps:.1f}s)...")

    cap.release()
    pose.close()

    # Reconstituer le signal de vitesse
    wrist_x = np.array([c[1] for c in wrist_coords])
    wrist_y = np.array([c[2] for c in wrist_coords])

    valid_mask = ~np.isnan(wrist_x)
    if np.sum(valid_mask) < 10:
        print("Erreur : Trop peu de detections de joueur sur la video.")
        return

    indices = np.arange(len(wrist_x))
    wrist_x = np.interp(indices, indices[valid_mask], wrist_x[valid_mask])
    wrist_y = np.interp(indices, indices[valid_mask], wrist_y[valid_mask])

    # Vitesse instantanée
    speeds = np.zeros(len(wrist_x))
    speeds[1:] = np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2)

    # Lissage par moyenne mobile (5 frames)
    window_size = 5
    smoothed_speeds = np.convolve(speeds, np.ones(window_size)/window_size, mode='same')

    # Filtrage des echanges actifs sur la table (elimination des temps morts / marches)
    table_motion = np.array(table_motion)
    win_m = max(5, int(0.4 * fps))
    smooth_motion = np.convolve(table_motion, np.ones(win_m)/win_m, mode='same')
    motion_thresh = np.percentile(smooth_motion, 65)
    is_rally = smooth_motion > motion_thresh

    # Signal de frappe balle-raquette : vitesse du poignet ponderee par l'activite de la table
    strike_signal = smoothed_speeds * is_rally

    # 2. Détection des contacts balle-raquette
    min_distance_frames = int(min_distance_sec * fps)
    peaks, _ = find_peaks(
        strike_signal,
        height=min_speed,
        distance=min_distance_frames,
        prominence=min_speed * 0.35
    )

    print(f"\n[2/3] Detection des impacts balle-raquette : {len(peaks)} contact(s) valide(s) !")

    # 3. Découpage et export des clips
    pre_frames = int(pre_impact_sec * fps)
    post_frames = int(post_impact_sec * fps)

    model_path = "data/processed/model.joblib"
    has_model = os.path.exists(model_path)
    extracted_clips = []

    print("\n[3/3] Export des sequences decoupees :")
    for i, peak_idx in enumerate(peaks):
        peak_time = peak_idx / fps
        start_f = max(0, peak_idx - pre_frames)
        end_f = min(len(landmarks_per_frame), peak_idx + post_frames)

        clip_landmarks = [landmarks_per_frame[f] for f in range(start_f, end_f) if landmarks_per_frame[f]]

        # Prédiction du coup si le modèle est disponible
        pred_label, pred_prob = "inconnu", 0.0
        if has_model and len(clip_landmarks) >= 3:
            pred_label, pred_prob = predict_clip_features(clip_landmarks, model_path)

        clip_name = f"{base_name}_hit_{i+1:02d}_{peak_time:.2f}s_{pred_label}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        start_time_sec = max(0.0, peak_time - pre_impact_sec)
        clip_duration = pre_impact_sec + post_impact_sec

        # Découpage ultra-rapide et propre avec ffmpeg via subprocess
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", f"{start_time_sec:.3f}",
                "-i", video_path,
                "-t", f"{clip_duration:.3f}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                clip_path,
                "-loglevel", "error"
            ], check=True)
        except Exception as e:
            print(f"    [WARN] Erreur export ffmpeg : {e}")

        extracted_clips.append({
            "id": i + 1,
            "peak_time": peak_time,
            "start_time": start_time_sec,
            "end_time": start_time_sec + clip_duration,
            "peak_speed": smoothed_speeds[peak_idx],
            "pred_label": pred_label,
            "confidence": pred_prob,
            "path": clip_path
        })

        pred_str = f"-> {pred_label.upper()} ({pred_prob*100:.1f}%)" if has_model else ""
        print(f"  * Impact #{i+1:02d} a {peak_time:5.2f}s | Vitesse: {smoothed_speeds[peak_idx]:.3f} {pred_str}")
        print(f"    Fichier : {clip_path}")

    # 4. Tracé graphique du profil de vitesse
    if save_plot:
        plot_path = os.path.join("data", "processed", f"velocity_profile_{base_name}.png")
        times = indices / fps
        plt.figure(figsize=(13, 5))
        # Zone des échanges réels sur la table en surbrillance
        plt.fill_between(times, 0, max(0.2, float(np.max(smoothed_speeds)*1.15)), where=is_rally, color="#2ecc71", alpha=0.15, label="Echanges actifs sur la table (Rally Gate)")
        plt.plot(times, speeds, color="lightgray", label="Vitesse brute (frame par frame)", alpha=0.5)
        plt.plot(times, smoothed_speeds, color="#2980b9", linewidth=1.5, label="Vitesse poignet lissee")
        plt.plot(times, strike_signal, color="#e67e22", linewidth=2, label="Signal frappe valide (Pendant echange)")

        # Marquer le seuil
        plt.axhline(y=min_speed, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Seuil de frappe ({min_speed})")

        # Marquer les vrais impacts balle-raquette détectés
        for clip in extracted_clips:
            t = clip["peak_time"]
            v = clip["peak_speed"]
            label = clip["pred_label"]
            plt.plot(t, v, "ro", markersize=9)
            plt.annotate(
                f"Impact #{clip['id']}\n{t:.1f}s | {label.upper()}",
                xy=(t, v),
                xytext=(t, v + 0.025),
                ha="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f1c40f", ec="#d35400", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5)
            )

        plt.title(f"Detection chirurgicale des impacts balle-raquette : {os.path.basename(video_path)}", fontsize=12, fontweight="bold")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Vitesse instantanee du poignet")
        plt.ylim(0, max(0.22, float(np.max(smoothed_speeds)*1.25)))
        plt.legend(loc="upper right", framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n[GRAPH] Graphique de vitesse enregistre -> {plot_path}")

    print("\n" + "="*50)
    print(f"[TERMINE] {len(extracted_clips)} clips crees dans '{output_dir}'.")
    print("="*50 + "\n")
    return extracted_clips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoupeur automatique de coups de tennis de table.")
    parser.add_argument("video", nargs="?", default="data/sample_long_match.mp4", help="Chemin de la video a decouper")
    parser.add_argument("--output-dir", default="data/auto_clips", help="Dossier ou enregistrer les clips decoupes")
    parser.add_argument("--min-speed", type=float, default=0.06, help="Seuil minimal de vitesse (defaut: 0.06)")
    parser.add_argument("--min-gap", type=float, default=1.0, help="Intervalle minimal en secondes entre 2 coups (defaut: 1.0s)")
    parser.add_argument("--roi", type=str, default="0.0,0.0,1.0,1.0", help="Zone d'interet normalisee xmin,ymin,xmax,ymax (ex: 0,0,0.55,1.0 pour isoler le joueur gauche)")
    args = parser.parse_args()

    roi_parts = [float(x.strip()) for x in args.roi.split(",")]
    roi_tuple = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])

    process_long_video(
        video_path=args.video,
        output_dir=args.output_dir,
        min_speed=args.min_speed,
        min_distance_sec=args.min_gap,
        roi=roi_tuple
    )
