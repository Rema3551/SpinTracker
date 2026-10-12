"""
Script utilitaire pour l'Étape 0 : Collecte et Découpage des vidéos d'entraînement.

Ce script permet de télécharger des séquences réelles de tennis de table sous licence
Creative Commons (Wikimedia Commons) et de découper des clips courts (1 geste par clip)
dans data/raw_videos/topspin/, data/raw_videos/poussette/ et data/raw_videos/bloc/.
"""

import os
import urllib.request
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")

def download_wikimedia_video(file_title: str, output_path: str):
    print(f"Téléchargement de {file_title}...")
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(file_title)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(api_url, headers={"User-Agent": "SpinTracker/1.0 (educational table tennis AI project)"})
    data = json.loads(urllib.request.urlopen(req).read())
    page = list(data["query"]["pages"].values())[0]
    media_url = page["imageinfo"][0]["url"]
    
    req_media = urllib.request.Request(media_url, headers={"User-Agent": "SpinTracker/1.0 (educational table tennis AI project)"})
    with open(output_path, "wb") as f:
        f.write(urllib.request.urlopen(req_media).read())
    print(f"Téléchargé -> {output_path}")

def export_clip(src_video: str, start_sec: float, end_sec: float, out_path: str, speed: float = 1.0, reverse: bool = False):
    cap = cv2.VideoCapture(src_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(max(1, end_f - start_f)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        return
        
    if reverse:
        frames = frames[::-1]
        
    if speed != 1.0:
        indices = np.linspace(0, len(frames) - 1, int(len(frames) / speed)).astype(int)
        frames = [frames[i] for i in indices]
        
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Clip généré : {out_path} ({len(frames)} frames)")

def main():
    os.makedirs(os.path.join(RAW_DIR, "topspin"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "poussette"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "bloc"), exist_ok=True)
    print("Dossiers prêts dans data/raw_videos/.")

if __name__ == "__main__":
    main()
