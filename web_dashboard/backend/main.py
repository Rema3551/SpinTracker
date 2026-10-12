# main.py - FastAPI backend for SpinTracker dashboard

"""
FastAPI server qui gère :
1. L'upload de vidéos de match complet (10-30 min) ou clips.
2. L'identification du joueur cible (joueur au premier plan 'near' ou fond de court 'far').
3. Le suivi des changements de côté entre les sets.
4. L'analyse des coups (TopSpins, Poussettes, Blocs), le calcul des réussites, fautes et points gagnants (Winners).
"""

import os
import sys
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ajout de la racine du projet dans sys.path pour importer src/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.match_analyzer import analyze_match_video
from src.predict import predict_video

app = FastAPI(title="SpinTracker Match Analytics API", version="2.0.0")

# Support CORS pour React en local et sur Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", response_class=JSONResponse)
async def upload_match_video(
    file: UploadFile = File(...),
    player_side: str = Form("near"),       # 'near' (premier plan) ou 'far' (fond de court)
    analysis_mode: str = Form("match")     # 'match' (analyse complète) ou 'clip' (un seul geste)
):
    """
    Traite la vidéo envoyée et retourne le bilan statistique complet :
    - Nombre de TopSpins, Poussettes, Blocs.
    - Pourcentage de réussite.
    - Points gagnants (Winners) & Fautes directes.
    - Évolution set par set avec prise en compte des changements de côté.
    """
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Format vidéo non supporté. Utilisez MP4, MOV, MKV ou AVI.")

    unique_id = uuid.uuid4().hex[:10]
    safe_filename = f"{unique_id}_{os.path.basename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Écriture du fichier sur disque par morceaux (optimisé pour vidéos lourdes)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Mode d'analyse sélectionné
        if analysis_mode == "clip":
            result = predict_video(file_path, return_results=True)
            if not result:
                raise HTTPException(status_code=500, detail="Échec de l'inférence sur le clip.")
            return JSONResponse(content={"mode": "clip", "result": result})

        # Mode par défaut : Analyse complète de match
        match_stats = analyze_match_video(
            video_path=file_path,
            initial_player_side=player_side,
            model_path=os.path.join(ROOT_DIR, "data", "processed", "model.joblib")
        )

        return JSONResponse(content={
            "mode": "match",
            "filename": file.filename,
            "player_initial_side": player_side,
            **match_stats
        })

    except Exception as e:
        print(f"[ERROR] Erreur pendant l'analyse : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse vidéo : {str(e)}")

    finally:
        # Nettoyage du fichier temporaire si souhaité
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0", "service": "SpinTracker API"}
