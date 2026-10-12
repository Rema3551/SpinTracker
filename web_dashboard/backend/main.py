# main.py - FastAPI backend for SpinTracker dashboard

"""FastAPI server that accepts a video upload, runs the existing
`predict_video` function (with `return_results=True`) and returns a JSON
payload with the prediction, class probabilities and the physical metrics.
"""

import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import the updated predict function from the project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.predict import predict_video

app = FastAPI(title="SpinTracker Dashboard API", version="0.1.0")

# Allow the frontend (running on localhost:5173 by default) to access the API
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
async def upload_video(file: UploadFile = File(...)):
    """Save the uploaded video to a temporary location, run prediction and
    return the structured results.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Generate a unique filename to avoid collisions
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Write the uploaded content to disk
    with open(file_path, "wb") as out_file:
        while content := await file.read(1024 * 1024):
            out_file.write(content)

    # Run the prediction – we request a structured dict
    result = predict_video(file_path, return_results=True)

    # Clean up the uploaded file to save space (optional)
    try:
        os.remove(file_path)
    except OSError:
        pass

    if result is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return JSONResponse(content=result)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
