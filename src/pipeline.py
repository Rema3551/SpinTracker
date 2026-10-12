"""
Chef d'orchestre : Exécution automatique de la pipeline SpinTracker.

Ce script permet d'enchaîner séquentiellement :
1. 01_extract_landmarks.py
2. 02_build_features.py
3. 03_train_model.py
"""

import os
import sys
import importlib

# S'assurer que stdout supporte l'UTF-8 sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# S'assurer que le dossier 'src' est présent dans sys.path pour les imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import dynamique des étapes
extract_mod = importlib.import_module("01_extract_landmarks")
features_mod = importlib.import_module("02_build_features")
train_mod = importlib.import_module("03_train_model")

run_extraction = extract_mod.run_extraction
process_features = features_mod.process_features
train_classifier = train_mod.train_classifier

def main():
    # Se positionner à la racine du projet
    os.chdir(PROJECT_ROOT)
    
    print("========================================")
    print("Demarrage de la pipeline SpinTracker...")
    print("========================================")
    
    print("\n[1/3] Extraction des reperes MediaPipe...")
    run_extraction()

    print("\n[2/3] Calcul des caracteristiques biomecaniques...")
    process_features()

    print("\n[3/3] Entrainement et evaluation du modele...")
    train_classifier()

    print("\n========================================")
    print("Pipeline terminee avec succes !")
    print("========================================")

if __name__ == "__main__":
    main()
