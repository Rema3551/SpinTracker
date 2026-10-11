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

# S'assurer que le dossier 'src' est présent dans sys.path pour les imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Note : On utilise importlib car les noms de modules débutant par un chiffre
# ne peuvent pas être importés directement avec 'from 01_... import ...' en syntaxe Python standard.
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
    print("🏓 Démarrage de la pipeline SpinTracker...")
    print("========================================")
    
    print("\n[1/3] Extraction des repères MediaPipe...")
    run_extraction()

    print("\n[2/3] Calcul des caractéristiques biomécaniques...")
    process_features()

    print("\n[3/3] Entraînement et évaluation du modèle...")
    train_classifier()

    print("\n========================================")
    print("✅ Pipeline terminée avec succès !")
    print("========================================")

if __name__ == "__main__":
    main()
