# 🏓 SpinTracker

> Reconnaissance et analyse biomécanique des gestes techniques de tennis de table à partir de vidéos par ordinateur.

---

## 📖 Sommaire

- [Présentation du projet](#-présentation-du-projet)
- [Architecture du projet](#-architecture-du-projet)
- [Installation et démarrage rapide](#-installation-et-démarrage-rapide)
- [Outils et Fonctionnalités](#-outils-et-fonctionnalités)
  - [1. Découpage automatique de vidéos longues](#1-découpage-automatique-de-vidéos-longues)
  - [2. Prédiction d'un coup](#2-prédiction-dun-coup)
  - [3. Visualisation vidéo avec squelette incrusté](#3-visualisation-vidéo-avec-squelette-incrusté)
- [Guide de développement complet](#-guide-de-développement-complet)

---

## 🎯 Présentation du projet

SpinTracker a pour but de classifier automatiquement des coups techniques de tennis de table (**Topspin**, **Poussette**, **Bloc**) à partir d'extraits vidéo enregistrés avec une caméra classique (smartphone ou webcam).

Le système repose sur 3 piliers :
1. **Vision par ordinateur** : Détection des repères du corps humain (épaules, coude, poignet, hanches) avec **MediaPipe Pose**.
2. **Ingénierie de caractéristiques** : Calcul de descripteurs biomécaniques (vitesse maximale du poignet, dénivelé vertical, angles articulaires du bras).
3. **Apprentissage automatique** : Classification supervisée avec un modèle **Random Forest**.

---

## 🏗️ Architecture du projet

```text
SpinTracker/
├── data/
│   ├── raw_videos/              # Extraits .mp4 classés par dossier (topspin/, poussette/, bloc/)
│   ├── auto_clips/              # Clips automatiquement découpés depuis des vidéos longues
│   └── processed/               # Données extraites, modèles et vidéos annotées
│       ├── raw_landmarks.csv    # Coordonnées des repères frame par frame
│       ├── dataset.csv          # 1 ligne par vidéo avec features calculées
│       ├── model.joblib         # Modèle Random Forest sauvegardé
│       ├── velocity_profile_*.png # Profils cinématiques et détection des impacts
│       └── annotated_*.mp4      # Vidéos exportées avec squelette et HUD
├── src/
│   ├── 01_extract_landmarks.py  # Étape 1 : Extraction frame par frame via MediaPipe Pose
│   ├── 02_build_features.py     # Étape 2 : Calcul des métriques physiques et agrégation
│   ├── 03_train_model.py        # Étape 3 : Entraînement Random Forest et métriques
│   ├── pipeline.py              # Orchestrateur qui exécute 01 -> 02 -> 03
│   ├── auto_cutter.py           # Découpeur automatique de matches / entraînements
│   ├── predict.py               # Inférence directe sur n'importe quel fichier vidéo
│   └── visualize.py             # Export vidéo annotée avec squelette, traînée et HUD
├── requirements.txt             # Dépendances Python
├── GUIDE_DEVELOPPEMENT.md       # Guide complet de développement pas à pas
└── README.md
```

---

## ⚡ Installation et démarrage rapide

### 1. Installer les dépendances
```powershell
pip install -r requirements.txt
```

### 2. Exécuter la pipeline complète
```powershell
python src/pipeline.py
```

---

## 🛠️ Outils et Fonctionnalités

### 1. Découpage automatique de vidéos longues
Tu as filmé un entraînement ou un échange complet de plusieurs minutes ? Le script analyse l'accélération du poignet, repère les impacts de balle et découpe chaque geste en clip individuel :
```powershell
python src/auto_cutter.py data/sample_long_match.mp4
```
- Détecte les impacts via `scipy.signal.find_peaks`.
- Découpe un clip centré sur chaque geste (armé + impact + accompagnement).
- Prédit automatiquement le coup avec le modèle IA (`topspin`, `poussette`, `bloc`).
- Sauvegarde la courbe cinématique dans `data/processed/velocity_profile_*.png`.

### 2. Prédiction d'un coup
Affiche les métriques physiques calculées et les probabilités de chaque classe pour une vidéo :
```powershell
python src/predict.py data/raw_videos/topspin/topspin_01.mp4
```

### 3. Visualisation vidéo avec squelette incrusté
Génère une vidéo MP4 avec :
- Le squelette tracé sur le joueur
- L'angle du coude affiché dynamiquement
- La traînée lumineuse suivant le poignet
- Le bandeau HUD avec la prédiction et la confiance :
```powershell
python src/visualize.py data/raw_videos/topspin/topspin_01.mp4
```

---

## 📘 Guide de développement complet

Pour comprendre chaque étape ou enrichir le code par toi-même, consulte le guide détaillé :
👉 **[GUIDE_DEVELOPPEMENT.md](GUIDE_DEVELOPPEMENT.md)**
