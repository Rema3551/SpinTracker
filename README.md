# 🏓 SpinTracker

> Reconnaissance et analyse biomécanique des gestes techniques de tennis de table à partir de vidéos par ordinateur.

---

## 📖 Sommaire

- [Présentation du projet](#-présentation-du-projet)
- [Architecture de la pipeline](#-architecture-de-la-pipeline)
- [Guide de développement complet](#-guide-de-développement-complet)
- [Installation et démarrage rapide](#-installation-et-démarrage-rapide)

---

## 🎯 Présentation du projet

SpinTracker a pour but de classifier automatiquement des coups techniques de tennis de table (**Topspin**, **Poussette**, **Bloc**) à partir d'extraits vidéo enregistrés avec une caméra classique (smartphone ou webcam).

Le système repose sur 3 piliers :
1. **Vision par ordinateur** : Détection des repères du corps humain (épaules, coude, poignet, hanches) avec **MediaPipe Pose**.
2. **Ingénierie de caractéristiques** : Calcul de descripteurs biomécaniques (vitesse maximale du poignet, dénivelé vertical, angles articulaires du bras).
3. **Apprentissage automatique** : Classification supervisée avec un modèle **Random Forest**.

---

## 🏗️ Architecture de la pipeline

```text
SpinTracker/
├── data/
│   ├── raw_videos/              # Extraits .mp4 classés par dossier (topspin/, poussette/, bloc/)
│   └── processed/               # Données extraites et modèles (raw_landmarks.csv, dataset.csv, model.joblib)
├── src/
│   ├── 01_extract_landmarks.py  # Étape 1 : Extraction frame par frame via MediaPipe Pose
│   ├── 02_build_features.py     # Étape 2 : Calcul des métriques physiques et agrégation
│   ├── 03_train_model.py        # Étape 3 : Entraînement Random Forest et métriques
│   └── pipeline.py              # Orchestrateur qui exécute 01 -> 02 -> 03
├── requirements.txt             # Dépendances Python
├── GUIDE_DEVELOPPEMENT.md       # Guide complet pour programmer le projet pas à pas
└── README.md
```

---

## 📘 Guide de développement complet

Pour programmer l'ensemble du projet par toi-même, consulte le guide détaillé :
👉 **[GUIDE_DEVELOPPEMENT.md](GUIDE_DEVELOPPEMENT.md)**

Il explique en détail :
- La théorie biomécanique et la géométrie des coups de tennis de table.
- Le fonctionnement et le système de coordonnées de MediaPipe Pose.
- Les calculs vectoriels (angles articulaires via produit scalaire, vitesses euclidiennes).
- La préparation du jeu de données pour le Machine Learning.
- L'évaluation du modèle et les pistes pour aller plus loin.

---

## ⚡ Installation et démarrage rapide

### 1. Cloner et installer les dépendances
```powershell
pip install -r requirements.txt
```

### 2. Ajouter tes vidéos
Dépose tes clips vidéo dans `data/raw_videos/<coup>/` :
- `data/raw_videos/topspin/`
- `data/raw_videos/poussette/`
- `data/raw_videos/bloc/`

### 3. Exécuter la pipeline
```powershell
python src/pipeline.py
```
