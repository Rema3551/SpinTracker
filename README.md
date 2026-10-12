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

**SpinTracker** est une solution complète de vision par ordinateur et d'intelligence artificielle dédiée à l'analyse de matches de tennis de table (vidéos de 10 à 30 minutes, 2 joueurs).

L'utilisateur importe sa vidéo de match sur l'application Web, indique quel joueur il est (côté initial et caractéristiques visuelles), et obtient automatiquement un **bilan statistique approfondi de sa performance** :
- **Volume et répartition des coups** : Nombre de **TopSpins**, **Poussettes** et **Blocs** joués.
- **Taux de réussite** : Pourcentage de coups réussis qui ont maintenu l'échange ou mis l'adversaire en difficulté.
- **Points gagnants (Winners)** : Nombre de points directement remportés grâce à un TopSpin, une Poussette ou un Bloc.
- **Fautes directes** : Nombre de coups ayant directement causé la perte du point.
- **Prise en compte des changements de côté** : Suivi continu du joueur malgré les permutations de côté à chaque set (et à 5 points à la manche décisive).

---

## 📊 Tableau de bord statistique cible

À l'issue de l'analyse, le joueur dispose d'un tableau d'efficacité tactique et technique :

| Coup technique | Total joués | Réussis (%) | Fautes directes | Points gagnants (Winners) |
| :--- | :---: | :---: | :---: | :---: |
| 🏓 **TopSpin** | **34** | **28 (82.4%)** | 6 (17.6%) | **11** |
| 🛡️ **Bloc** | **12** | **9 (75.0%)** | 3 (25.0%) | **4** |
| 🪓 **Poussette** | **18** | **17 (94.4%)** | 1 (5.6%) | **2** |
| **TOTAL** | **64** | **54 (84.4%)** | **10 (15.6%)** | **17** |

Le tableau est complété par :
- Un découpage statistique **set par set** (Set 1, Set 2, Set 3, etc.).
- La **timeline interactive de tous les échanges (rallyes)** du match.
- Le ratio d'efficacité offensive (Topspin) vs défensive/contrôle (Bloc, Poussette).

---

## 🏗️ Architecture du projet

```text
SpinTracker/
├── data/
│   ├── raw_videos/              # Extraits d'entraînement .mp4 par geste (topspin/, poussette/, bloc/)
│   ├── auto_clips/              # Clips de rallyes découpés automatiquement
│   └── processed/               # Modèles et métriques
│       ├── raw_landmarks.csv    # Coordonnées biomécaniques
│       ├── dataset.csv          # Jeu de données d'entraînement
│       └── model.joblib         # Modèle de classification Random Forest
├── src/
│   ├── match_analyzer.py        # Moteur d'analyse de match complet (2 joueurs, sets, winners)
│   ├── auto_cutter.py           # Détection des échanges et impacts raquette-balle
│   ├── 01_extract_landmarks.py  # Extraction squelette MediaPipe Pose
│   ├── 02_build_features.py     # Calcul des descripteurs physiques
│   ├── 03_train_model.py        # Entraînement du modèle de reconnaissance de coup
│   ├── pipeline.py              # Pipeline d'entraînement
│   ├── predict.py               # Inférence sur un clip isolé
│   └── visualize.py             # Rendu vidéo avec overlay squelette et HUD
├── web_dashboard/
│   ├── backend/                 # API FastAPI (upload vidéo, orchestration de l'analyse)
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/                # Application Web React + Vite (Dashboard moderne)
│       ├── src/
│       │   ├── App.jsx          # Dashboard, sélecteur de joueur, tableau de stats
│       │   └── App.css          # Design système premium (Dark Glassmorphism)
│       └── vercel.json          # Configuration de déploiement Vercel
├── requirements.txt             # Dépendances Python globales
├── GUIDE_DEVELOPPEMENT.md       # Guide complet de développement et de conception
└── README.md
```

---

## 🧠 Les 3 Piliers de l'Analyse de Match

### 1. Identification du joueur & Changements de côté (Player Re-ID)
Dans un match complet filmé en plan large ou 3/4 :
- Deux joueurs s'affrontent : l'un au **premier plan** (proche caméra), l'autre au **fond de court**.
- À chaque fin de set, les joueurs **permutent de côté** (ainsi qu'au 5e point de la manche décisive).
- Le moteur combine la position initiale déclarée avec un suivi d'apparence (histogramme de couleur du maillot / Re-ID) et la détection des temps morts entre sets pour suivre l'utilisateur tout au long du match.

### 2. Séquençage des échanges (Rally & Hit Engine)
- Détection des débuts et fins d'échange via le mouvement sur la table et la dynamique corporelle.
- Détection des impacts et attribution du coup au joueur actif.
- Classification du type de geste (**TopSpin**, **Poussette**, **Bloc**).

### 3. Logique de qualification du coup (Réussi vs Faute vs Winner)
- **Coup réussi** : Le coup est renvoyé sur la table adverse et l'échange continue.
- **Point gagnant (Winner)** : Le coup est le dernier coup valide joué par notre joueur et l'adversaire commet la faute (ou ne touche pas la balle).
- **Faute directe** : Le coup est le dernier joué par notre joueur avant l'arrêt de l'échange sans que la balle ne soit remise en jeu.

---

## ⚡ Guide Pas à Pas pour Tester en Local

### 1. Installation des dépendances (à faire une seule fois)

Ouvrez un terminal PowerShell à la racine du projet :
```powershell
# Dépendances IA & Vision par ordinateur
pip install -r requirements.txt

# Dépendances API Backend FastAPI
pip install -r web_dashboard/backend/requirements.txt

# Dépendances Frontend React
cd web_dashboard/frontend
npm install
cd ../..
```

---

### 2. Lancer l'Application Web (Méthode recommandée)

Pour faire tourner le site complet avec le tableau de bord, vous avez besoin de **2 terminaux ouverts en parallèle** :

#### 🔹 Terminal 1 : Lancer le Backend Python (IA & FastAPI)
```powershell
cd web_dashboard/backend
python -m uvicorn main:app --reload --port 8000
```
> Le terminal affiche : `INFO: Uvicorn running on http://127.0.0.1:8000`. Laissez ce terminal ouvert !

#### 🔹 Terminal 2 : Lancer l'Interface Web (React / Vite)
```powershell
cd web_dashboard/frontend
npm run dev
```
> Le terminal affiche : `Local: http://localhost:5173/`. Laissez également ce terminal ouvert !

#### 🔹 Utilisation dans votre navigateur :
1. Ouvrez votre navigateur sur **[http://localhost:5173](http://localhost:5173)**.
2. Choisissez votre position au 1er set :
   - 🧍‍♂️ **Joueur Premier Plan** (bas de l'écran)
   - 🏃‍♂️ **Joueur Fond de Court** (haut de l'écran)
3. Cliquez sur **"Choisir votre vidéo de match"** et sélectionnez votre fichier `.mp4`.
4. Cliquez sur **⚡ Lancer l'Analyse du Match** !
5. Le tableau de bord s'affiche avec le bilan complet (**TopSpins, Blocs, Poussettes**, réussites %, fautes et points gagnants).

> [!TIP]
> **En cas de modification du code Python :** 
> Allez dans le Terminal 1, faites `Ctrl + C` pour stopper le serveur, puis relancez `python -m uvicorn main:app --reload --port 8000`.

---

### 3. Tester directement en ligne de commande (Sans le site Web)

Si vous voulez analyser un match directement depuis votre terminal sans lancer le navigateur :

```powershell
# Analyse de match (joueur au premier plan) :
python src/match_analyzer.py chemin_vers_votre_match.mp4 --player-side near

# Analyse de match (joueur au fond de court) :
python src/match_analyzer.py chemin_vers_votre_match.mp4 --player-side far
```
Le terminal affichera directement le tableau des statistiques, le décompte des points et le taux de réussite.

---

## 📘 Guide de Développement Complet

Pour comprendre en détail l'algorithme d'analyse de match, la détection des changements de côté et la logique de scoring :
👉 **[GUIDE_DEVELOPPEMENT.md](GUIDE_DEVELOPPEMENT.md)**

