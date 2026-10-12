# Guide de Déploiement Vercel - SpinTracker

Ce document explique comment déployer l'application **SpinTracker** en ligne avec **Vercel**.

---

## ⚠️ Pourquoi séparer le Front-end et le Back-end ?

- **Front-end (React / Vite) :** Parfait pour Vercel. Il est déployé sur un CDN mondial, ultra-rapide et gratuit.
- **Back-end (FastAPI + OpenCV + MediaPipe) :** 
  - Une vidéo de match (10 à 30 minutes) pèse entre 100 Mo et 2 Go.
  - Vercel limite la taille d'upload de ses Serverless Functions à **4.5 Mo** et a un timeout strict (10 à 60s).
  - De plus, les bibliothèques C++ de vision par ordinateur (`mediapipe`, `opencv-python`) dépassent le quota de taille de Vercel.
  - Le backend doit donc tourner sur un service adapté aux calculs lourds (gratuitement sur **Render**, **Railway**, ou via votre machine avec un tunnel sécurisé **ngrok** / **Cloudflare Tunnel**).

---

## 🚀 Étape 1 : Déployer le Front-end sur Vercel

1. Rendez-vous sur [vercel.com](https://vercel.com) et connectez-vous avec votre compte **GitHub**.
2. Cliquez sur **"Add New..."** > **"Project"**.
3. Sélectionnez le dépôt Git `SpinTracker`.
4. Dans les paramètres du projet Vercel :
   - **Root Directory :** Cliquez sur *Edit* et sélectionnez `web_dashboard/frontend`.
   - **Framework Preset :** `Vite` (détecté automatiquement).
   - **Build Command :** `npm run build`
   - **Output Directory :** `dist`
5. Dans la section **Environment Variables** :
   - Clé : `VITE_API_URL`
   - Valeur : L'URL de votre API backend (ex: `https://votre-backend.onrender.com` ou l'URL fournie par ngrok).
6. Cliquez sur **Deploy**. Votre site Web est en ligne avec une adresse HTTPS `https://spintracker.vercel.app` !

---

## ⚙️ Étape 2 : Héberger le Backend pour traiter les vidéos

Vous avez 2 options simples :

### Option A (100% cloud & gratuit) : Déployer sur Render.com
1. Créez un compte sur [render.com](https://render.com).
2. Cliquez sur **New +** > **Web Service**.
3. Connectez votre repo `SpinTracker`.
4. Configurez :
   - **Root Directory :** `web_dashboard/backend`
   - **Runtime :** `Python 3`
   - **Build Command :** `pip install -r requirements.txt -r ../../requirements.txt`
   - **Start Command :** `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Copiez l'URL HTTPS fournie par Render (ex: `https://spintracker-api.onrender.com`) et collez-la dans la variable `VITE_API_URL` sur Vercel.

### Option B (Utiliser la puissance de votre PC avec Ngrok - Zéro configuration de serveur)
Si vous voulez que vos calculs tournent sur votre GPU/CPU sans payer de serveur cloud :
1. Lancez le backend sur votre machine :
   ```powershell
   cd web_dashboard/backend
   python -m uvicorn main:app --reload
   ```
2. Ouvrez un tunnel public sécurisé avec [ngrok](https://ngrok.com) :
   ```powershell
   ngrok http 8000
   ```
3. Mettez l'URL `https://xxxx.ngrok-free.app` dans `VITE_API_URL` sur Vercel. N'importe qui sur Internet pourra utiliser votre site Vercel et le traitement s'exécutera sur votre machine !
