# Portfolio & Investment Analyzer (Streamlit)

Application Streamlit pour gérer un portefeuille (transactions, positions, P/L) et analyser un actif via ISIN → ticker, avec graphique interactif.

## Features
- Transactions + positions + P/L
- Résolution ISIN (Wikidata / OpenFIGI / FMP)
- Graphique interactif (Lightweight Charts)
- Analyse fondamentale (yfinance) + scoring
- Option IA (Ollama / OpenAI)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

---

## Partie 2 — Créer le repo sur GitHub
1) Va sur GitHub → **New repository**
2) Repo name : par ex `portfolio-analyzer`
3) Choisis **Public** (ou Private si tu préfères)
4) ✅ Coche rien (pas de README, pas de .gitignore, pas de licence) car tu les as déjà
5) Clique **Create repository**

➡️ GitHub te donne une URL du style :
`https://github.com/luluaroro/Portefeuille-analyse.git`

---

## Partie 3 — Envoyer ton projet sur GitHub (push)
### 3.1 Ouvre un terminal dans ton dossier projet
Assure-toi d’être dans le bon dossier :
- tu dois voir `app.py` quand tu fais `ls` (Mac/Linux) ou `dir` (Windows)

### 3.2 Lance exactement ces commandes
Remplace `TON_USER` et `portfolio-analyzer` par ton lien GitHub :

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/luluaroro/Portefeuille-analyse.git
git push -u origin main