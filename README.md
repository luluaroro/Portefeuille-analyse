# 📊 Portefeuille & Analyse — Streamlit App

Application Streamlit pour :
- gérer un portefeuille (transactions Buy/Sell)
- afficher positions, PRU, P/L
- analyser un investissement via ISIN → Ticker (Wikidata + cache)
- afficher un graphique interactif (Lightweight Charts)

## ✅ Fonctionnalités
- Ajout et historique des transactions (CSV local)
- Calcul des positions & P/L
- Résolution ISIN → ticker (Wikidata + cache local)
- Graphique interactif (area)
- Analyse fondamentale simple (uniquement ACTION pour l’instant)
- Option IA : Ollama (local) ou OpenAI (si configuré)

## 🧱 Tech Stack
- Python
- Streamlit
- yfinance
- httpx
- Lightweight Charts (bundle local)
- Ollama (optionnel)
- OpenAI API (optionnel)

## ▶️ Lancer le projet en local

### 1) Installer les dépendances
```bash
pip install -r requirements.txt
