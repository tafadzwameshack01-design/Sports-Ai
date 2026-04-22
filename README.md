# 🏀 NBA AI Prediction Engine v2

Self-improving Double DQN agent for NBA game prediction.  
**Zero TensorFlow/PyTorch** — pure NumPy neural network, works on Python 3.14.

---

## Setup

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Add `ANTHROPIC_API_KEY` in **App Settings → Secrets**.  
No Python version override needed — runs on the default Python 3.14 environment.

## Architecture

- **Net**: 12 → 256 → 128 → 64 → 2 (pure NumPy, LeakyReLU, Adam)
- **Algorithm**: Double DQN + Priority Experience Replay
- **Self-improvement**: Claude analyzes missed predictions every 5 misses
- **Data**: ESPN (free) + SportRadar (trial key bundled)
- **State**: 12 normalized features (pts/ast/reb/to × 2 teams, win%, odds, BPI)

## Pages

| Page | Purpose |
|---|---|
| 🏀 Home | Live game cards + KPIs |
| 🔴 Live Predictions | 60s auto-refresh, score tracking, on-the-fly training |
| ⚗️ Training Lab | Monte Carlo, synthetic training, hyperparams, feature importance |
| 📈 Analytics | Accuracy/reward curves, prediction log, AI lessons |
| 🔍 Data Explorer | Raw ESPN/SR data, feature vectors, Q-value inspector |
| 🤖 AI Optimizer | Claude optimization loop with rollback and diff |
