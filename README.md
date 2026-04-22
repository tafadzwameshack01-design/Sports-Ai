# 🏀 NBA AI Prediction Engine

A self-improving Deep Q-Network (DQN) agent that predicts NBA game outcomes using live data from **SportRadar** and **ESPN**, with a **Claude-powered hyperparameter optimization loop**.

---

## Features

- **Live predictions** — real-time NBA game predictions with 60-second auto-refresh
- **Dual data sources** — SportRadar (paid, official stats) + ESPN (free, public API)
- **Double DQN** — 128→64→32 network with Huber loss, batch normalization, and dropout
- **Monte Carlo validation** — 1 000+ simulations to benchmark agent performance with 95% CI
- **Claude AI optimizer** — autonomous hyperparameter calibration loop with rollback
- **SQLite persistence** — all predictions, training sessions, and optimizer runs stored locally
- **5-page Streamlit dashboard** — live feed, training lab, analytics, data explorer, AI optimizer

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | Powers the AI optimizer and Claude integration |
| `SPORTRADAR_API_KEY` | ⚠️ Optional | Override the bundled trial key with a paid key |

The app ships with a **SportRadar trial API key** embedded (rate-limited to ~1 req/s, NBA only).  
ESPN data requires **no API key**.

---

## Setup — Local

```bash
# 1. Clone / unzip the project
cd sports_ai_dqn

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 5. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Setup — Streamlit Cloud

1. Push this directory to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `app.py` as the main file
4. Add secrets in the Streamlit Cloud dashboard:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   SPORTRADAR_API_KEY = "..."   # optional
   ```

> **Note:** TensorFlow adds ~500 MB to the deployment. Streamlit Cloud's free tier may time out on first cold start. Consider using a paid tier or Cloud Run for production.

---

## Pages

| Page | Description |
|---|---|
| 🏠 Home | KPI dashboard — accuracy, rewards, agent status |
| 🔴 Live Predictions | Real-time game predictions · score-change detection · on-the-fly training |
| ⚗️ Training Lab | Monte Carlo simulation · hyperparameter controls · replay buffer inspector |
| 📈 Analytics | Accuracy curves · reward history · full prediction log · MC history |
| 🔍 Data Explorer | Raw ESPN + SportRadar payloads · feature vectors · Q-value inspector · news |
| 🤖 AI Optimizer | Claude-powered self-improvement loop · rollback · before/after diff |

---

## DQN Architecture

```
Input (12 features, normalized [0,1])
  ↓
Dense(128, relu) → BatchNorm → Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Dense(32, relu)
  ↓
Dense(2, linear)  →  Q[away_wins], Q[home_wins]
```

**State vector (12 features):**

| # | Feature | Source |
|---|---|---|
| 0 | home_pts | SR boxscore / ESPN avg |
| 1 | home_ast | SR boxscore / ESPN avg |
| 2 | home_reb | SR boxscore / ESPN avg |
| 3 | home_to  | SR boxscore / ESPN avg |
| 4 | away_pts | SR boxscore / ESPN avg |
| 5 | away_ast | SR boxscore / ESPN avg |
| 6 | away_reb | SR boxscore / ESPN avg |
| 7 | away_to  | SR boxscore / ESPN avg |
| 8 | home_win_pct | SR / ESPN standings |
| 9 | away_win_pct | SR / ESPN standings |
| 10 | odds_spread | ESPN competition odds |
| 11 | bpi_delta | ESPN Power Index (home − away) |

---

## SportRadar Trial Key Limits

- ~1 request/second
- NBA only
- No historical game data beyond current season
- Upgrade to a paid key at [developer.sportradar.com](https://developer.sportradar.com)

---

## License

MIT
