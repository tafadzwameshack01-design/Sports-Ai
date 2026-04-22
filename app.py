import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="NBA AI Prediction Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── Ensure project root is on sys.path so pages/ can import utils/ ────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Module-level constants ─────────────────────────────────────────────────────
MODEL_NAME          = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS  = 1024
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
SPORTRADAR_API_KEY  = os.environ.get("SPORTRADAR_API_KEY", "")

# ── API key guard ──────────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    st.error(
        "⚠️  **ANTHROPIC_API_KEY not set.**  "
        "Add it to `.env` or Streamlit Secrets before launching."
    )
    st.stop()

# ── Session state initialisation (all keys declared here) ─────────────────────
_DEFAULTS: dict = {
    "db_initialized":    False,
    "agent_loaded":      False,
    "last_refresh_ts":   0.0,
    "stop_flag":         False,
    "iteration_log":     [],
    "optimizer_scores":  [],
    "previous_agent_state": None,
    "mc_results":        None,
    "live_games":        [],
    "sr_standings":      {},
    "espn_bpi":          {},
    "score_cache":       {},
    "cycle_count":       0,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Initialise database once per session ──────────────────────────────────────
if not st.session_state.db_initialized:
    from utils.database import init_db
    init_db()
    st.session_state.db_initialized = True

# ── Load DQN agent once per session ───────────────────────────────────────────
if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    _agent = DQNAgent(state_size=12, action_size=2)
    _agent.load()
    st.session_state.agent       = _agent
    st.session_state.agent_loaded = True

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, .stMetric label {
        font-family: 'Space Mono', monospace;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .status-badge-ok   { color: #00ff99; font-weight: 700; }
    .status-badge-warn { color: #ffcc00; font-weight: 700; }
    .status-badge-err  { color: #ff4444; font-weight: 700; }
    [data-testid="stMetricValue"]        { font-family: 'Space Mono', monospace; font-size: 1.6rem; }
    [data-testid="stMetricDelta"]        { font-size: 0.8rem; }
    div[data-testid="stSidebarContent"]  { background: #0a0a14; }
    .stProgress > div > div > div       { background: linear-gradient(90deg, #6c63ff, #00d4ff); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pull live metrics for dashboard ───────────────────────────────────────────
from utils.database import get_model_metrics

agent   = st.session_state.agent
metrics = get_model_metrics()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏀 NBA AI Prediction Engine")
st.caption(
    f"DQN Self-Improving Agent  ·  SportRadar + ESPN  ·  "
    f"{datetime.now().strftime('%A %B %d, %Y  %H:%M')}"
)
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Total Predictions",
        metrics["total"],
        delta=f"+{metrics['today']} today",
    )
with c2:
    acc_delta = metrics["accuracy_delta"]
    st.metric(
        "Win/Loss Accuracy",
        f"{metrics['accuracy']:.1f}%",
        delta=f"{acc_delta:+.1f}% vs last week",
        delta_color="normal",
    )
with c3:
    eps_delta = round(agent.epsilon - 1.0, 3)
    st.metric(
        "Exploration ε",
        f"{agent.epsilon:.4f}",
        delta=f"{eps_delta:+.3f} from init",
        delta_color="inverse",
    )
with c4:
    mem_pct = int(len(agent.memory) / 50)
    st.metric(
        "Replay Buffer",
        f"{len(agent.memory):,}",
        delta=f"/ 5 000 capacity",
    )
with c5:
    st.metric(
        "Training Steps",
        f"{agent.steps:,}",
    )

st.divider()

# ── System status + model info ─────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("🔌 API Status")
    sr_ok  = bool(SPORTRADAR_API_KEY)
    sr_lbl = "✅ Custom key" if sr_ok else "⚠️ Trial key (default)"
    st.info(f"**SportRadar:** {sr_lbl}")
    st.success("**ESPN API:** ✅ Public — no auth required")
    st.success("**Anthropic Claude:** ✅ Connected")

with col_b:
    st.subheader("🧠 Model Status")
    model_path = Path("dqn_sports_model.keras")
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        st.success("**Weights file:** ✅ Loaded from disk")
        st.caption(f"Last saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("**Weights file:** ⚠️ New session — training required")
    st.caption("Architecture : 128 → BN → Dropout → 64 → 32 → 2")
    st.caption("Loss / Opt   : Huber / Adam  ·  Double DQN")
    st.caption("State vector : 12 features, normalized to [0, 1]")

with col_c:
    st.subheader("📊 Performance Snapshot")
    if metrics["total"] > 0:
        st.progress(
            min(metrics["accuracy"] / 100, 1.0),
            text=f"Accuracy: {metrics['accuracy']:.1f}%",
        )
        reward_val = metrics["avg_reward"]
        colour     = "green" if reward_val >= 0 else "red"
        st.markdown(f"Avg Reward: :{colour}[**{reward_val:+.4f}**]")
        st.caption(f"Correct: {metrics['correct']} / {metrics['total']} games")
    else:
        st.info(
            "No prediction history yet.  "
            "Visit **🔴 Live Predictions** to start fetching games."
        )

st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
st.subheader("🗺️ Navigate")
n1, n2, n3, n4, n5 = st.columns(5)

with n1:
    st.page_link("pages/1_Live_Predictions.py", label="🔴 Live Predictions")
    st.caption("Real-time game predictions · 60 s auto-refresh · on-the-fly training")

with n2:
    st.page_link("pages/2_Training_Lab.py", label="⚗️ Training Lab")
    st.caption("Monte Carlo · DQN hyper-params · replay buffer control")

with n3:
    st.page_link("pages/3_Performance_Analytics.py", label="📈 Analytics")
    st.caption("Accuracy curves · reward history · prediction log")

with n4:
    st.page_link("pages/4_Data_Explorer.py", label="🔍 Data Explorer")
    st.caption("Raw ESPN + SportRadar payloads · feature vectors")

with n5:
    st.page_link("pages/5_AI_Optimizer.py", label="🤖 AI Optimizer")
    st.caption("Claude-powered self-improvement loop · hyperparameter calibration")
