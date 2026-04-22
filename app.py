import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="NBA AI Prediction Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME         = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
SPORTRADAR_API_KEY = os.environ.get("SPORTRADAR_API_KEY", "")

# ── API key guard ──────────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    st.error("⚠️  **ANTHROPIC_API_KEY not set.**  Add it to `.env` or Streamlit Secrets.")
    st.stop()

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "db_ready":          False,
    "agent_ready":       False,
    "improver_ready":    False,
    "stop_flag":         False,
    "iteration_log":     [],
    "optimizer_scores":  [],
    "previous_hp":       None,
    "mc_results":        None,
    "live_games":        [],
    "standings":         {},
    "bpi":               {},
    "score_cache":       {},
    "cycle_count":       0,
    "last_heavy_ts":     0.0,
    "lessons_display":   [],
    "retrain_loss_log":  [],
    "background_losses": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── One-time initialisation ────────────────────────────────────────────────────
if not st.session_state.db_ready:
    from utils.database import init_db
    init_db()
    st.session_state.db_ready = True

if not st.session_state.agent_ready:
    from utils.dqn_agent import DQNAgent
    agent = DQNAgent()
    agent.load()
    st.session_state.agent       = agent
    st.session_state.agent_ready = True

if not st.session_state.improver_ready:
    import anthropic
    from utils.self_improver import AutonomousImprover
    _client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    _improver = AutonomousImprover(st.session_state.agent, _client)
    st.session_state.improver       = _improver
    st.session_state.anthropic_client = _client
    st.session_state.improver_ready = True

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'Space Mono', monospace; }
[data-testid="stMetricValue"]  { font-family: 'Space Mono', monospace; font-size:1.55rem !important; }
[data-testid="stMetricLabel"]  { font-size:0.78rem; letter-spacing:0.05em; text-transform:uppercase; }
div[data-testid="stSidebarContent"]  { background:#0a0a14 !important; }
.stProgress > div > div > div { background:linear-gradient(90deg,#6c63ff,#00d4ff) !important; }
.block-container { padding-top:1.5rem !important; }
.kpi-card { background:linear-gradient(135deg,#0f0f1a,#1a1a2e);
            border:1px solid #2d2d4e; border-radius:10px;
            padding:1rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Auto-fetch fresh data (throttled to once per 3 min) ───────────────────────
now = time.time()
if (now - st.session_state.last_heavy_ts) > 180:
    with st.spinner("🔄 Loading live NBA data…"):
        try:
            from utils.espn_api import fetch_espn_scoreboard, fetch_espn_standings, fetch_espn_power_index
            games    = fetch_espn_scoreboard()
            standings = fetch_espn_standings()
            bpi       = fetch_espn_power_index()
            st.session_state.live_games    = games
            st.session_state.standings     = standings
            st.session_state.bpi           = bpi
            st.session_state.last_heavy_ts = now
        except Exception as e:
            st.warning(f"Data fetch warning: {e}")

# ── Pull metrics ───────────────────────────────────────────────────────────────
from utils.database import get_model_metrics
agent   = st.session_state.agent
metrics = get_model_metrics()
games   = st.session_state.live_games
improver = st.session_state.improver

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏀 NBA AI Prediction Engine")
st.caption(
    f"Double DQN · Priority Experience Replay · Autonomous Self-Improvement  ·  "
    f"{datetime.now().strftime('%A %B %d, %Y  %H:%M')}"
)
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Games Today",     len(games))
with k2:
    live_count = sum(1 for g in games if g.get("is_live"))
    st.metric("🔴 Live Now",     live_count)
with k3:
    st.metric("Predictions",     metrics["total"],
              delta=f"+{metrics['today']} today")
with k4:
    st.metric("Accuracy",        f"{metrics['accuracy']:.1f}%",
              delta=f"{metrics['accuracy_delta']:+.1f}%")
with k5:
    st.metric("Agent ε",         f"{agent.epsilon:.4f}")
with k6:
    st.metric("Lessons Learned", len(improver.lessons))

st.divider()

# ── Today's Games (shown on homepage automatically) ───────────────────────────
if games:
    st.subheader(f"📅 Today's {len(games)} Games")
    from utils.features import build_state
    standings = st.session_state.standings
    bpi       = st.session_state.bpi

    cols_per_row = 3
    rows = [games[i:i+cols_per_row] for i in range(0, len(games), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, game in zip(cols, row):
            with col:
                try:
                    state       = build_state(game, standings, bpi)
                    action, conf = agent.act_greedy(state)
                    pred_team   = game["home_team"] if action == 1 else game["away_team"]
                    h_score     = game.get("home_score", 0)
                    a_score     = game.get("away_score", 0)
                    score_str   = f"{a_score:.0f}–{h_score:.0f}" if (h_score or a_score) else "TBD"
                    status_icon = "🔴" if game.get("is_live") else ("✅" if game.get("is_final") else "⏳")

                    h_col = game.get("home_color", "6c63ff")
                    a_col = game.get("away_color", "00d4ff")
                    conf_bar_w = int(conf)

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);
                                border:1px solid #2d2d4e;border-radius:12px;
                                padding:0.85rem 1rem;margin-bottom:0.5rem">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#888;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase">
                          {status_icon} {game.get('status_desc','Scheduled')}
                        </span>
                        <span style="color:#aaa;font-size:0.7rem">{game.get('source','espn').upper()}</span>
                      </div>
                      <div style="font-weight:700;font-size:0.95rem;margin:0.4rem 0;color:#e8e8f0">
                        <span style="color:#{a_col}">{game['away_abbr']}</span>
                        <span style="color:#888;margin:0 0.3rem">@</span>
                        <span style="color:#{h_col}">{game['home_abbr']}</span>
                      </div>
                      <div style="font-size:1.1rem;font-weight:700;color:#fff;margin:0.2rem 0">{score_str}</div>
                      <div style="margin-top:0.5rem;background:#1e1e3a;border-radius:6px;padding:0.45rem 0.6rem">
                        <div style="font-size:0.72rem;color:#aaa;margin-bottom:0.2rem">🏆 {pred_team}</div>
                        <div style="background:#2d2d4e;border-radius:3px;height:4px;overflow:hidden">
                          <div style="background:linear-gradient(90deg,#6c63ff,#00d4ff);
                                      width:{conf_bar_w}%;height:100%"></div>
                        </div>
                        <div style="font-size:0.68rem;color:#888;margin-top:0.15rem">{conf:.1f}% confidence</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.warning(f"Error: {ex}")
else:
    st.info("📭 No games on today's ESPN scoreboard — this may be an off-day. Check the **🔍 Data Explorer** to browse historical data.")

st.divider()

# ── System status + model ──────────────────────────────────────────────────────
sa, sb, sc = st.columns(3)

with sa:
    st.subheader("🔌 Data Sources")
    sr_label = "⚠️ Trial key (rate-limited)" if not SPORTRADAR_API_KEY else "✅ Custom key"
    st.info(f"**SportRadar:** {sr_label}")
    st.success("**ESPN API:** ✅ Live — no auth")
    st.success("**Anthropic Claude:** ✅ Connected")
    st.caption(f"Last refresh: {datetime.fromtimestamp(st.session_state.last_heavy_ts).strftime('%H:%M:%S') if st.session_state.last_heavy_ts else 'pending'}")

with sb:
    st.subheader("🧠 Model Architecture")
    model_file = Path("dqn_sports_model.npy")
    if model_file.exists():
        mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
        st.success(f"**Weights:** ✅ Loaded  ·  saved {mtime.strftime('%H:%M')}")
    else:
        st.warning("**Weights:** ⚠️ New session — no saved weights yet")
    st.caption("Net   : 12 → 256 → 128 → 64 → 2 (pure NumPy)")
    st.caption("Algo  : Double DQN + Priority Experience Replay")
    st.caption("Loss  : Huber  |  Optim: Adam  |  Act: LeakyReLU")
    st.caption("Self-improve: Claude analysis every 5 misses")

with sc:
    st.subheader("📊 Live Agent Stats")
    sd = agent.get_state_dict()
    buf_pct = sd["buffer_size"] / 10_000
    st.progress(buf_pct, text=f"Replay buffer: {sd['buffer_size']:,} / 10 000")
    if metrics["total"] > 0:
        st.progress(min(metrics["accuracy"] / 100, 1.0),
                    text=f"Accuracy: {metrics['accuracy']:.1f}%")
    acc_col = "green" if improver.accuracy_rate() >= 0.55 else "red"
    st.markdown(f"Session accuracy: :{acc_col}[**{improver.accuracy_rate()*100:.1f}%**]")
    st.caption(f"Misses: {improver.total_misses}  |  Correct: {improver.total_correct}")
    st.caption(f"Lessons generated: {len(improver.lessons)}")

st.divider()

# ── Latest AI lesson ───────────────────────────────────────────────────────────
if improver.last_analysis:
    la = improver.last_analysis
    with st.expander("🎓 Latest AI Self-Improvement Lesson", expanded=True):
        lc1, lc2 = st.columns([2, 1])
        with lc1:
            st.markdown(f"**Bias detected:** {la.get('bias_detected','')}")
            st.markdown(f"**Lesson:** {la.get('lesson','')}")
            st.markdown(f"**Reasoning:** {la.get('reasoning','')}")
        with lc2:
            imp = la.get('predicted_improvement', 0)
            st.metric("Expected Δ accuracy", f"+{imp*100:.1f}%")
            st.caption(f"At {la.get('timestamp','')[:16]}")

st.divider()

# ── Navigation ─────────────────────────────────────────────────────────────────
st.subheader("🗺️ Navigate")
n1, n2, n3, n4, n5 = st.columns(5)
with n1:
    st.page_link("pages/1_Live_Predictions.py",     label="🔴 Live Predictions")
    st.caption("60s auto-refresh · live training · score tracking")
with n2:
    st.page_link("pages/2_Training_Lab.py",          label="⚗️ Training Lab")
    st.caption("Monte Carlo · replay buffer · hyperparameters")
with n3:
    st.page_link("pages/3_Performance_Analytics.py", label="📈 Analytics")
    st.caption("Accuracy curves · reward history · lessons log")
with n4:
    st.page_link("pages/4_Data_Explorer.py",         label="🔍 Data Explorer")
    st.caption("Raw API data · feature vectors · Q-values")
with n5:
    st.page_link("pages/5_AI_Optimizer.py",          label="🤖 AI Optimizer")
    st.caption("Claude self-improvement · hyperparameter tuning")
