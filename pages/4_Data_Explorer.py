import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import init_db
from utils.features  import FEATURE_NAMES, build_state, get_feature_importance

st.set_page_config(page_title="Data Explorer · NBA AI", page_icon="🔍", layout="wide")

if "db_ready" not in st.session_state:
    init_db(); st.session_state.db_ready = True
if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    a = DQNAgent(); a.load(); st.session_state.agent = a

for key, default in [
    ("explorer_games",    None),
    ("explorer_stand",    None),
    ("explorer_bpi",      None),
    ("explorer_sr_sched", None),
    ("explorer_teams",    None),
    ("explorer_news",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent = st.session_state.agent
st.markdown("## 🔍 Data Explorer")
st.caption("Browse raw ESPN & SportRadar data · inspect feature vectors · query Q-values")
st.divider()

# ── Auto-load ESPN data once per session ───────────────────────────────────────
if st.session_state.explorer_games is None:
    with st.spinner("Auto-loading ESPN scoreboard…"):
        try:
            from utils.espn_api import fetch_espn_scoreboard, fetch_espn_standings, fetch_espn_power_index
            st.session_state.explorer_games = fetch_espn_scoreboard()
            st.session_state.explorer_stand = fetch_espn_standings()
            st.session_state.explorer_bpi   = fetch_espn_power_index()
        except Exception as e:
            st.warning(f"Auto-load: {e}")
            st.session_state.explorer_games = []
            st.session_state.explorer_stand = {}
            st.session_state.explorer_bpi   = {}

tab_espn, tab_sr, tab_fv, tab_qv, tab_news = st.tabs([
    "📺 ESPN Live", "📡 SportRadar", "🧮 Feature Vectors", "🧠 Q-Values", "📰 News"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_espn:
    ec1, ec2 = st.columns([2,1])
    with ec2:
        if st.button("🔄 Reload ESPN Data", key="btn_reload_espn", use_container_width=True):
            with st.spinner("Fetching…"):
                try:
                    from utils.espn_api import fetch_espn_scoreboard, fetch_espn_standings, fetch_espn_power_index
                    st.session_state.explorer_games = fetch_espn_scoreboard()
                    st.session_state.explorer_stand = fetch_espn_standings()
                    st.session_state.explorer_bpi   = fetch_espn_power_index()
                    st.toast("ESPN data refreshed!", icon="📺")
                except Exception as e:
                    st.error(str(e))
    with ec1:
        view = st.radio("View mode", ["Table","Raw JSON"], horizontal=True, key="espn_view")

    games     = st.session_state.explorer_games or []
    standings = st.session_state.explorer_stand or {}
    bpi       = st.session_state.explorer_bpi   or {}

    st.caption(f"**{len(games)} games** · **{len(standings)} teams in standings** · **{len(bpi)} BPI entries**")

    if games:
        if view == "Table":
            df_g = pd.DataFrame(games)
            display = [c for c in ["away_team","away_abbr","home_team","home_abbr",
                                   "away_score","home_score","status_desc","odds_spread",
                                   "over_under","home_win_pct","away_win_pct","is_live","is_final","source"]
                       if c in df_g.columns]
            st.dataframe(df_g[display], use_container_width=True, hide_index=True)
        else:
            st.json(games[:5])  # show first 5 to avoid clutter

    if standings:
        st.subheader("Standings (Win %)")
        df_s = pd.DataFrame(standings.items(), columns=["Team","Win%"]).sort_values("Win%", ascending=False)
        sc1, sc2 = st.columns([1,2])
        with sc1:
            st.dataframe(df_s, hide_index=True, use_container_width=True)
        with sc2:
            st.bar_chart(df_s.set_index("Team")["Win%"], use_container_width=True)

    if bpi:
        st.subheader("ESPN Power Index (BPI)")
        df_bpi = pd.DataFrame(bpi.items(), columns=["Team","BPI"]).sort_values("BPI", ascending=False)
        st.bar_chart(df_bpi.set_index("Team")["BPI"], use_container_width=True)

    # Teams
    if st.button("📋 Load All NBA Teams", key="btn_teams"):
        with st.spinner():
            from utils.espn_api import fetch_espn_teams
            st.session_state.explorer_teams = fetch_espn_teams()

    if st.session_state.explorer_teams:
        st.subheader("All NBA Teams")
        st.dataframe(pd.DataFrame(st.session_state.explorer_teams),
                     hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_sr:
    sr1, sr2 = st.columns([2,1])
    with sr2:
        if st.button("🔄 Fetch SR Schedule", key="btn_sr", use_container_width=True):
            with st.spinner():
                try:
                    from utils.sportradar import fetch_sr_todays_schedule, fetch_sr_standings
                    st.session_state.explorer_sr_sched = {
                        "schedule": fetch_sr_todays_schedule(),
                        "standings": fetch_sr_standings(),
                    }
                    st.toast("SportRadar loaded!", icon="📡")
                except Exception as e:
                    st.error(str(e))
    with sr1:
        st.caption("⚠️ Trial API key — rate-limited to ~1 req/s · errors expected on fast calls")

    if st.session_state.explorer_sr_sched:
        sr_data = st.session_state.explorer_sr_sched
        sched   = sr_data.get("schedule", [])
        stand   = sr_data.get("standings", {})
        st.caption(f"SR schedule: {len(sched)} games  |  standings: {len(stand)} teams")

        if sched:
            df_sr = pd.DataFrame(sched)
            st.dataframe(df_sr, hide_index=True, use_container_width=True)

        if stand:
            st.subheader("SR Standings")
            df_ss = pd.DataFrame(stand.items(), columns=["Abbr","Win%"]).sort_values("Win%", ascending=False)
            st.dataframe(df_ss, hide_index=True, use_container_width=True)

        st.subheader("Manual Boxscore Lookup")
        gid = st.text_input("Game ID", key="ti_gid", placeholder="paste SR game id…")
        if st.button("Fetch Boxscore", key="btn_box") and gid:
            with st.spinner():
                from utils.sportradar import fetch_sr_boxscore
                box = fetch_sr_boxscore(gid.strip())
                st.json(box)
    else:
        st.info("Click **Fetch SR Schedule** to load SportRadar data.")

# ══════════════════════════════════════════════════════════════════════════════
with tab_fv:
    st.subheader("Feature Vector Inspector")
    games = st.session_state.explorer_games or []
    stand = st.session_state.explorer_stand or {}
    bpi   = st.session_state.explorer_bpi   or {}

    if not games:
        st.info("Load ESPN data in the 📺 ESPN Live tab first.")
    else:
        options = [f"{g.get('away_abbr','?')} @ {g.get('home_abbr','?')}" for g in games]
        sel     = st.selectbox("Select game", options, key="sel_fv_game")
        idx     = options.index(sel)
        game    = games[idx]

        try:
            state = build_state(game, stand, bpi)
            st.success("✅ Feature vector built")

            fv1, fv2 = st.columns([1,2])
            with fv1:
                df_fv = pd.DataFrame({
                    "Feature":    FEATURE_NAMES,
                    "Normalized": [round(float(v),4) for v in state],
                })
                st.dataframe(df_fv, hide_index=True, use_container_width=True)
            with fv2:
                st.bar_chart(pd.DataFrame({"value": state}, index=FEATURE_NAMES),
                             use_container_width=True)

            # Show raw game dict used for feature building
            with st.expander("Raw game dict"):
                st.json({k: v for k,v in game.items()
                         if k not in ("state","home_color","away_color")})
        except Exception as e:
            st.error(f"Feature build error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_qv:
    st.subheader("Q-Value Inspector")
    mode = st.radio("Input source", ["Live game","Custom sliders"], horizontal=True, key="qv_mode")
    state_qv: np.ndarray | None = None

    if mode == "Live game":
        games = st.session_state.explorer_games or []
        if not games:
            st.info("Load ESPN data first.")
        else:
            opts = [f"{g.get('away_abbr','?')} @ {g.get('home_abbr','?')}" for g in games]
            sel  = st.selectbox("Game", opts, key="sel_qv")
            try:
                state_qv = build_state(
                    games[opts.index(sel)],
                    st.session_state.explorer_stand or {},
                    st.session_state.explorer_bpi   or {},
                )
            except Exception as e:
                st.error(str(e))
    else:
        vals = []
        qc1, qc2 = st.columns(2)
        for i, name in enumerate(FEATURE_NAMES):
            col = qc1 if i < 6 else qc2
            with col:
                vals.append(st.slider(name, 0.0, 1.0, 0.5, 0.01, key=f"sl_qv_{name}"))
        state_qv = np.array(vals, dtype=np.float32)

    if state_qv is not None:
        from utils.numpy_dqn import softmax
        q      = agent.get_q_values(state_qv)
        probs  = softmax(q)
        action = int(np.argmax(q))
        lbl    = "🏠 Home wins" if action == 1 else "✈️ Away wins"

        qr1,qr2,qr3 = st.columns(3)
        with qr1: st.metric("Prediction",    lbl)
        with qr2: st.metric("Confidence",    f"{float(np.max(probs))*100:.1f}%")
        with qr3: st.metric("Q-value spread",f"{abs(float(q[0]-q[1])):.4f}")

        df_q = pd.DataFrame({
            "Action":      ["Away wins (0)", "Home wins (1)"],
            "Q-Value":     [round(float(q[0]),4), round(float(q[1]),4)],
            "Probability": [round(float(probs[0]),4), round(float(probs[1]),4)],
        })
        st.dataframe(df_q, hide_index=True, use_container_width=True)
        st.bar_chart(df_q.set_index("Action")["Q-Value"], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.subheader("ESPN NBA News")
    if st.session_state.explorer_news is None:
        with st.spinner("Loading headlines…"):
            try:
                from utils.espn_api import fetch_espn_news
                st.session_state.explorer_news = fetch_espn_news(15)
            except Exception as e:
                st.warning(str(e))
                st.session_state.explorer_news = []

    if st.button("🔄 Refresh News", key="btn_refresh_news"):
        with st.spinner():
            from utils.espn_api import fetch_espn_news
            st.session_state.explorer_news = fetch_espn_news(15)

    news = st.session_state.explorer_news or []
    if news:
        for a in news:
            with st.container():
                st.markdown(f"**{a.get('headline','')}**")
                if a.get("description"):
                    st.caption(a["description"])
                meta = []
                if a.get("published"):   meta.append(a["published"][:10])
                if a.get("source"):      meta.append(a["source"])
                if meta:
                    st.caption("  ·  ".join(meta))
                if a.get("url"):
                    st.markdown(f"[Read full article →]({a['url']})")
                st.markdown("---")
    else:
        st.info("No news loaded.")
